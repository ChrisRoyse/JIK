"""Generation-pinned ZIP directory reads without downloading archive payloads.

PC-07/13: bounded metadata ranges, cached for the immutable object generation.
No sparse whole-archive file, extraction, model input, or ignored-Range fallback.
"""
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request
import zipfile

MAX_METADATA_READ = 8 * 1024 * 1024
MAX_METADATA_TOTAL = 16 * 1024 * 1024


def require(ok, code):
    if not ok:
        raise ValueError("MNLM_RANGE_ZIP_"+code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def bound_url(identity):
    require(set(identity) == {"url", "generation", "etag", "bytes"}, "IDENTITY")
    url = urllib.parse.urlsplit(identity["url"])
    require(url.scheme == "https" and url.netloc == "storage.googleapis.com"
            and not url.query and not url.fragment
            and isinstance(identity["generation"], str) and identity["generation"].isdigit()
            and isinstance(identity["etag"], str) and identity["etag"]
            and type(identity["bytes"]) is int and identity["bytes"] > 0, "IDENTITY")
    return identity["url"]+"?generation="+identity["generation"]


def response(identity, start, end):
    """Validate headers before allowing any payload read, including large members."""
    require(type(start) is int and type(end) is int and 0 <= start <= end < identity["bytes"], "REQUEST_RANGE")
    req = urllib.request.Request(bound_url(identity), headers={"Range": f"bytes={start}-{end}",
        "If-Match": identity["etag"], "Accept-Encoding": "identity", "User-Agent": "MNLM-source-census/1"})
    result = urllib.request.urlopen(req, timeout=60)
    try:
        headers = {k.lower(): v for k, v in result.headers.items()}
        require(result.status == 206, "RANGE_IGNORED")
        require(headers.get("content-range") == f"bytes {start}-{end}/{identity['bytes']}"
                and int(headers.get("content-length", "-1")) == end-start+1, "RESPONSE_RANGE")
        require(headers.get("x-goog-generation") == identity["generation"]
                and headers.get("etag") == identity["etag"]
                and headers.get("content-encoding", "identity") == "identity", "PROVIDER_DRIFT")
        return result, headers
    except BaseException:
        result.close()
        raise


class DirectoryReader(io.RawIOBase):
    def __init__(self, root, identity, *, offline_pin=None):
        self.root, self.identity = Path(root), identity
        self.url, self.pos, self.ranges = bound_url(identity), 0, []
        self.offline = offline_pin is not None
        if self.offline:
            raw = (self.root/"ranges.json").read_bytes()
            require(sha(raw) == offline_pin, "MANIFEST_PIN")
            v = json.loads(raw)
            require(set(v) == {"schema", "identity", "ranges"} and v["schema"] == "range-zip-v1"
                    and v["identity"] == identity, "IDENTITY")
            self.ranges = v["ranges"]
            for e in self.ranges:
                require(set(e) == {"start", "end", "sha256", "headers"}
                        and type(e["start"]) is int and type(e["end"]) is int
                        and 0 <= e["start"] <= e["end"] < identity["bytes"]
                        and e["end"]-e["start"]+1 <= MAX_METADATA_READ, "RANGE_LAYOUT")
                self.cached(e)
        else:
            self.root.mkdir(parents=True, exist_ok=False)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        require(whence in (0, 1, 2), "SEEK")
        target = offset + (0 if whence == 0 else self.pos if whence == 1 else self.identity["bytes"])
        require(type(target) is int and 0 <= target <= self.identity["bytes"], "SEEK")
        self.pos = target
        return target

    def cached(self, e):
        raw = (self.root/f"{e['start']}-{e['end']}.bin").read_bytes()
        require(len(raw) == e["end"]-e["start"]+1 and sha(raw) == e["sha256"], "CACHE_BYTES")
        return raw

    def read(self, size=-1):
        require(type(size) is int and size >= -1, "READ")
        count = self.identity["bytes"]-self.pos if size == -1 else min(size, self.identity["bytes"]-self.pos)
        require(count <= MAX_METADATA_READ, "READ_BUDGET")
        if count == 0:
            return b""
        start, end = self.pos, self.pos+count-1
        for e in self.ranges:
            if e["start"] <= start and e["end"] >= end:
                raw = self.cached(e)[start-e["start"]:end-e["start"]+1]
                self.pos += count
                return raw
        require(not self.offline, "UNCACHED_RANGE")
        require(sum(e["end"]-e["start"]+1 for e in self.ranges)+count <= MAX_METADATA_TOTAL, "TOTAL_BUDGET")
        stream, headers = response(self.identity, start, end)
        with stream:
            raw = stream.read(count+1)
        require(len(raw) == count, "SHORT_RANGE")
        entry = dict(start=start, end=end, sha256=sha(raw), headers=headers)
        with (self.root/f"{start}-{end}.bin").open("xb") as f:
            f.write(raw)
        self.ranges.append(entry)
        self.pos += count
        return raw

    def seal(self):
        require(not self.offline, "ALREADY_SEALED")
        raw = encode(dict(schema="range-zip-v1", identity=self.identity, ranges=self.ranges))
        with (self.root/"ranges.json").open("xb") as f:
            f.write(raw)
        return sha(raw)


def entries(reader):
    with zipfile.ZipFile(reader) as archive:
        rows = []
        for e in archive.infolist():
            require(re.fullmatch(r"(?:videos/)?video_[0-9]+\.mp4", e.filename) is not None,
                    "MEMBER_NAME")
            require(not e.flag_bits & 1 and e.compress_type in (
                        zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2)
                    and e.file_size > 0 and e.compress_size > 0
                    and e.header_offset >= 0 and e.header_offset+e.compress_size < reader.identity["bytes"],
                    "MEMBER_LAYOUT")
            rows.append(dict(name=e.filename, crc32=e.CRC, bytes=e.file_size,
                             compressed_bytes=e.compress_size, header_offset=e.header_offset,
                             method=e.compress_type, flags=e.flag_bits))
        require(len({r["name"] for r in rows}) == len(rows), "DUPLICATE_MEMBER")
        return sorted(rows, key=lambda r: r["name"])
