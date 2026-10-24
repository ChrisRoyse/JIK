"""Acquire exact native media members from a pinned archive, with bounded resume.

Compressed member bytes and original MP4s stay separate and create-only when
accepted. A partial is reusable only under its original generation/range record.
No provider validation/test URLs or evaluation labels are accepted here.
"""
import argparse
import bz2
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import time
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from range_zip import DirectoryReader, encode, require, response, sha
from acquire import NET_ERRORS, RETRIES

CHUNK = 1024 * 1024


def file_sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def header(reader, member):
    reader.seek(member["header_offset"])
    raw = reader.read(256)
    require(raw[:4] == b"PK\x03\x04", "LOCAL_SIGNATURE")
    _, version, flags, method, _, _, crc, compressed, size, nname, nextra = struct.unpack("<I5H3I2H", raw[:30])
    length = 30+nname+nextra
    if length > len(raw):
        raw += reader.read(length-len(raw))
    require(raw[30:30+nname].decode("utf8") == member["name"]
            and flags == member["flags"] == 0 and method == member["method"] == 12
            and crc == member["crc32"] and compressed == member["compressed_bytes"]
            and size == member["bytes"], "LOCAL_DIRECTORY_DISAGREEMENT")
    return member["header_offset"]+length


def compressed(identity, member, start, root):
    path = root/"compressed.bz2"
    partial = root/"compressed.part"
    key = dict(identity=identity, member=member, data_start=start)
    binding = root/"range.json"
    if binding.exists():
        require(binding.read_bytes() == encode(key), "RESUME_IDENTITY")
    else:
        require(not partial.exists() and not path.exists(), "UNBOUND_PARTIAL")
        binding.write_bytes(encode(key))
    if path.exists():
        receipt = json.loads((root/"compressed.json").read_bytes())
        require(path.stat().st_size == member["compressed_bytes"] and file_sha(path) == receipt["sha256"],
                "COMPRESSED_CACHE")
        return path, receipt
    n = partial.stat().st_size if partial.exists() else 0
    require(n <= member["compressed_bytes"], "RESUME_SIZE")
    attempts, requests = 0, []
    while n < member["compressed_bytes"]:
        before = n
        try:
            stream, headers = response(identity, start+n, start+member["compressed_bytes"]-1)
            with stream, partial.open("ab") as target:
                requests.append(dict(start=start+n, end=start+member["compressed_bytes"]-1, headers=headers))
                while n < member["compressed_bytes"]:
                    raw = stream.read(min(CHUNK, member["compressed_bytes"]-n))
                    if not raw:
                        break
                    target.write(raw)
                    n += len(raw)
                require(not stream.read(1), "EXTRA_COMPRESSED_BYTES")
                target.flush()
                os.fsync(target.fileno())
            if n == before:
                raise OSError("no progress in bounded member range")
            attempts = 0
        except NET_ERRORS:
            n = partial.stat().st_size if partial.exists() else 0
            attempts = 0 if n > before else attempts+1
            if attempts >= RETRIES:
                raise
            time.sleep(min(8, 2**attempts))
    require(n == member["compressed_bytes"], "COMPRESSED_LENGTH")
    receipt = dict(bytes=n, sha256=file_sha(partial), requests=requests)
    partial.rename(path)
    (root/"compressed.json").write_bytes(encode(receipt))
    return path, receipt


def decompress(path, member, dest):
    require(not dest.exists(), "MEDIA_EXISTS")
    decoder = bz2.BZ2Decompressor()
    total, crc, h = 0, 0, hashlib.sha256()
    with path.open("rb") as src, dest.open("xb") as out:
        while raw := src.read(CHUNK):
            require(not decoder.eof, "BZIP_TRAILING_BYTES")
            piece = raw
            while True:
                decoded = decoder.decompress(piece, max_length=CHUNK)
                piece = b""
                total += len(decoded)
                require(total <= member["bytes"], "DECODE_BUDGET")
                crc = zlib.crc32(decoded, crc)
                h.update(decoded)
                out.write(decoded)
                if decoder.eof or decoder.needs_input:
                    break
        out.flush()
        os.fsync(out.fileno())
    require(decoder.eof and not decoder.unused_data and total == member["bytes"]
            and crc == member["crc32"], "DECODE_INTEGRITY")
    return dict(bytes=total, crc32=crc, sha256=h.hexdigest())


def acquire(identity, member, root):
    require(re.fullmatch(r"videos/video_[0-9]+\.mp4", member["name"]) is not None, "MEMBER_NAME")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    complete = root/"receipt.json"
    if complete.exists():
        r = json.loads(complete.read_bytes())
        require(r["identity"] == identity and r["member"] == member
                and (root/"media.mp4").stat().st_size == r["media"]["bytes"]
                and file_sha(root/"media.mp4") == r["media"]["sha256"], "ACCEPTED_MEDIA_DRIFT")
        return {**r, "reused": True}
    started = time.perf_counter()
    t = time.perf_counter()
    hr = root/"header"
    if (hr/"ranges.json").exists():
        pin = (root/"header.pin").read_text()
        reader = DirectoryReader(hr, identity, offline_pin=pin)
    else:
        reader = DirectoryReader(hr, identity)
    data_start = header(reader, member)
    if not reader.offline:
        pin = reader.seal()
        (root/"header.pin").write_text(pin)
    phases = dict(header=time.perf_counter()-t)
    t = time.perf_counter()
    packed, compressed_receipt = compressed(identity, member, data_start, root)
    phases["network_and_compressed_hash"] = time.perf_counter()-t
    t = time.perf_counter()
    # A failed decode is quarantined explicitly; a later retry never overwrites it.
    partial = root/"decoded.partial"
    require(not partial.exists(), "DECODE_PARTIAL_REVIEW")
    media = decompress(packed, member, partial)
    phases["decode"] = time.perf_counter()-t
    t = time.perf_counter()
    require(file_sha(partial) == media["sha256"], "COLD_MEDIA_HASH")
    partial.rename(root/"media.mp4")
    phases["cold_readback"] = time.perf_counter()-t
    wall = time.perf_counter()-started
    result = dict(identity=identity, member=member, data_start=data_start, header_pin=pin,
        compressed=compressed_receipt, media=media, phases=phases, seconds=wall,
        phase_coverage=sum(phases.values())/wall, reused=False,
        source_scope="provider_train", scientific_admission=False, training_ready=False)
    complete.write_bytes(encode(result))
    require(json.loads(complete.read_bytes()) == result, "RECEIPT_READBACK")
    return result


def main(args):
    raw = args.plan.read_bytes()
    require(sha(raw) == args.plan_pin, "PLAN_PIN")
    p = json.loads(raw)
    require(p["schema"] == "native-foundation-source-plan-v1" and len(p["splits"]) == 512
            and p["identity"]["url"] == "https://storage.googleapis.com/dm-perception-test/zip_data/train_videos.zip"
            and p["identity"]["generation"] == "1685446268139925", "PLAN_SCOPE")
    require(len({x["clip"] for x in p["splits"]}) == 512
            and [x["role"] for x in p["splits"]] == ["increment_one"]*256+["increment_two"]*128+["held_out"]*128,
            "PLAN_SPLITS")
    args.output.mkdir(parents=True, exist_ok=True)
    journal = args.output/("journal-"+str(os.getpid())+".jsonl")
    start = time.perf_counter()
    results, errors = [], []
    for index, row in enumerate(p["splits"]):
        try:
            require(row["archive"]["name"] == "videos/"+row["clip"]+".mp4", "PLAN_MEMBER")
            r = acquire(p["identity"], row["archive"], args.output/"members"/row["clip"])
            result = dict(clip=row["clip"], role=row["role"], index=index, media=r["media"],
                          receipt_sha256=file_sha(args.output/"members"/row["clip"]/"receipt.json"),
                          seconds=r["seconds"], reused=r["reused"])
            results.append(result)
        except Exception as exc:
            result = dict(clip=row["clip"], index=index, error=type(exc).__name__+": "+str(exc))
            errors.append(result)
        with journal.open("ab") as f:
            f.write(encode(result)+b"\n")
            f.flush()
        print(json.dumps(dict(done=index+1, ok=len(results), errors=len(errors), clip=row["clip"])), flush=True)
    report = dict(schema="native-member-acquisition-v1", plan_pin=args.plan_pin,
        complete=len(results) == 512 and not errors, results=results, errors=errors,
        seconds=time.perf_counter()-start, source_scope="provider_train",
        scientific_admission=False, training_ready=False, native_timing_attested=False)
    dest = args.output/("report-"+str(os.getpid())+".json")
    dest.write_bytes(encode(report))
    print(json.dumps(dict(report=str(dest), pin=sha(encode(report)), complete=report["complete"])), flush=True)
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--plan-pin", required=True)
    ap.add_argument("--output", type=Path, required=True)
    raise SystemExit(main(ap.parse_args()))
