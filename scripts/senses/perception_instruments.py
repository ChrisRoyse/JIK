"""Native source prefixes -> separate frozen instrument readings, never a fused vector.

The bridge retains every source frame as row-major, nonoverlapping bounded RGB
tiles and every authorized audio span/channel. Tiles do not create instruments.
Source clocks are presentation clocks, not calibrated physical synchronization.
These two candidates are NOT an admitted panel or a training-ready publication.
"""
import hashlib
import io
import json
from pathlib import Path
import struct
import time

import numpy as np

from senses.blobs import MAX_BYTES
from senses.contract import CHANNELS, absent, canonical, digest, field, present, seal_record
from senses.perception_native import check, fraction

SCHEMA = "native-instrument-readings-v1"
EDGE, DIM = 512, 384
FLAGS = dict(meaning_asserted=False, scientific_admission=False, training_ready=False,
             physical_sync_calibrated=False, transcript_available=False,
             candidate_readings_only=True, no_flatten=True)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def contract(lens, medium):
    return dict(instrument=lens.identity, bridge_schema=SCHEMA, modality=medium,
                bridge_sha256=sha(Path(__file__).read_bytes()),
                image_partition="all pixels; row-major 512-edge disjoint tiles; lossless RGB PNG",
                audio_partition="contiguous authorized source blocks; separate channels; IEEE-float WAV",
                audio_clock_mapping="descriptor.span[0] + resampled_index / 16000 seconds; native span.start is wrapper-relative",
                pooling=False, admission=False)


def descriptors(prefix, request):
    """Construct the complete expected roster before any payload or model work."""
    chosen = prefix.nodes(**request)
    check(set(chosen) == {"image", "audio"}, "BRIDGE_MEDIA")
    image, audio = [], []
    h, w, _ = prefix.value["formats"]["image"]["shape"]
    for r in chosen["image"]:
        for y in range(0, h, EDGE):
            for x in range(0, w, EDGE):
                height, width = min(EDGE, h-y), min(EDGE, w-x)
                image.append(dict(modality="image", node=r["node"], source_payload=r["sha256"],
                    rectangle=[x, y, width, height], frame_shape=[h, w, 3],
                    shape=[1+((height+13)//14)*((width+13)//14), DIM]))
    spec = prefix.value["formats"]["audio"]
    groups = []
    for r in chosen["audio"]:
        n = r["node"]
        if not groups or groups[-1][-1]["node"]["span"][1] != n["span"][0]:
            groups.append([])
        groups[-1].append(r)
    for rows in groups:
        samples = sum(r["node"]["sample_count"] for r in rows)
        check(samples*spec["channels"]*4 <= MAX_BYTES, "BRIDGE_AUDIO_BUDGET")
        n16 = (samples*16000 + spec["sample_rate"]-1)//spec["sample_rate"]
        # Existing encoder limit, across channels, is not raised here.
        from learned import MAX_WINDOWS
        check(((n16+479999)//480000)*spec["channels"] <= MAX_WINDOWS, "BRIDGE_AUDIO_BUDGET")
        for channel in range(spec["channels"]):
            audio.append(dict(modality="audio", nodes=[r["node"] for r in rows],
                source_payloads=[r["sha256"] for r in rows], channel=channel,
                span=[rows[0]["node"]["span"][0], rows[-1]["node"]["span"][1]],
                rate=spec["sample_rate"], channels=spec["channels"], samples=samples,
                shape=[(n16+319)//320, DIM]))
    return dict(image=image, audio=audio)


class Blob:
    """One bounded immutable wrapper for the unchanged UCR instrument APIs."""
    def __init__(self, raw, media):
        check(0 < len(raw) <= MAX_BYTES, "BRIDGE_BLOB_BUDGET")
        self.raw, self.media, self.pin = raw, media, sha(raw)

    def read(self, pin, media):
        check(pin == self.pin and media == self.media and sha(self.raw) == pin, "BRIDGE_BLOB")
        return self.raw


def image_input(rgb, descriptor, prefix):
    from PIL import Image
    x, y, w, h = descriptor["rectangle"]
    tile = rgb[y:y+h, x:x+w]
    check(tile.dtype == np.uint8 and tile.shape == (h, w, 3), "BRIDGE_PIXELS")
    stream = io.BytesIO()
    Image.fromarray(tile).save(stream, format="PNG")
    blob = Blob(stream.getvalue(), "image/png")
    # Wrapping is lossless before the original frozen instrument preprocesses.
    with Image.open(io.BytesIO(blob.raw)) as im:
        check(np.asarray(im).tobytes() == tile.tobytes(), "BRIDGE_PNG_LOSS")
    item = dict(modality="image", sha256=blob.pin, media_type=blob.media,
                measurements=[field("width", w, "integer"), field("height", h, "integer"),
                              field("channels", 3, "integer"), field("layout", "RGB", "category")])
    record = dict(schema="ucr-v1", unit=prefix.value["split_group"], referents=[],
        locator=dict(source="perception-native", shard=prefix.value["clip"],
                     row=descriptor["node"]["ordinal"], span=[y*rgb.shape[1]+x, y*rgb.shape[1]+x+w],
                     record_digest=digest(descriptor), manifest_digest=prefix.value["source"]["graph"]),
        channels={n: absent("not_observed") for n in CHANNELS})
    record["channels"]["perception"] = present([item])
    return seal_record(record), blob


def audio_input(pcm, descriptor):
    import soundfile as sf
    check(pcm.dtype == np.float32 and pcm.shape == (descriptor["samples"], descriptor["channels"])
          and bool(np.isfinite(pcm).all()), "BRIDGE_PCM")
    # Canonical IEEE-float RIFF header: no encoder-inserted date/PEAK chunk.
    channels, rate = descriptor["channels"], descriptor["rate"]
    raw = pcm.astype("<f4", copy=False).tobytes()
    header = (b"RIFF"+struct.pack("<I", 36+len(raw))+b"WAVEfmt "+struct.pack(
        "<IHHIIHH", 16, 3, channels, rate, rate*channels*4, channels*4, 32)
        +b"data"+struct.pack("<I", len(raw)))
    blob = Blob(header+raw, "audio/wav")
    reread, rate = sf.read(io.BytesIO(blob.raw), dtype="float32", always_2d=True)
    check(rate == descriptor["rate"] and reread.tobytes() == pcm.tobytes(), "BRIDGE_WAV_LOSS")
    span = dict(sha256=blob.pin, media_type=blob.media, rate=rate,
                channel=descriptor["channel"], start=0, end=len(pcm))
    return span, blob


def inputs(prefix, request, roster):
    """Stream image wrappers; hold only one bounded contiguous audio span."""
    by_frame = {}
    for d in roster["image"]:
        by_frame.setdefault(d["node"]["id"], []).append(d)
    for obs in prefix.consume(**{**request, "required_media": ["image"]}):
        for d in by_frame[obs["node"]["id"]]:
            value, blob = image_input(obs["values"], d, prefix)
            yield d, value, blob
    it = iter(prefix.consume(**{**request, "required_media": ["audio"]}))
    channels = prefix.value["formats"]["audio"]["channels"]
    for i in range(0, len(roster["audio"]), channels):
        ds = roster["audio"][i:i+channels]
        pcm = np.empty((ds[0]["samples"], channels), dtype=np.float32)
        offset = 0
        for node in ds[0]["nodes"]:
            obs = next(it)
            check(obs["node"] == node, "BRIDGE_AUDIO_ORDER")
            count = len(obs["values"])
            pcm[offset:offset+count] = obs["values"]
            offset += count
        check(offset == len(pcm), "BRIDGE_AUDIO_COUNT")
        for d in ds:
            span, blob = audio_input(pcm, d)
            yield d, span, blob
    check(next(it, None) is None, "BRIDGE_AUDIO_EXTRA")


def read_slot(root, entry):
    check(entry["key"] == digest(dict(instrument=entry["instrument_id"], descriptor=entry["descriptor"])),
          "READING_KEY")
    path = Path(root)/"slots"/(entry["key"]+".bin")
    raw = path.read_bytes()
    d = entry["descriptor"]
    check(len(raw) == d["shape"][0]*DIM*4 and sha(raw) == entry["sha256"], "READING_BYTES")
    values = np.frombuffer(raw, dtype="<f4").reshape(d["shape"])
    check(bool(np.isfinite(values).all()), "READING_NONFINITE")
    return values


def validate_native(entry):
    d, windows = entry["descriptor"], entry["native"]
    if d["modality"] == "image":
        x, y, w, h = d["rectangle"]
        ph, pw = (h+13)//14*14, (w+13)//14*14
        check(len(windows) == 1, "READING_NATIVE_WINDOWS")
        expected = dict(sha256=windows[0].get("sha256"), height=h, width=w,
                        padded_height=ph, padded_width=pw, grid=[ph//14, pw//14], patch=14,
                        order="CLS then row-major spatial patches",
                        padding="right/bottom zero RGB before normalization",
                        token_start=0, token_end=d["shape"][0])
        from senses.contract import is_sha256
        check(windows[0] == expected and is_sha256(expected["sha256"]), "READING_NATIVE_WINDOWS")
    else:
        count = (d["samples"]*16000+d["rate"]-1)//d["rate"]
        expected, offset = [], 0
        check(bool(windows), "READING_NATIVE_WINDOWS")
        for start in range(0, count, 480000):
            end = min(count, start+480000)
            n = (end-start+319)//320
            expected.append(dict(sha256=windows[0].get("sha256"), media_type="audio/wav",
                rate=d["rate"], channel=d["channel"], start=0, end=d["samples"],
                resampled_start=start, resampled_end=end, token_start=offset, token_end=offset+n,
                original_coordinate_mapping="span.start + resampled_index * rate / 16000"))
            offset += n
        from senses.contract import is_sha256
        check(windows == expected and is_sha256(windows[0]["sha256"]), "READING_NATIVE_WINDOWS")


def measure(prefix, request, lenses, root, name):
    """Publish a complete typed reading roster; reuse only fully bound cache keys."""
    started = time.perf_counter()
    root = Path(root)
    check(isinstance(name, str) and name and all(c.isascii() and (c.isalnum() or c in "_-") for c in name),
          "READING_NAME")
    slots = root/"slots"
    slots.mkdir(parents=True, exist_ok=True)
    target = root/(name+".json")
    check(not target.exists(), "READING_EXISTS")
    roster = descriptors(prefix, request)
    identities = {m: contract(lenses[m], m) for m in ("image", "audio")}
    ids = {m: digest(c) for m, c in identities.items()}
    result = {m: [] for m in identities}
    phases = dict(preparation=0., forward=0., cache=0., write=0.)
    reused, forwarded = 0, 0
    iterator = iter(inputs(prefix, request, roster))
    while True:
        t = time.perf_counter()
        item = next(iterator, None)
        phases["preparation"] += time.perf_counter()-t
        if item is None:
            break
        d, value, blob = item
        m = d["modality"]
        key = digest(dict(instrument=ids[m], descriptor=d))
        meta = slots/(key+".json")
        t = time.perf_counter()
        if meta.exists():
            entry = json.loads(meta.read_bytes())
            check(entry["descriptor"] == d and entry["instrument_id"] == ids[m], "READING_CACHE_ID")
            validate_native(entry)
            read_slot(root, entry)
            reused += 1
            phases["cache"] += time.perf_counter()-t
        else:
            if m == "image":
                observed = lenses[m].measure(value, blob)
                tokens, native = observed["tokens"], observed["regions"]
            else:
                tokens, native = lenses[m].embed([value], blob)
            check(list(tokens.shape) == d["shape"] and tokens.dtype == np.float32
                  and bool(np.isfinite(tokens).all()), "READING_SHAPE")
            phases["forward"] += time.perf_counter()-t
            t = time.perf_counter()
            raw = tokens.astype("<f4", copy=False).tobytes()
            entry = dict(key=key, instrument_id=ids[m], descriptor=d, sha256=sha(raw), native=native)
            validate_native(entry)
            with (slots/(key+".bin")).open("xb") as f:
                f.write(raw)
            with meta.open("xb") as f:
                f.write(canonical(entry))
            read_slot(root, entry)
            phases["write"] += time.perf_counter()-t
            forwarded += 1
        result[m].append(entry)
    check(all([e["descriptor"] for e in result[m]] == roster[m] for m in roster), "READING_ROSTER")
    body = dict(schema=SCHEMA, source=prefix.value["source"], clock=prefix.value["clock"],
                split_group=prefix.value["split_group"], request=request, flags=FLAGS,
                instruments=identities, instrument_ids=ids, readings=result)
    raw = canonical(body)
    with target.open("xb") as f:
        f.write(raw)
    wall = time.perf_counter()-started
    return dict(path=target.name, pin=sha(raw), reused=reused, forwarded=forwarded,
                phases=phases, seconds=wall, phase_coverage=sum(phases.values())/wall)


class Readings:
    """Read every native window separately; never represent the panel by one vector."""
    def __init__(self, root, name, pin, prefix, request, instrument_ids):
        self.root = Path(root)
        raw = (self.root/(name+".json")).read_bytes()
        check(sha(raw) == pin, "READING_MANIFEST")
        v = json.loads(raw)
        check(set(v) == {"schema", "source", "clock", "split_group", "request", "flags",
                         "instruments", "instrument_ids", "readings"}, "READING_SCHEMA")
        check(v["schema"] == SCHEMA and v["flags"] == FLAGS, "READING_FLAGS")
        check(v["source"] == prefix.value["source"] and v["clock"] == prefix.value["clock"]
              and v["split_group"] == prefix.value["split_group"] and v["request"] == request, "READING_SOURCE")
        check(set(v["instruments"]) == set(v["readings"]) == set(instrument_ids) == {"image", "audio"}
              and v["instrument_ids"] == instrument_ids
              and all(digest(v["instruments"][m]) == instrument_ids[m] for m in instrument_ids),
              "READING_INSTRUMENTS")
        expected = descriptors(prefix, request)
        for m in expected:
            check([e["descriptor"] for e in v["readings"][m]] == expected[m], "READING_ROSTER")
            for e in v["readings"][m]:
                check(set(e) == {"key", "instrument_id", "descriptor", "sha256", "native"}
                      and e["instrument_id"] == instrument_ids[m], "READING_SLOT_ID")
                validate_native(e)
                read_slot(self.root, e)
        self.value = v

    def consume(self, *, purpose="engineering_boundary"):
        check(purpose == "engineering_boundary", "READING_NATIVE_ADMISSION_REQUIRED")
        for m in ("image", "audio"):
            for e in self.value["readings"][m]:
                yield dict(modality=m, instrument_id=e["instrument_id"], descriptor=e["descriptor"],
                           native=e["native"], values=read_slot(self.root, e),
                           clock=self.value["clock"], split_group=self.value["split_group"],
                           through=self.value["request"]["through"], meaning_asserted=False)
