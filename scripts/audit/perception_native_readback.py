"""Independent source-clock and decoded-byte readback; no production selector."""
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np


def need(ok, code):
    if not ok:
        raise ValueError("MNLM_NATIVE_READ_"+code)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for raw in iter(lambda: f.read(1024**2), b""):
            h.update(raw)
    return h.hexdigest()


def read(root, episodes, pin, episode_pin):
    root, episodes = Path(root), Path(episodes)
    need(sha(root/"manifest.json") == pin == (root/"COMMITTED").read_text(), "PIN")
    need(sha(episodes/"report.json") == episode_pin, "SOURCE_REPORT")
    manifest = json.loads((root/"manifest.json").read_bytes())
    clip = manifest["clip"]
    report = json.loads((episodes/"report.json").read_bytes())
    files = [f"media/{clip}.mp4", f"public/{clip}.json", f"probes/{clip}.json"]
    for name in files:
        need(sha(episodes/name) == report["files"][name]["sha256"]
             and (episodes/name).stat().st_size == report["files"][name]["bytes"], "SOURCE_BYTES")
    graph = json.loads((episodes/files[1]).read_bytes())
    probe = json.loads((episodes/files[2]).read_bytes())
    streams = {s["index"]: s for s in probe["streams"]}
    clock_hash = hashlib.sha256(json.dumps(["container-presentation-clock", sha(episodes/files[0])],
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    need(manifest["clock"] == graph["clock"] and manifest["clock"]["id"] == clock_hash
         and manifest["split_group"] == graph["split_group"]
         and manifest["provider_split"] == graph["provider_split"], "CLOCK")
    end = Fraction(*manifest["through"])
    expected, ordinal = {"image": [], "audio": []}, {"image": 0, "audio": 0}
    for frame in probe["frames"]:
        s = streams[frame["stream_index"]]
        m = "image" if frame["media_type"] == "video" else "audio"
        start = frame["pts"]*Fraction(s["time_base"])
        duration = Fraction(frame["nb_samples"], int(s["sample_rate"])) if m == "audio" else (
            frame.get("duration", frame.get("pkt_duration", 0))*Fraction(s["time_base"]))
        if start+duration <= end:
            expected[m].append(dict(ordinal=ordinal[m], pts=frame["pts"],
                start=start, end=start+duration, samples=frame.get("nb_samples"), stream=s))
        ordinal[m] += 1
    receipt = {}
    for medium, rows in expected.items():
        saved = manifest["records"][medium]
        need(len(saved) == len(rows), "COUNT")
        log = (root/f"{medium}.log").read_text(encoding="utf8", errors="replace")
        label = "Parsed_showinfo_" if medium == "image" else "Parsed_ashowinfo_"
        logged = [line for line in log.splitlines() if label in line and re.search(r"\bn:\s*\d+", line)]
        need(len(logged) == len(rows), "LOG_COUNT")
        for row, record, line in zip(rows, saved, logged):
            n = record["node"]
            need(n["pts"] == row["pts"] and n["ordinal"] == row["ordinal"]
                 and Fraction(*n["span"][0]) == row["start"]
                 and Fraction(*n["span"][1]) == row["end"]
                 and Fraction(*n["available_after"]) == row["end"], "PTS")
            fields = dict(re.findall(r"\b(n|pts|nb_samples|channels|rate):\s*(-?\d+)", line))
            need(int(fields["n"]) == row["ordinal"] and int(fields["pts"]) == row["pts"], "DECODE_PTS")
            raw = (root/record["file"]).read_bytes()
            if medium == "image":
                s = row["stream"]
                size = s["width"]*s["height"]*3
                need(f"s:{s['width']}x{s['height']}" in line and "fmt:rgb24" in line, "RGB")
            else:
                size = row["samples"]*row["stream"]["channels"]*4
                need(int(fields["nb_samples"]) == row["samples"]
                     and int(fields["rate"]) == int(row["stream"]["sample_rate"])
                     and int(fields["channels"]) == row["stream"]["channels"]
                     and np.isfinite(np.frombuffer(raw, dtype="<f4")).all(), "PCM")
            need(len(raw) == record["bytes"] == size
                 and hashlib.sha256(raw).hexdigest() == record["sha256"], "PAYLOAD")
        receipt[medium] = dict(records=len(rows), bytes=sum(r["bytes"] for r in saved),
            payloads=[r["sha256"] for r in saved])
    return manifest, receipt


def alternate(root, episodes, manifest, through, output, *, fault=None):
    """Fresh FFmpeg selection by decoded ordinal, not production trim filters."""
    root, episodes, output = Path(root), Path(episodes), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cutoff = Fraction(*through)
    result = {}
    for medium in ("image", "audio"):
        selected = [r for r in manifest["records"][medium]
                    if Fraction(*r["node"]["available_after"]) <= cutoff]
        n = len(selected)
        need(n > 0, "ALTERNATE_EMPTY")
        command = [manifest["runtime"]["executable"], "-hide_banner", "-nostdin", "-nostats",
            "-loglevel", "error", "-threads", "0", "-copyts", "-noautorotate", "-i",
            str(episodes/"media"/f"{manifest['clip']}.mp4"), "-map", f"0:{selected[0]['node']['stream']}"]
        if medium == "image":
            command += ["-vf", f"select=lt(n\\,{n}),format=pix_fmts=rgb24",
                        "-frames:v", str(n), "-c:v", "rawvideo", "-pix_fmt", "rgb24"]
            command += ["-r", "25", "-fps_mode", "cfr"] if fault == "fps" else ["-fps_mode", "passthrough"]
            command += ["-f", "rawvideo", "pipe:1"]
        else:
            command += ["-af", f"aselect=lt(n\\,{n}),aformat=sample_fmts=flt",
                        "-frames:a", str(n), "-c:a", "pcm_f32le"]
            if fault == "audio_rate":
                command += ["-ar", "16000"]
            command += ["-f", "f32le", "pipe:1"]
        h, count, chunks, sizes = hashlib.sha256(), 0, [], []
        with (output/f"{medium}.log").open("xb") as log:
            p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            try:
                for record in selected:
                    pieces, left = [], record["bytes"]
                    while left:
                        raw = p.stdout.read(left)
                        if not raw:
                            break
                        pieces.append(raw)
                        left -= len(raw)
                    block = b"".join(pieces)
                    h.update(block); count += len(block); sizes.append(len(block))
                    chunks.append(hashlib.sha256(block).hexdigest())
                    if left:
                        break
                extra = p.stdout.read(1)
                if extra:
                    p.kill()
                code = p.wait()
            finally:
                if p.poll() is None:
                    p.kill(); p.wait()
                p.stdout.close()
        same = (code == 0 and not extra and sizes == [r["bytes"] for r in selected]
                and chunks == [r["sha256"] for r in selected])
        result[medium] = dict(matches=same, records=len(chunks), bytes=count,
                              sha256=h.hexdigest(), command=command)
        if fault is None:
            need(same, "ALTERNATE_BYTES")
    if fault == "fps":
        need(not result["image"]["matches"], "FPS_CONTROL")
    if fault == "audio_rate":
        need(not result["audio"]["matches"], "RATE_CONTROL")
    (output/"receipt.json").write_text(json.dumps(result, sort_keys=True), encoding="utf8")
    return result
