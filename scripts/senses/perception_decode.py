"""Persist exact native RGB/PCM prefixes before they reach sensing instruments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.perception_native import SCHEMA, FLAGS, check, source, layout
from text.evaluation_multiscale import file_sha


def runtime(executable="ffmpeg"):
    path = shutil.which(executable)
    check(path is not None, "FFMPEG_MISSING")
    version = subprocess.check_output([path, "-version"],
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0).decode()
    return dict(executable=path, sha256=file_sha(Path(path)), version=version)


def command(executable, media, medium, data):
    count = len(data["nodes"])
    base = [executable, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info",
            "-threads", "0", "-copyts", "-noautorotate", "-i", str(media),
            "-map", f"0:{data['stream']}"]
    if medium == "image":
        return base+["-vf", f"trim=end_frame={count},format=pix_fmts=rgb24,showinfo",
                     "-frames:v", str(count), "-fps_mode", "passthrough",
                     "-c:v", "rawvideo", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    samples = sum(n["sample_count"] for n in data["nodes"])
    return base+["-af", f"atrim=end_sample={samples},aformat=sample_fmts=flt,ashowinfo",
                 "-frames:a", str(count), "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"]


def read_exact(pipe, size):
    parts, remaining = [], size
    while remaining:
        raw = pipe.read(remaining)
        check(raw, "DECODE_SHORT")
        parts.append(raw)
        remaining -= len(raw)
    return b"".join(parts)


def check_log(raw, medium, data):
    lines = raw.decode("utf8", errors="replace").splitlines()
    token = "Parsed_showinfo_" if medium == "image" else "Parsed_ashowinfo_"
    rows = [line for line in lines if token in line and re.search(r"\bn:\s*\d+", line)]
    check(len(rows) == len(data["nodes"]), "DECODE_ROSTER")
    for row, node in zip(rows, data["nodes"]):
        number = lambda name: int(re.search(r"\b"+name+r":\s*(-?\d+)", row).group(1))
        check(number("n") == node["ordinal"] and number("pts") == node["pts"], "DECODE_PTS")
        if medium == "image":
            h, w, _ = data["spec"]["shape"]
            check(re.search(r"\bfmt:rgb24\b", row) and f"s:{w}x{h}" in row, "DECODE_FORMAT")
        else:
            check(number("nb_samples") == node["sample_count"]
                  and number("channels") == data["spec"]["channels"]
                  and number("rate") == data["spec"]["sample_rate"]
                  and re.search(r"\bfmt:flt\b", row), "DECODE_FORMAT")


def decode(cmd, destination, medium, data):
    records = []
    (destination/medium).mkdir()
    log_path = destination/f"{medium}.log"
    with log_path.open("xb") as log:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            for node, size in zip(data["nodes"], data["sizes"]):
                raw = read_exact(process.stdout, size)
                relative = f"{medium}/{node['ordinal']:06d}.bin"
                with (destination/relative).open("xb") as output:
                    output.write(raw)
                records.append(dict(node=node, file=relative, bytes=size,
                                    sha256=hashlib.sha256(raw).hexdigest()))
            check(not process.stdout.read(1), "DECODE_OVERFLOW")
            check(process.wait() == 0, "DECODE_EXIT")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
    check_log(log_path.read_bytes(), medium, data)
    return records


def build(episodes, episode_pin, clip, through, output, executable="ffmpeg",
          media=("image", "audio")):
    tick = time.perf_counter()
    graph, streams, bound = source(episodes, episode_pin, clip)
    data = layout(graph, streams, through, media)
    decoder = runtime(executable)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    phases = {"bind": time.perf_counter()-tick}
    records, commands = {}, {}
    for medium, d in data.items():
        start = time.perf_counter()
        cmd = command(decoder["executable"], Path(episodes)/"media"/f"{clip}.mp4", medium, d)
        commands[medium] = cmd
        records[medium] = decode(cmd, output, medium, d)
        phases[medium] = time.perf_counter()-start
    value = dict(schema=SCHEMA, clip=clip, source=bound, clock=graph["clock"],
        split_group=graph["split_group"], provider_split=graph["provider_split"], through=through,
        runtime=decoder, records=records, formats={m: d["spec"] for m, d in data.items()},
        flags=FLAGS, commands=commands, logs={m: file_sha(output/f"{m}.log") for m in data})
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    with (output/"manifest.json").open("xb") as f:
        f.write(raw)
    pin = hashlib.sha256(raw).hexdigest()
    with (output/"COMMITTED").open("x") as f:
        f.write(pin)
    wall = time.perf_counter()-tick
    return dict(pin=pin, phases=phases, wall_seconds=wall,
                bytes=sum(r["bytes"] for rows in records.values() for r in rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--episode-pin", required=True)
    p.add_argument("--clip", required=True)
    p.add_argument("--through", type=int, nargs=2, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--media", choices=("image", "audio"), nargs="+", default=["image", "audio"])
    a = p.parse_args()
    print(json.dumps(build(a.episodes, a.episode_pin, a.clip, a.through, a.output, a.ffmpeg, a.media)))
