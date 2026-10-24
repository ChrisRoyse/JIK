"""Independent real-source/frontend/readback checks for the native audio bank."""
import io
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np

from senses.audio_bank import AudioBank, FLAGS, NAMES, input_views
from senses.contract import canonical
from senses.perception_instruments import sha
from senses.perception_native import NativePrefix, fraction


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def original_wave(blob, span, rate):
    import soundfile as sf
    from scipy.signal import resample_poly
    raw, observed_rate = sf.read(io.BytesIO(blob.raw), dtype="float32", always_2d=True)
    need(observed_rate == span["rate"], "independent source rate")
    raw = raw[span["start"]:span["end"], span["channel"]]
    divisor = math.gcd(rate, observed_rate)
    return resample_poly(raw, rate//divisor, observed_rate//divisor,
                         window=("kaiser", 5.0), padtype="constant").astype(np.float32)


def frontend_probe(lens, name, row, blob, capture):
    import torch
    from transformers import ASTFeatureExtractor, ClapFeatureExtractor
    from transformers.utils import is_speech_available
    span = row["span"]
    rate = 48000 if name == "clap" else 16000
    wave = original_wave(blob, span, rate)
    width = rate*(30 if name == "whisper" else 10)
    expected = []
    processor = lens.identity.get("preprocessing", {}).get("processor")
    if name == "ast":
        need(not is_speech_available(), "independent AST audit requires the installed upstream NumPy path")
        extractor = ASTFeatureExtractor.from_dict(processor)
    elif name == "clap":
        extractor = ClapFeatureExtractor.from_dict(processor)
    for start in range(0, len(wave), width):
        piece = wave[start:start+width]
        if name == "whisper":
            values = lens.extractor(piece, sampling_rate=16000, padding="max_length", truncation=False,
                return_tensors="pt", return_attention_mask=True, device="cpu", do_normalize=False).input_features
        elif name == "ast":
            values = extractor(piece, sampling_rate=16000, return_tensors="pt").input_values
        elif name == "clap":
            padded = np.pad(piece, (0, width-len(piece)))
            values = torch.tensor(extractor._np_extract_fbank_features(padded,
                extractor.mel_filters_slaney)[None, None].astype(np.float32))
        else:
            normalized = piece
            if processor["do_normalize"]:
                normalized = (piece-piece.mean())/np.sqrt(piece.var()+1e-7)
            values = torch.from_numpy(normalized[None])
        expected.append((start, values.numpy()))
    index = 0
    def hook(module, positional, kwargs):
        nonlocal index
        need(index < len(expected), "unexpected extra model forward")
        key = "input_features" if name in ("whisper", "clap") else "input_values"
        actual = (positional[0] if positional else kwargs[key]).detach().cpu().numpy()
        start, target = expected[index]
        need(actual.shape == target.shape and actual.dtype == target.dtype
             and actual.tobytes() == target.tobytes(), name+" actual frontend tensor")
        if "attention_mask" in kwargs:
            need(bool((kwargs["attention_mask"] == 1).all()), "whole authorized waveform attention")
        if name == "clap":
            need(not bool(kwargs["is_longer"].any()), "CLAP oversized input refused")
        capture.append(dict(instrument=name, clip=row["clip"], channel=span["channel"],
            resampled_start=start, shape=list(actual.shape), sha256=sha(actual.tobytes()), exact=True))
        index += 1
    return lens.model.register_forward_pre_hook(hook, with_kwargs=True), len(expected)


def sources(args, views):
    import soundfile as sf
    proofs, controls = [], []
    for clip in ("video_10993", "video_874"):
        selected = [x for x in views if x[0]["clip"] == clip]
        row, _, blob = selected[0]
        prefix_root = args.prefixes/("windows-prefix-"+clip)
        prefix = NativePrefix(prefix_root, expected=row["prefix_pin"], episodes=args.episodes,
                              episode_pin=args.episode_pin, media=("audio",))
        need(row["source"] == prefix.value["source"] and row["clock"] == prefix.value["clock"]
             and row["split_group"] == prefix.value["split_group"]
             and row["through"] == prefix.value["through"], "native source binding")
        request = dict(clock_id=row["clock"]["id"], split_group=row["split_group"],
                       through=row["through"], required_media=["audio"])
        samples = list(prefix.consume(**request))
        pcm = b"".join(x["values"].tobytes() for x in samples)
        wrapped, rate = sf.read(io.BytesIO(blob.raw), dtype="float32", always_2d=True)
        need(pcm == wrapped.tobytes() and rate == prefix.value["formats"]["audio"]["sample_rate"],
             "original native blocks -> lossless wrapper")
        need([x[1]["channel"] for x in selected] == list(range(wrapped.shape[1]))
             and all(x[1]["start"] == 0 and x[1]["end"] == len(wrapped) for x in selected),
             "complete separate source channels")
        need(row["source_span"] == [samples[0]["node"]["span"][0], samples[-1]["node"]["span"][1]]
             and all(fraction(x["node"]["available_after"]) <= fraction(row["through"]) for x in samples),
             "native rational cutoff")
        # Decode the original entire stream with no prefix trim/filter/format helper.
        executable = prefix.value["runtime"]["executable"]
        destination = args.output/(clip+"-independent-full.pcm")
        cmd = [executable, "-v", "error", "-nostdin", "-i", str(args.episodes/"media"/(clip+".mp4")),
               "-map", "0:1", "-vn", "-c:a", "pcm_f32le", "-f", "f32le", str(destination)]
        with (args.output/(clip+"-independent-full.log")).open("xb") as log:
            process = subprocess.run(cmd, stdout=log, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        need(process.returncode == 0 and len(pcm) <= destination.stat().st_size <= 64*1024*1024,
             "bounded independent full-source decode")
        with destination.open("rb") as f:
            need(f.read(len(pcm)) == pcm, "independent complete-stream prefix bytes")
        proofs.append(dict(clip=clip, blocks=len(samples), samples=len(wrapped), channels=wrapped.shape[1],
                           pcm_bytes=len(pcm), pcm_sha256=sha(pcm), wrapper_sha256=blob.pin))
        for label, changed, code in (
            ("missing-image", dict(required_media=["image"]), "MISSING_MODALITY"),
            ("future-cutoff", dict(through=[32, 1]), "UNDECODED_FUTURE"),
            ("wrong-clock", dict(clock_id="foreign"), "CLOCK"),
            ("wrong-split", dict(split_group="foreign"), "SPLIT")):
            try:
                list(prefix.consume(**{**request, **changed}))
            except ValueError as exc:
                need(str(exc) == "MNLM_NATIVE_PREFIX_"+code, "source control reason")
                controls.append(dict(clip=clip, name=label, code=str(exc)))
            else:
                raise AssertionError("invalid source request accepted")
    return proofs, controls


def cold(args):
    raw = (args.output/"report.json").read_bytes()
    need(sha(raw) == args.report_pin, "cold report pin")
    report = json.loads(raw)
    bank = AudioBank(args.output, report["bank_pin"], args.inputs, args.input_pin, report["instruments"])
    counts = {}
    for name, _, values in bank.consume():
        count = counts.setdefault(name, dict(streams=0, rows=0, bytes=0, dim=values.shape[1]))
        count["streams"] += 1
        count["rows"] += len(values)
        count["bytes"] += values.nbytes
    need(set(counts) == set(NAMES) and all(x["streams"] == 4 for x in counts.values()),
         "complete cold typed roster")
    proofs, controls = sources(args, input_views(args.inputs, args.input_pin))
    need(len(report["frontends"]) == 72 and len(report["replays"]) == 20, "actual frontend/replay coverage")
    value = dict(author_checked=True, report_pin=args.report_pin, bank_pin=report["bank_pin"],
                 counts=counts, sources=proofs, controls=controls, flags=FLAGS)
    with (args.output/"cold.json").open("xb") as f:
        f.write(canonical(value))
    print(json.dumps(dict(cold_pin=sha(canonical(value)), counts=counts)), flush=True)
