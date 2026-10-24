"""Frozen CUDA Whisper encoder as a temporal instrument, without its decoder.

Explicit CPU decode/resampling/frontend, no CPU model fallback, channel mixing,
text input, temporal pooling, or silent truncation. Original coordinate ranges
and valid-token counts accompany every window; encoder attention includes its
declared zero-padding context. A token is not a verified word alignment.
"""
import hashlib
import io
import math
import platform
from pathlib import Path

import numpy as np

from learned import MAX_WINDOWS, check_gpu, snapshot_dir, weights_sha256
from senses.blobs import MAX_BYTES
from senses.contract import require

SPEC = {"name": "ucr-whisper-tiny-en-audio", "repo": "openai/whisper-tiny.en",
        "revision": "87c7102498dcde7456f24cfd30239ca606ed9063", "kind": "multi", "inputs": "ucr",
        "channel": "perception", "perspective": "perception", "role": "universal_instrument",
        "module": "audio_ucr", "group": "universal", "sim": "maxsim", "sparse": False}


def spans(record):
    """Union contiguous authorized frames, preserving gaps and separate channels."""
    groups = {}
    for item in record["channels"]["perception"]["items"]:
        if item["modality"] != "audio":
            continue
        fields = item["measurements"]
        values = {f["name"]: f for f in fields}
        require(len(values) == len(fields), "MNLM_AUDIO_SPAN", "duplicate audio measurement names")
        keys = ("sample_start", "sample_count", "sample_rate", "audio_channel")
        require(all(k in values and values[k]["type"] == "integer" and type(values[k]["value"]) is int for k in keys),
                "MNLM_AUDIO_SPAN", "audio requires explicit sample coordinates, rate and channel")
        start, count, rate, channel = (values[k]["value"] for k in keys)
        require(start >= 0 and count > 0 and rate > 0 and channel >= 0
                and values["sample_start"]["unit"] == values["sample_count"]["unit"] == "sample"
                and values["sample_rate"]["unit"] == "Hz", "MNLM_AUDIO_SPAN", "invalid audio coordinates/units")
        key = (item["sha256"], item["media_type"], rate, channel)
        groups.setdefault(key, []).append((start, start + count))
    out = []
    for (sha, media, rate, channel), ranges in sorted(groups.items()):
        merged = []
        for start, end in sorted(ranges):
            require(not merged or start >= merged[-1][1], "MNLM_AUDIO_SPAN", "overlapping audio frames")
            if merged and start == merged[-1][1]:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        for start, end in merged:
            out.append({"sha256": sha, "media_type": media, "rate": rate, "channel": channel, "start": start, "end": end})
    return out


def waveform(blob, span):
    import soundfile as sf
    from scipy.signal import resample_poly
    with sf.SoundFile(io.BytesIO(blob)) as audio:
        count = span["end"] - span["start"]
        require(span["media_type"] == "audio/" + audio.format.lower(), "MNLM_UCR_BLOB_TYPE", "declared media type differs from decoded format")
        require(audio.samplerate == span["rate"] and span["channel"] < audio.channels and span["end"] <= len(audio),
                "MNLM_AUDIO_SPAN", "source geometry differs from authorized percept")
        require(count * audio.channels * 4 <= MAX_BYTES, "MNLM_AUDIO_DECODE_BUDGET", "split the retained atom's sample span")
        expected = (count * 16000 + audio.samplerate - 1) // audio.samplerate
        require(expected <= MAX_WINDOWS * 480000, "MNLM_LENS_WINDOW_BUDGET", "audio atom exceeds the fixed window budget")
        audio.seek(span["start"])
        x = audio.read(count, dtype="float32", always_2d=True)
        require(len(x) == count and bool(np.isfinite(x).all()), "MNLM_AUDIO_DECODE", "incomplete/nonfinite audio")
        x = np.ascontiguousarray(x[:, span["channel"]])
        divisor = math.gcd(16000, audio.samplerate)
        # Resample ONLY the authorized span, never neighboring/held-out samples.
        x = resample_poly(x, 16000 // divisor, audio.samplerate // divisor, window=("kaiser", 5.0), padtype="constant")
        require(len(x) == expected and bool(np.isfinite(x).all()), "MNLM_AUDIO_RESAMPLE", "unexpected resampled geometry")
        return x.astype(np.float32, copy=False)


class AudioEncoder:
    dim = 384

    def __init__(self, spec=SPEC):
        import torch
        import transformers
        import scipy
        import soundfile as sf
        from safetensors.torch import load_file
        from transformers import WhisperConfig, WhisperFeatureExtractor
        from transformers.models.whisper.modeling_whisper import WhisperEncoder
        runtime = check_gpu()
        snap = snapshot_dir(spec["repo"])
        require(snap.name == spec["revision"], "MNLM_LENS_FROZEN_VIOLATION", "cached audio checkpoint revision differs")
        sha, files = weights_sha256(snap)
        config = WhisperConfig.from_pretrained(str(snap), local_files_only=True)
        config._attn_implementation = "eager"
        self.extractor = WhisperFeatureExtractor.from_pretrained(str(snap), local_files_only=True)
        require(config.d_model == self.dim and config.max_source_positions == 1500 and config.num_mel_bins == 80
                and self.extractor.sampling_rate == 16000 and self.extractor.n_samples == 480000
                and self.extractor.hop_length == 160, "MNLM_AUDIO_MODEL_SHAPE", "unsupported encoder geometry")
        state = load_file(str(snap / "model.safetensors"))
        state = {k.removeprefix("model.encoder."): v for k, v in state.items() if k.startswith("model.encoder.")}
        require(bool(state), "MNLM_LENS_WEIGHTS_MISSING", "no pinned encoder tensors")
        self.model = WhisperEncoder(config)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False).to(device="cuda", dtype=torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        runtime = check_gpu()  # identity describes the effective deterministic runtime
        self.identity = {**spec, "shape": f"Multi(var,{self.dim})", "dtype": "f32", "modality": "audio",
            "weights_sha256": sha, "weight_files": files, "corpus_hash": spec["revision"],
            "pretraining_data": "upstream checkpoint; training corpus bytes not supplied", "norm": "per-token L2",
            "preprocessing": {"rate": 16000, "window_samples": 480000, "token_stride_samples": 320,
                              "resample": "scipy.resample_poly;kaiser=5;constant-zero;authorized-span-only",
                              "frontend": "WhisperFeatureExtractor;CPU;no-normalization;right-zero-pad",
                              "padding": "encoder attends padded window; padded output positions removed",
                              "channels": "separate; never mixed", "decoder": False, "text_input": False},
            "assets": {n: hashlib.sha256((snap / n).read_bytes()).hexdigest() for n in ("config.json", "preprocessor_config.json")},
            "runtime": {**runtime, "python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "scipy": scipy.__version__, "numpy": np.__version__,
                        "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__, "platform": platform.platform()},
            "code_sha256": {n: hashlib.sha256((Path(__file__).parent / n).read_bytes()).hexdigest()
                            for n in ("audio_encoder.py", "audio_ucr.py", "universal.py", "learned.py")}}
        for n in ("blobs.py", "views.py", "contract.py"):
            self.identity["code_sha256"][n] = hashlib.sha256((Path(__file__).parents[1] / "senses" / n).read_bytes()).hexdigest()
        self.identity["encoder_state_sha256"] = self.state_digest()

    def state_digest(self):
        h = hashlib.sha256()
        for name, value in self.model.state_dict().items():
            h.update(name.encode())
            h.update(value.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def embed(self, regions, blobs):
        import torch
        parts, windows, offset = [], [], 0
        raw, previous = None, None
        total_windows = 0
        for span in regions:
            total_windows += math.ceil((span["end"] - span["start"]) * 16000 / span["rate"] / 480000)
        require(total_windows <= MAX_WINDOWS, "MNLM_LENS_WINDOW_BUDGET", "whole atom exceeds fixed audio window budget")
        for span in regions:
            if (span["sha256"], span["media_type"]) != previous:
                raw = blobs.read(span["sha256"], span["media_type"])
                previous = span["sha256"], span["media_type"]
            wave = waveform(raw, span)
            for start in range(0, len(wave), 480000):
                piece = wave[start:start + 480000]
                features = self.extractor(piece, sampling_rate=16000, padding="max_length", truncation=False,
                                          return_tensors="pt", return_attention_mask=True, device="cpu", do_normalize=False)
                valid = (len(piece) + 319) // 320
                with torch.inference_mode():
                    hidden = self.model(features.input_features.to("cuda")).last_hidden_state[0, :valid]
                    require(bool(torch.isfinite(hidden).all()) and bool((hidden.norm(dim=-1) > 0).all()),
                            "MNLM_LENS_NONFINITE", "invalid audio token measurement")
                    t = torch.nn.functional.normalize(hidden, p=2, dim=-1).cpu().numpy()
                parts.append(t)
                windows.append({**span, "resampled_start": start, "resampled_end": start + len(piece),
                                "token_start": offset, "token_end": offset + valid,
                                "original_coordinate_mapping": "span.start + resampled_index * rate / 16000"})
                offset += valid
        return (np.concatenate(parts) if parts else np.empty((0, self.dim), np.float32)), windows
