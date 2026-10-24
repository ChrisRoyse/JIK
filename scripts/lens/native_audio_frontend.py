"""Explicit CPU audio frontends; each CUDA encoder receives one authorized channel."""
import io
import math

import numpy as np

from learned import MAX_WINDOWS
from senses.blobs import MAX_BYTES
from senses.contract import require


def waveform(raw, span, rate):
    import soundfile as sf
    from scipy.signal import resample_poly
    keys = ("start", "end", "rate", "channel")
    require(all(type(span.get(k)) is int for k in keys)
            and 0 <= span["start"] < span["end"] and span["rate"] > 0 and span["channel"] >= 0,
            "MNLM_NATIVE_AUDIO_SPAN", "explicit positive source sample range required")
    with sf.SoundFile(io.BytesIO(raw)) as audio:
        count = span["end"]-span["start"]
        require(span["media_type"] == "audio/"+audio.format.lower()
                and audio.samplerate == span["rate"] and span["channel"] < audio.channels
                and span["end"] <= len(audio), "MNLM_NATIVE_AUDIO_SOURCE", "source geometry differs")
        require(count*audio.channels*4 <= MAX_BYTES and count*rate <= MAX_WINDOWS*10*rate*span["rate"],
                "MNLM_NATIVE_AUDIO_BUDGET", "split the source span; do not truncate it")
        audio.seek(span["start"])
        original = audio.read(count, dtype="float32", always_2d=True)
    require(len(original) == count and bool(np.isfinite(original).all()),
            "MNLM_NATIVE_AUDIO_PCM", "invalid source samples")
    wave = np.ascontiguousarray(original[:, span["channel"]])
    divisor = math.gcd(rate, span["rate"])
    wave = resample_poly(wave, rate//divisor, span["rate"]//divisor,
                         window=("kaiser", 5.0), padtype="constant")
    require(len(wave) == (count*rate+span["rate"]-1)//span["rate"],
            "MNLM_NATIVE_AUDIO_RESAMPLE", "sample conservation failed")
    return wave.astype(np.float32, copy=False)


class ASTNumpyFrontend:
    """The upstream AST NumPy equations, selected explicitly regardless of torchaudio."""
    def __init__(self, config):
        from transformers.audio_utils import mel_filter_bank, window_function
        self.config = config
        require(config["sampling_rate"] == 16000 and config["num_mel_bins"] == 128
                and config["max_length"] == 1024 and config["do_normalize"] is True,
                "MNLM_NATIVE_AUDIO_AST_CONFIG", "unsupported AST frontend")
        self.filters = mel_filter_bank(num_frequency_bins=257, num_mel_filters=128,
            min_frequency=20, max_frequency=8000, sampling_rate=16000,
            norm=None, mel_scale="kaldi", triangularize_in_mel_space=True)
        self.window = window_function(400, "hann", periodic=False)

    def __call__(self, piece):
        import torch
        from transformers.audio_utils import spectrogram
        require(400 <= len(piece) <= 160000, "MNLM_NATIVE_AUDIO_SHORT", "AST needs >=25 ms")
        frames = spectrogram(piece, self.window, frame_length=400, hop_length=160,
            fft_length=512, power=2.0, center=False, preemphasis=0.97,
            mel_filters=self.filters, log_mel="log", mel_floor=1.192092955078125e-07,
            remove_dc_offset=True).T
        require(len(frames) <= 1024, "MNLM_NATIVE_AUDIO_TRUNCATION", "AST truncation prohibited")
        padded = np.pad(frames, ((0, 1024-len(frames)), (0, 0)))
        values = (padded-self.config["mean"])/(2*self.config["std"])
        return {"input_values": torch.from_numpy(values.astype(np.float32)[None])}


def features(name, extractor, piece, rate):
    require(len(piece) <= rate*10 and bool(np.isfinite(piece).all()),
            "MNLM_NATIVE_AUDIO_WINDOW", "finite pre-windowed audio required")
    if name == "ast":
        return extractor(piece)
    if name == "clap":
        # The oversized-input random-crop path is unreachable.
        return dict(extractor(piece, sampling_rate=rate, truncation="rand_trunc",
                              padding="pad", return_tensors="pt"))
    require(len(piece) >= 400, "MNLM_NATIVE_AUDIO_SHORT", "wave encoder needs >=25 ms")
    return dict(extractor(piece, sampling_rate=rate, padding=False,
                          truncation=False, return_tensors="pt"))
