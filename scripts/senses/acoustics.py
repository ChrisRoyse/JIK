"""Frozen frame measurements for arbitrary retained audio, independent of source.

No resampling, channel mixing or temporal pooling. Original bytes remain required.
The final short frame is retained at its real length; it is not padded into data.
"""
import numpy as np
import soundfile as sf

from .contract import field, require

CONTRACT = "acoustic-frames-20ms-v1"


def frames(path, sha256):
    with sf.SoundFile(path) as audio:
        rate, channels, total = audio.samplerate, audio.channels, len(audio)
        require(rate > 0 and channels > 0 and total > 0, "MNLM_AUDIO_EMPTY", str(path))
        width = max(1, round(rate * .020))
        start = 0
        geometry = {}
        while start < total:
            samples = audio.read(min(width, total - start), dtype="float64", always_2d=True)
            require(len(samples) > 0 and bool(np.isfinite(samples).all()), "MNLM_AUDIO_DECODE", str(path))
            # Frequency geometry and window are invariant for every complete frame.
            if len(samples) not in geometry:
                geometry[len(samples)] = np.hanning(len(samples)), np.fft.rfftfreq(len(samples), 1 / rate)
            spectrum_window, frequencies = geometry[len(samples)]
            for channel in range(channels):
                wave = samples[:, channel]
                power = np.abs(np.fft.rfft(wave * spectrum_window)) ** 2
                energy = float(power.sum())
                values = [field("sample_start", start, "integer", "sample"),
                          field("sample_count", len(wave), "integer", "sample"),
                          field("sample_rate", rate, "integer", "Hz"),
                          field("audio_channel", channel, "integer"),
                          field("rms", float(np.sqrt(np.mean(wave * wave))), "number", "full_scale"),
                          field("peak", float(np.abs(wave).max()), "number", "full_scale"),
                          field("zero_crossings", int(np.count_nonzero(np.diff(np.signbit(wave)))), "integer", "count"),
                          field("spectral_centroid", float((frequencies * power).sum() / energy) if energy else 0., "number", "Hz")]
                for band, bins in enumerate(np.array_split(power, 8)):
                    values.append(field(f"power_band_{band}", float(bins.sum()), "number", "fft_power"))
                yield {"modality": "audio", "sha256": sha256, "media_type": "audio/ogg" if audio.format == "OGG" else "audio/" + audio.format.lower(),
                       "measurements": values}
            start += len(samples)
        require(start == total, "MNLM_AUDIO_DECODE", "decoder did not consume the complete recording")


def runtime():
    return {"contract": CONTRACT, "numpy": np.__version__, "soundfile": sf.__version__,
            "libsndfile": sf.__libsndfile_version__, "frame_ms": 20, "resampling": False,
            "mixing": False, "pooling": False}
