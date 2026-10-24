"""Frozen native audio candidates: ordered temporal rows and frequency-time grids."""
import hashlib
import inspect
import json
from pathlib import Path
import platform
import time

import numpy as np

from learned import MAX_WINDOWS, check_gpu
from native_assets import PLAN_PIN, bind, read_assets, state_sha
from native_audio_frontend import ASTNumpyFrontend, features, waveform
from senses.contract import require

SPECS = {
    "wavlm": dict(repo="microsoft/wavlm-base-plus", rate=16000, dim=768),
    "hubert": dict(repo="facebook/hubert-base-ls960", rate=16000, dim=768),
    "ast": dict(repo="MIT/ast-finetuned-audioset-10-10-0.4593", rate=16000, dim=768),
    "clap": dict(repo="laion/clap-htsat-unfused", rate=48000, dim=768),
}
KERNELS, STRIDES = (10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)


def geometry(name, count):
    require(name in SPECS and type(count) is int and 0 < count <= 10*SPECS[name]["rate"],
            "MNLM_NATIVE_AUDIO_GEOMETRY", "invalid native audio window")
    if name in ("wavlm", "hubert"):
        length = count
        for kernel, stride in zip(KERNELS, STRIDES):
            length = (length-kernel)//stride+1
        require(length > 0, "MNLM_NATIVE_AUDIO_SHORT", "native convolution needs >=25 ms")
        return dict(shape=[length, 768], grid=[length], special_rows=0,
                    order="temporal convolution rows", sample_stride=320, receptive_samples=400)
    if name == "ast":
        return dict(shape=[1214, 768], grid=[12, 101], special_rows=2,
                    order="CLS, distillation, then frequency-major time patches",
                    padded_mel_frames=1024, valid_mel_frames=(count-400)//160+1)
    return dict(shape=[64, 768], grid=[2, 32], special_rows=0,
                order="frequency-major native HTSAT time cells",
                padded_samples=480000, padding="right zero; no repeats or random crop")


def construct(name, config, processor, tensors):
    from transformers import (WavLMConfig, WavLMModel, HubertConfig, HubertModel,
        ASTConfig, ASTModel, ClapAudioConfig, ClapAudioModel, Wav2Vec2FeatureExtractor,
        ClapFeatureExtractor)
    if name in ("wavlm", "hubert"):
        cfg = (WavLMConfig if name == "wavlm" else HubertConfig).from_dict(config)
        cfg._attn_implementation = "eager"
        require(tuple(cfg.conv_kernel) == KERNELS and tuple(cfg.conv_stride) == STRIDES
                and cfg.hidden_size == 768, "MNLM_NATIVE_AUDIO_CONFIG", "unsupported conv geometry")
        model = (WavLMModel if name == "wavlm" else HubertModel)(cfg)
        expected = set(model.state_dict())
        def selected(k):
            if k in expected:
                return k
            return k.replace("pos_conv_embed.conv.weight_g", "pos_conv_embed.conv.parametrizations.weight.original0"
                ).replace("pos_conv_embed.conv.weight_v", "pos_conv_embed.conv.parametrizations.weight.original1")
        excluded = lambda k: False
        extractor = Wav2Vec2FeatureExtractor.from_dict(processor)
    elif name == "ast":
        cfg = ASTConfig.from_dict(config)
        cfg._attn_implementation = "eager"
        require((cfg.num_mel_bins, cfg.max_length, cfg.patch_size, cfg.frequency_stride,
                 cfg.time_stride, cfg.hidden_size) == (128, 1024, 16, 10, 10, 768),
                "MNLM_NATIVE_AUDIO_CONFIG", "unsupported AST grid")
        model = ASTModel(cfg)
        selected = lambda k: k.removeprefix("audio_spectrogram_transformer.") if k.startswith(
            "audio_spectrogram_transformer.") else None
        excluded = lambda k: k.startswith("classifier.")
        extractor = ASTNumpyFrontend(processor)
    else:
        cfg = ClapAudioConfig.from_dict(config["audio_config"])
        require(cfg.enable_fusion is False and cfg.hidden_size == 768 and cfg.spec_size == 256
                and cfg.num_mel_bins == 64 and cfg.patch_stride == [4, 4]
                and cfg.depths == [2, 2, 6, 2], "MNLM_NATIVE_AUDIO_CONFIG", "unsupported CLAP grid")
        model = ClapAudioModel(cfg)
        selected = lambda k: k.removeprefix("audio_model.") if k.startswith("audio_model.") else None
        excluded = lambda k: k.startswith(("text_model.", "text_projection.", "audio_projection.")) or k in (
            "logit_scale_a", "logit_scale_t")
        extractor = ClapFeatureExtractor.from_dict(processor)
    return model, extractor, cfg, bind(model, tensors, selected, excluded,
                                      rename_family="vit" if name == "ast" else None)


class NativeAudioEncoder:
    def __init__(self, name, *, asset_root, plan_path):
        import torch
        import transformers
        import scipy
        import soundfile as sf
        import transformers.audio_utils as audio_utils
        require(name in SPECS, "MNLM_NATIVE_AUDIO_NAME", "unknown native audio candidate")
        check_gpu()
        self.name, self.spec = name, SPECS[name]
        t = time.perf_counter()
        row, config, processor, tensors = read_assets(asset_root, plan_path, self.spec["repo"])
        self.load_phases = dict(asset_read=time.perf_counter()-t)
        t = time.perf_counter()
        self.model, self.extractor, cfg, layout = construct(name, config, processor, tensors)
        del tensors
        self.model.eval().requires_grad_(False).to(device="cuda", dtype=torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        self.load_phases["construct_bind_cuda"] = time.perf_counter()-t
        modules = dict(model=inspect.getfile(type(self.model)), audio_utils=inspect.getfile(audio_utils),
                       frontend=inspect.getfile(type(self.extractor)))
        self.identity = dict(name="ucr-native-"+name+"-audio", modality="audio", shape="Multi(var,768)",
            dtype="f32", norm="none", frozen=True, repo=row["repo"], revision=row["revision"],
            assets=row["files"], asset_plan=PLAN_PIN, checkpoint_layout=layout,
            effective_config=json.loads(cfg.to_json_string(use_diff=False)),
            preprocessing=dict(**self.spec, window_seconds=10, max_windows=MAX_WINDOWS,
                processor=processor, channels="separate; never mixed",
                resample="scipy.resample_poly;kaiser=5;constant-zero;authorized-span-only",
                ast_frontend="explicit transformers NumPy spectrogram;400/160/512;128 kaldi mel bins",
                clap_padding="right-zero; overrides checkpoint repeatpad; no random crop",
                output="all native rows including declared padding/special rows; no pooling",
                execution="one channel-window float32 CUDA; explicit CPU frontend"),
            runtime={**check_gpu(), "python": platform.python_version(), "torch": torch.__version__,
                "transformers": transformers.__version__, "scipy": scipy.__version__,
                "numpy": np.__version__, "soundfile": sf.__version__,
                "libsndfile": sf.__libsndfile_version__, "platform": platform.platform()},
            dependency_sha256={k: hashlib.sha256(Path(p).read_bytes()).hexdigest() for k, p in modules.items()},
            code_sha256={n: hashlib.sha256((Path(__file__).parents[1]/n).read_bytes()).hexdigest()
                for n in ("lens/native_audio.py", "lens/native_audio_frontend.py", "lens/native_assets.py",
                          "lens/learned.py", "senses/blobs.py", "senses/contract.py")},
            pretraining_overlap="upstream training bytes unavailable; unassessed", scientific_admission=False)
        self.identity["state_sha256"] = state_sha(self.model)

    def embed(self, regions, blobs):
        import torch
        require(torch.are_deterministic_algorithms_enabled()
                and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
                and not torch.backends.cudnn.benchmark, "MNLM_NATIVE_AUDIO_RUNTIME", "restore pinned runtime")
        rate, parts, windows, offset = self.spec["rate"], [], [], 0
        count = sum(((s["end"]-s["start"])+s["rate"]*10-1)//(s["rate"]*10) for s in regions)
        require(count <= MAX_WINDOWS, "MNLM_NATIVE_AUDIO_BUDGET", "too many native windows; no truncation")
        phases = dict(source=0., frontend=0., forward=0., materialize=0.)
        for span in regions:
            t = time.perf_counter()
            wave = waveform(blobs.read(span["sha256"], span["media_type"]), span, rate)
            phases["source"] += time.perf_counter()-t
            for start in range(0, len(wave), rate*10):
                t = time.perf_counter()
                piece = wave[start:start+rate*10]
                g = geometry(self.name, len(piece))
                batch = features(self.name, self.extractor, piece, rate)
                batch = {k: v.to("cuda") for k, v in batch.items()}
                torch.cuda.synchronize()
                phases["frontend"] += time.perf_counter()-t
                t = time.perf_counter()
                with torch.inference_mode():
                    values = self.model(**batch).last_hidden_state[0]
                    if self.name == "clap":
                        require(list(values.shape) == [768, *g["grid"]],
                                "MNLM_NATIVE_AUDIO_GRID", "unexpected frequency-time grid")
                        values = values.permute(1, 2, 0).reshape(-1, 768)
                torch.cuda.synchronize()
                phases["forward"] += time.perf_counter()-t
                t = time.perf_counter()
                array = values.cpu().numpy().copy()
                require(list(array.shape) == g["shape"] and array.dtype == np.float32
                        and bool(np.isfinite(array).all()), "MNLM_NATIVE_AUDIO_OUTPUT", "invalid native rows")
                parts.append(array)
                windows.append(dict(**span, **g, target_rate=rate, resampled_start=start,
                    resampled_end=start+len(piece), token_start=offset, token_end=offset+len(array),
                    original_coordinate_mapping="span.start + resampled_index * source_rate / target_rate"))
                offset += len(array)
                phases["materialize"] += time.perf_counter()-t
        self.last_phases = phases
        return (np.concatenate(parts) if parts else np.empty((0, 768), np.float32)), windows
