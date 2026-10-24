"""Source-relative waveform correspondence, retaining channels and uncertain time.

This is an audit of two recordings, never an admitted sense or physical AV clock
calibration. PC-07/28/35: inputs are immutable; index intervals once; no dense
sample-by-lag matrix, resampling, channel averaging or silent extrapolation.
"""
from bisect import bisect_right
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import correlate

from senses.perception_corpus import pinned, sha
from senses.contract import digest

POLICY = dict(window_seconds=[1, 2], search_seconds=[1, 4], minimum_rms=.0001,
    minimum_correlation=.99, alternative_exclusion_seconds=[1, 1000],
    minimum_margin=.05, near_peak_delta=.01, maximum_lag_disagreement_samples=1,
    minimum_windows_per_channel=3, consensus="lower_median_of_qualified_window_lags",
    lag_convention="wav_sample = mp4_sample + lag",
    scope="observed_waveform_correspondence_only", physical_sync_calibrated=False)


def need(ok, code):
    if not ok:
        raise ValueError("MNLM_AUDIO_ALIGNMENT_"+code)


def fraction(value):
    return [value.numerator, value.denominator]


def direct(x, y):
    """Separate direct computation of the selected fit; offset and gain remain visible."""
    x, y = x.astype(np.float64), y.astype(np.float64)
    x0, y0 = x-x.mean(), y-y.mean()
    xx, yy = float(x0@x0), float(y0@y0)
    if xx <= 0 or yy <= 0:
        return dict(correlation=-1., gain=None, intercept=None, residual_rms=None)
    gain = float(x0@y0)/xx
    intercept = float(y.mean()-gain*x.mean())
    return dict(correlation=float(x0@y0)/np.sqrt(xx*yy), gain=gain, intercept=intercept,
                residual_rms=float(np.sqrt(np.mean((y-gain*x-intercept)**2))))


def profile(mp4, wav, start, end, rate):
    need(mp4.ndim == wav.ndim == 1 and 0 <= start < end <= len(mp4)
         and type(rate) == int and rate > 0, "PROFILE_INPUT")
    width, radius = end-start, rate//4
    lo, hi = max(0, start-radius), min(len(wav), end+radius)
    need(hi-lo >= width, "PROFILE_OVERLAP")
    x, y = mp4[start:end].astype(np.float64), wav[lo:hi].astype(np.float64)
    x0 = x-x.mean()
    energy = float(x0@x0)
    rms = float(np.sqrt(energy/width))
    sums = np.concatenate(([0.], np.cumsum(y)))
    squares = np.concatenate(([0.], np.cumsum(y*y)))
    sy = sums[width:]-sums[:-width]
    ey = np.maximum(squares[width:]-squares[:-width]-sy*sy/width, 0.)
    numerator = correlate(y, x0, mode="valid", method="fft")
    denominator = np.sqrt(energy*ey)
    scores = np.full(len(denominator), -1., dtype=np.float64)
    np.divide(numerator, denominator, out=scores, where=denominator > 0)
    scores = np.clip(scores, -1., 1.)
    lags = np.arange(len(scores), dtype=np.int64)+lo-start
    best = int(np.argmax(scores))
    alternatives = abs(lags-lags[best])*1000 > rate
    need(bool(alternatives.any()), "NO_ALTERNATIVE_LAGS")
    competitor = int(np.flatnonzero(alternatives)[np.argmax(scores[alternatives])])
    near = lags[scores >= scores[best]-POLICY["near_peak_delta"]]
    lag = int(lags[best])
    selected = direct(x, wav[start+lag:end+lag])
    need(abs(selected["correlation"]-scores[best]) <= 1e-8, "DIRECT_REPLAY")
    margin = float(scores[best]-scores[competitor])
    accepted = (rms >= POLICY["minimum_rms"] and scores[best] >= POLICY["minimum_correlation"]
                and margin >= POLICY["minimum_margin"])
    return scores, dict(mp4_samples=[start, end], wav_samples=[start+lag, end+lag],
        lag_samples=lag, lag_seconds=fraction(Fraction(lag, rate)),
        profile_lag_start=int(lags[0]), profile_lag_end=int(lags[-1]),
        peak=float(scores[best]), competitor_lag=int(lags[competitor]),
        competitor_score=float(scores[competitor]), margin=margin, template_rms=rms,
        near_peak_lag_range=[int(near.min()), int(near.max())],
        direct=selected, window_qualified=bool(accepted),
        unresolved_reasons=[name for name, failed in (
            ("low_signal", rms < POLICY["minimum_rms"]),
            ("low_correlation", scores[best] < POLICY["minimum_correlation"]),
            ("ambiguous_lag", margin < POLICY["minimum_margin"])) if failed])


def summary(rows, channels):
    good = [r for r in rows if r["window_qualified"]]
    lags = sorted(r["lag_samples"] for r in good)
    lag = lags[(len(lags)-1)//2] if lags else None
    enough = all(sum(r["channel"] == c for r in good) >= POLICY["minimum_windows_per_channel"]
                 for c in range(channels))
    consistent = bool(lags) and max(abs(v-lag) for v in lags) <= 1
    return dict(offset_samples=lag, original_channels=channels,
        qualified_windows=len(good), unresolved_windows=len(rows)-len(good),
        sufficient_anchors=bool(enough), consistent_offset=bool(consistent),
        correspondence_supported=bool(enough and consistent),
        physical_sync_calibrated=False, source_waveforms_identical=False)


class Alignment:
    """Map covered provider WAV samples into native MP4 blocks, with explicit ambiguity."""
    def __init__(self, root, report_pin):
        self.root, self.pin = Path(root), report_pin
        report = pinned(self.root/"report.json", report_pin)
        need(report["schema"] == "source-audio-alignment-v1" and report["policy"] == POLICY
             and not report["scientific_admission"] and not report["physical_sync_calibrated"],
             "REPORT_SCOPE")
        self.records = {}
        self.intervals = {}
        self.native = {}
        for clip in report["clips"]:
            need(clip["clip"] not in self.records and clip["summary"] ==
                 summary(clip["windows"], clip["channels"]), "SUMMARY")
            self.records[clip["clip"]] = clip
            graph = pinned(Path(clip["graph_file"]), clip["graph_sha256"])
            need(graph["clip"] == clip["clip"] and graph["blob_sha256"] ==
                 clip["sources"]["mp4"]["source_sha256"]
                 and digest(graph["clock"]) == clip["native_clock_pin"], "GRAPH_BINDING")
            nodes = [n for n in graph["nodes"] if n["modality"] == "audio"]
            offsets, cursor = [], 0
            for node in nodes:
                offsets.append(cursor)
                need(node["sample_count"] > 0, "NATIVE_SAMPLES")
                cursor += node["sample_count"]
            need(cursor == clip["sources"]["mp4"]["samples"], "NATIVE_SAMPLE_COUNT")
            self.native[clip["clip"]] = (graph, nodes, offsets)
            by_channel = {}
            for row in clip["windows"]:
                p = self.root/row["profile"]
                need(sha(p) == row["profile_sha256"] and p.stat().st_size ==
                     (row["profile_lag_end"]-row["profile_lag_start"]+1)*8, "PROFILE_BYTES")
                if row["window_qualified"]:
                    by_channel.setdefault(row["channel"], []).append(row)
            for channel, windows in by_channel.items():
                windows.sort(key=lambda r: r["wav_samples"][0])
                self.intervals[clip["clip"], channel] = (
                    [r["wav_samples"][0] for r in windows], windows)

    def project(self, clip, span, *, wav_sha256, mp4_sha256, clock_id, split_group,
                required_channels, purpose="source_audit"):
        need(purpose == "source_audit", "ADMISSION_REQUIRED")
        need(clip in self.records, "UNKNOWN_CLIP")
        record = self.records[clip]
        graph, nodes, offsets = self.native[clip]
        need(wav_sha256 == record["sources"]["wav"]["source_sha256"], "WAV_SOURCE")
        need(mp4_sha256 == graph["blob_sha256"] and clock_id == graph["clock"]["id"]
             and split_group == graph["split_group"], "NATIVE_SOURCE_CLOCK")
        need(type(span) == list and len(span) == 2 and all(type(v) == int for v in span)
             and 0 <= span[0] < span[1] <= record["sources"]["wav"]["samples"], "SAMPLE_INTERVAL")
        need(type(required_channels) == list and all(type(c) == int for c in required_channels)
             and required_channels == list(range(record["channels"])), "MISSING_CHANNEL")
        need(record["summary"]["correspondence_supported"], "CORRESPONDENCE_UNRESOLVED")
        support, mapped = [], []
        for channel in required_channels:
            starts, windows = self.intervals[clip, channel]
            index = max(0, bisect_right(starts, span[0])-1)
            cursor = span[0]
            used = []
            while index < len(windows) and cursor < span[1]:
                row = windows[index]
                start, end = row["wav_samples"]
                need(start <= cursor < end, "UNCOVERED_INTERVAL")
                stop = min(end, span[1])
                lag = row["lag_samples"]
                used.append(dict(window=row["index"], channel=channel,
                    wav_samples=[cursor, stop], mp4_samples=[cursor-lag, stop-lag],
                    empirical_lag_range=list(row["near_peak_lag_range"])))
                mapped.append([cursor-lag, stop-lag])
                cursor = stop
                index += 1
            need(cursor == span[1], "UNCOVERED_INTERVAL")
            support.extend(used)
        native = []
        for start, end in sorted(set(map(tuple, mapped))):
            i = max(0, bisect_right(offsets, start)-1)
            position = start
            while i < len(nodes) and position < end:
                n, base = nodes[i], offsets[i]
                stop = min(end, base+n["sample_count"])
                need(base <= position < stop, "NATIVE_SAMPLE_GAP")
                rate = Fraction(n["sample_count"], 1)/(
                    Fraction(*n["span"][1])-Fraction(*n["span"][0]))
                need(rate == record["rate"], "NATIVE_RATE")
                native.append(dict(node=n["id"], sample_in_block=[position-base, stop-base],
                    span=[fraction(Fraction(*n["span"][0])+Fraction(position-base, record["rate"])),
                          fraction(Fraction(*n["span"][0])+Fraction(stop-base, record["rate"]))],
                    available_after=n["available_after"]))
                position, i = stop, i+1
            need(position == end, "NATIVE_SAMPLE_GAP")
        return dict(clip=clip, wav_samples=span, wav_sha256=wav_sha256,
            mp4_sha256=graph["blob_sha256"], clock=deepcopy(graph["clock"]), split_group=graph["split_group"],
            alignment_report=self.pin, support=support, native=native,
            mapping_scope=POLICY["scope"], physical_sync_calibrated=False,
            scientific_admission=False, annotation_truth_established=False)
