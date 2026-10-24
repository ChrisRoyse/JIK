"""Native medium floor: actual historical panel plus labelled boundary controls.

Generated control identities exercise counting only; they are NOT instruments,
observations, admission evidence or a claim that image/audio panels exist.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from production.gate import require_admitted, require_native_floor, LensError
from text.evaluation_multiscale import file_sha
from senses.trajectory_population import publish


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    before = file_sha(a.panel); actual = json.loads(a.panel.read_bytes()); checks = []
    publish(a.output/"actual-panel.json", actual)
    actual = json.loads((a.output/"actual-panel.json").read_bytes())
    def check(name, ok, **details): checks.append(dict(name=name, passed=bool(ok), **details))
    def refuses(name, code, fn):
        observed = None
        try: fn()
        except LensError as exc: observed = exc.code
        check(name, code == observed, expected=code, observed=observed)
    counts = {m:len({r["contract"]["weights_sha256"] for r in actual["lenses"] if r["contract"].get("modality") == m})
              for m in ("text", "image", "audio")}
    check("B1 actual historical bank is not a native panel", len(actual["lenses"]) >= 5 and any(v < 5 for v in counts.values()),
          bank_count=len(actual["lenses"]), native_counts=counts)
    refuses("B2 historical global bank cannot cover three media", "MNLM_STAGE4_NATIVE_FLOOR",
            lambda: require_native_floor(actual, ["text", "image", "audio"]))
    legacy_root = a.output/"historical"; legacy_root.mkdir(); publish(legacy_root/"panel.json", actual)
    refuses("B3 historical production still unadmitted", "MNLM_STAGE4_PANEL_UNADMITTED",
            lambda: require_admitted(legacy_root, None, ["text", "image", "audio"]))
    def fixture(amounts):
        return dict(control_only=True, lenses=[dict(name=f"control-{m}-{i}", contract=dict(modality=m,
            weights_sha256=hashlib.sha256(f"control-only-{m}-{i}".encode()).hexdigest()))
            for m,n in amounts.items() for i in range(n)])
    for name, amount, passes in (("four",4,False), ("five",5,True), ("six",6,True), ("eight",8,True)):
        path = a.output/(name+"-control.json"); publish(path, fixture(dict(text=amount,image=amount,audio=amount)))
        control = json.loads(path.read_bytes())
        if passes:
            check("B2 "+name+" has no artificial ceiling", require_native_floor(control,["text","image","audio"]) == dict(text=amount,image=amount,audio=amount))
        else:
            refuses("B2 four per medium refuses", "MNLM_STAGE4_NATIVE_FLOOR",
                    lambda: require_native_floor(control,["text","image","audio"]))
    control = fixture(dict(text=12,image=4,audio=5)); publish(a.output/"mixed-control.json", control)
    refuses("B2 text surplus cannot fill image deficit", "MNLM_STAGE4_NATIVE_FLOOR",
            lambda: require_native_floor(control,["text","image","audio"]))
    control = fixture(dict(text=5)); control["lenses"][-1]["contract"]["weights_sha256"] = control["lenses"][0]["contract"]["weights_sha256"]
    publish(a.output/"renamed-control.json", control)
    refuses("B2 duplicate weights do not create a sense", "MNLM_STAGE4_NATIVE_FLOOR",
            lambda: require_native_floor(control,["text"]))
    control = fixture(dict(text=12))
    for r in control["lenses"]: r["name"] = "image-caption-"+r["name"]
    publish(a.output/"caption-control.json", control)
    refuses("B2 names cannot turn text into images", "MNLM_STAGE4_NATIVE_FLOOR",
            lambda: require_native_floor(control,["image"]))
    control = fixture(dict(audio=5)); control["lenses"][0]["contract"].pop("weights_sha256")
    publish(a.output/"unpinned-control.json", control)
    refuses("B2 unpinned instrument refuses", "MNLM_STAGE4_NATIVE_IDENTITY",
            lambda: require_native_floor(control,["audio"]))
    control = fixture(dict(text=5))
    refuses("B3 implicit media scope refuses", "MNLM_STAGE4_MEDIA_SCOPE", lambda: require_native_floor(control,None))
    false_root = a.output/"false-certificate"; false_root.mkdir()
    control.update(contract="universal-panel-v1", gates=dict(shannon_conditional_certified=True,families_served_ge3=True))
    publish(false_root/"panel.json", control)
    refuses("B3 global five flags cannot omit native scope", "MNLM_STAGE4_MEDIA_SCOPE",
            lambda: require_admitted(false_root,None))
    # Merely satisfying counts is never sufficient to open production.
    try: require_admitted(false_root,None,["text"])
    except LensError as exc: check("B3 counts alone cannot certify admission", True, refusal=exc.code)
    else: check("B3 counts alone cannot certify admission", False)
    check("B1 real historical panel unchanged", file_sha(a.panel) == before)
    report = dict(author_checked=True, source_sha256=before, source_instruments=len(actual["lenses"]),
                  source_native_counts=counts, scientific_admission=False, checks=checks,
                  passed=all(r["passed"] for r in checks), control_scope="engineering-boundaries-only")
    publish(a.output/"report.json", report); print(json.dumps(report))
    if not report["passed"]: raise SystemExit(2)


if __name__ == "__main__": main()
