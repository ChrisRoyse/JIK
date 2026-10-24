"""Retired diagnostic entry point; historical artifacts remain audit evidence.

The old implementation mislabelled classifier gains as Shannon lower bounds,
mixed populations across greedy steps, split sibling clusters across folds, and
could emit served=true without five independent senses. It is not an admission
instrument. Keeping it runnable would preserve a second, unsound success path.

Use scripts/lens/assay.py for current, sealed usable-information diagnostics.
Those diagnostics also remain unserved until Shannon admission is commissioned.
"""
import json
import sys


def main():
    print(json.dumps({
        "ok": False,
        "code": "MNLM_DIAGNOSTIC_RETIRED",
        "message": "assay_probe.py used invalid information bounds and could falsely report served=true.",
        "remediation": "Run scripts/lens/assay.py against current sealed inputs. It reports diagnostics only; admission remains uncommissioned.",
        "served": False,
        "shannon_conditional_certified": False,
    }), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
