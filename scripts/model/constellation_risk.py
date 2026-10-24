"""Training-only expected native distance, with one cost array per instrument."""
import torch

from claims.codata_source import check


OBJECTIVE = "typed_native_region_expected_risk_v1"


def expected(output, costs, allowed, specs):
    """Categorical probabilities mix scalar losses, never native meaning vectors."""
    evidence = output["constellation_regions"]
    check(set(costs) == set(output["regions"]) == {s.name for s in specs},
          "REGION_RISK_ROSTER")
    check(allowed.dtype == torch.bool and bool(allowed.any())
          and evidence.shape == (1, len(allowed))
          and evidence.device == allowed.device, "REGION_RISK_MASK")
    check(bool(torch.isfinite(evidence[:, allowed]).all())
          and bool(torch.isneginf(evidence[:, ~allowed]).all()), "REGION_RISK_MASK")
    probabilities = torch.softmax(evidence[:, allowed], dim=-1)[0]
    risks = {}
    for spec in specs:
        identity, values = costs[spec.name]
        check(identity == spec.identity and isinstance(values, torch.Tensor)
              and values.shape == allowed.shape and values.device == evidence.device
              and values.dtype == evidence.dtype and not values.requires_grad,
              "REGION_RISK_INSTRUMENT")
        check(bool(torch.isfinite(values).all()) and bool((values >= 0).all()),
              "REGION_RISK_COST")
        risks[spec.name] = (probabilities*values[allowed]).sum()
    return risks
