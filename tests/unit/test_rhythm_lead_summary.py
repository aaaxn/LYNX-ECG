import pytest
from rhythm_lead_summary import variant_keys

pytestmark = pytest.mark.unit

INTERVENTIONS = {
    "intact": [],
    "template": [],
    "template_rr": [],
    "shuffled_0": [],
    "shuffled_1": [],
    "rr_regularized": [],
    "rr_irregularized_0": [],
    "rr_irregularized_1": [],
}


def test_template_family_excludes_template_rr():
    assert variant_keys(INTERVENTIONS, "template") == ["template"]
    assert variant_keys(INTERVENTIONS, "template_rr") == ["template_rr"]


def test_random_families_keep_every_draw():
    assert variant_keys(INTERVENTIONS, "shuffled") == ["shuffled_0", "shuffled_1"]
    assert variant_keys(INTERVENTIONS, "rr_regularized") == ["rr_regularized"]
    assert variant_keys(INTERVENTIONS, "rr_irregularized") == [
        "rr_irregularized_0",
        "rr_irregularized_1",
    ]
