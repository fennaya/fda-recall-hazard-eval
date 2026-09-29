"""Tests for the Step 4 distribution-breadth bucketing rule.

Pure logic, no DB, no model. The rule is fixed and descriptive-only -- these
tests just confirm it behaves as documented in analysis/distribution_buckets.py
on the real phrasing variants seen in the corpus.
"""

from __future__ import annotations

from analysis.distribution_buckets import bucket_distribution


def test_none_and_empty_are_other():
    assert bucket_distribution(None) == "other"
    assert bucket_distribution("") == "other"


def test_nationwide_variants_all_match():
    variants = [
        "Nationwide in the USA", "US Nationwide.", "U.S. Nationwide",
        "Nationwide within the United States", "USA Nationwide",
        "U.S.A. Nationwide", "Nationwide within the U.S", "Nationwide",
        "Distributed Nationwide in the USA",
        "US Nationwide , Alaska, and Puerto Rico.",
        "Nationwide in the USA and Antigua",
    ]
    for v in variants:
        assert bucket_distribution(v) == "nationwide", v


def test_single_state():
    assert bucket_distribution("MA") == "single_state"
    assert bucket_distribution("Within U.S. - Tennessee") in ("single_state", "other")


def test_multi_state():
    assert bucket_distribution("DE and NC") == "multi_state"
    assert bucket_distribution("CA, CO, FL, PR, WA") == "multi_state"


def test_territories_count_as_states():
    assert bucket_distribution("PR") == "single_state"
    assert bucket_distribution("PR and VI") == "multi_state"


def test_non_state_text_is_other():
    assert bucket_distribution("Product was distributed via the internet.") == "other"
    assert bucket_distribution("Within U.S") == "other"


def test_nationwide_takes_priority_over_state_tokens():
    """"Nationwide ... and Antigua" must not be miscounted as a state list."""
    assert bucket_distribution("Nationwide, including HI and AK") == "nationwide"


def test_rule_is_case_insensitive_on_nationwide():
    assert bucket_distribution("NATIONWIDE") == "nationwide"
    assert bucket_distribution("nationwide") == "nationwide"
