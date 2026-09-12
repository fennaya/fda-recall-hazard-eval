"""Tests for the agent's output handling and the drug-name heuristic.

No network and no model: these cover the parsing that sits between a provider's
response and a scored prediction, where a silent failure would turn into a
wrong label rather than an error.
"""

from __future__ import annotations

import json

import pytest

from fda_hazard.agent import (
    _normalise_class,
    _parse_loose_json,
    _result_from_args,
    prompt_sha,
)
from fda_hazard.drug_context import extract_drug_name

I, II, III = "Class I", "Class II", "Class III"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Class I", I),
        ("class i", I),
        ("CLASS III", III),
        ("  Class II  ", II),
        ("I", I),
        ("ii", II),
        ("3", III),
        ("Class 2", II),
    ],
)
def test_class_normalisation(raw, expected):
    label, error = _normalise_class(raw)
    assert label == expected
    assert error is None


def test_unparseable_class_falls_back_and_reports_an_error():
    """A garbage label must not be silently scored as a confident answer."""
    label, error = _normalise_class("Class IV")
    assert label == II  # most common class, so the fallback is the least-harm guess
    assert error is not None and "unparseable" in error


def test_non_string_class_reports_an_error():
    label, error = _normalise_class(None)
    assert label == II
    assert error is not None


def test_result_from_args_clamps_confidence():
    r = _result_from_args({"classification": I, "confidence": 5.0}, [], [], 0, True)
    assert r.confidence == 1.0
    r = _result_from_args({"classification": I, "confidence": -2}, [], [], 0, True)
    assert r.confidence == 0.0


def test_result_from_args_handles_bad_confidence():
    r = _result_from_args(
        {"classification": I, "confidence": "high"}, [], [], 0, True
    )
    assert r.confidence == 0.0


def test_result_from_args_coerces_non_list_citations():
    r = _result_from_args(
        {"classification": II, "confidence": 0.5, "precedents_cited": "D-1"},
        [], [], 0, True,
    )
    assert r.precedents_cited == ["D-1"]


def test_loose_json_recovers_a_json_block():
    text = 'Here is my answer: {"classification": "Class I", "confidence": 0.9, ' \
           '"reasoning": "sterility", "precedents_cited": []} done'
    parsed = _parse_loose_json(text)
    assert parsed["classification"] == I


def test_loose_json_recovers_from_prose():
    parsed = _parse_loose_json("I believe this is a Class III recall, minor labeling.")
    assert parsed["classification"] == III
    assert parsed["_recovered"] is True
    assert parsed["confidence"] < 0.5  # a recovered answer is not a confident one


def test_loose_json_returns_none_on_nothing_usable():
    assert _parse_loose_json("") is None
    assert _parse_loose_json("I cannot determine this.") is None


def test_prompt_sha_is_stable_and_short():
    assert prompt_sha() == prompt_sha()
    assert len(prompt_sha()) == 12


@pytest.mark.parametrize(
    "description,expected_token",
    [
        ("Progesterone 100 mg/mL in Corn Oil Injection, 2 mL vials, Rx only", "Progesterone"),
        ("Metformin Hydrochloride Tablets, USP 500 mg, 100 count bottle", "Metformin"),
        ("Atorvastatin Calcium Tablets 20 mg", "Atorvastatin"),
    ],
)
def test_extract_drug_name_finds_the_active_ingredient(description, expected_token):
    name = extract_drug_name(description)
    assert expected_token.lower() in name.lower()


def test_extract_drug_name_strips_units_and_boilerplate():
    name = extract_drug_name("Ibuprofen Tablets USP 200 mg, Rx only, 500 count")
    assert "mg" not in name.lower().split()
    assert "tablets" not in name.lower()
    assert "ibuprofen" in name.lower()


def test_extract_drug_name_on_junk_returns_empty_not_error():
    assert extract_drug_name("500 mg") == ""
    assert extract_drug_name("") == ""


# -- tool result serialisation ---------------------------------------------
# Regression guard: slicing json.dumps(...) mid-string sent the model invalid
# JSON whenever a precedent set was large, which happened on real data.


def _precedents(n, text_len=2000):
    return {
        "count": n,
        "precedents": [
            {
                "record_key": f"D-{i}-2020",
                "classification": "Class II",
                "reason_for_recall": "x" * text_len,
                "product_description": "y" * text_len,
                "similarity": 0.5,
            }
            for i in range(n)
        ],
    }


def test_serialised_tool_result_is_always_valid_json():
    from fda_hazard.agent import serialise_tool_result

    for n in (0, 1, 3, 5, 10, 40):
        blob = serialise_tool_result(_precedents(n))
        json.loads(blob)  # must not raise


def test_serialised_tool_result_respects_the_budget():
    from fda_hazard.agent import TOOL_RESULT_BUDGET, serialise_tool_result

    blob = serialise_tool_result(_precedents(10, 5000))
    assert len(blob) <= TOOL_RESULT_BUDGET
    json.loads(blob)


def test_small_results_pass_through_intact():
    from fda_hazard.agent import serialise_tool_result

    result = {"found": True, "query": "metformin"}
    assert json.loads(serialise_tool_result(result)) == result


def test_precedents_are_dropped_from_the_end_not_cut_mid_string():
    from fda_hazard.agent import serialise_tool_result

    parsed = json.loads(serialise_tool_result(_precedents(20, 4000)))
    assert parsed.get("truncated") is True
    assert len(parsed["precedents"]) < 20
    # Whatever survived must be whole records, not fragments.
    for p in parsed["precedents"]:
        assert set(p) >= {"record_key", "classification", "similarity"}
