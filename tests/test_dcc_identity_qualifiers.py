"""Generic qualification guards use artificial names and the ISO name data."""

import json

import pycountry
import pytest

from slime_plugins.dcc.widesearch_metrics import (
    _align,
    _has_missing_identity_qualifier,
    _verify_entity_identity,
)


@pytest.mark.parametrize("origin,candidate", [
    ("Solstice Archive", "Solstice Archive (novel)"),
    ("Solstice Archive", "Solstice Archive (1968 novel)"),
    ("Solstice Archive", "Solstice Archive (novel, 1968)"),
    ("Solstice Archive", "Solstice Archive (film)"),
    ("Solstice Archive", "Solstice Archive (television series)"),
    ("Solstice Archive", "Solstice Archive (video game)"),
    ("Solstice Archive", "Solstice Archive (1968)"),
    ("Solstice Games", "Solstice Games 2040"),
    ("Solstice Games", "Solstice Games: 2040"),
    ("Solstice Atlas", "Solstice Atlas: Third Edition"),
    ("Solstice Atlas", "Solstice Atlas (Edition 3)"),
    ("Solstice Atlas", "Solstice Atlas 2023 edition"),
    ("Solstice Engine", "Solstice Engine Version 2.1"),
    ("Solstice Engine", "Solstice Engine (v2.1)"),
    ("Solstice Archive", "Solstice Archive (novel) (1968)"),
    ("Solstice Archive (novel)", "Solstice Archive (novel) 1968"),
    (" ＳＯＬＳＴＩＣＥ\tArchive ", "solstice archive (FILM)"),
])
def test_missing_explicit_qualifier_returns_effective_uncertain_without_model(origin, candidate):
    def judge(prompt):
        pytest.fail("A deterministic guard decision is not a raw model verdict")

    assert _has_missing_identity_qualifier(origin, candidate)
    assert _verify_entity_identity(origin, candidate, judge) == "UNCERTAIN"


def test_country_and_subdivision_suffixes_are_driven_by_general_iso_name_data():
    names = {
        getattr(country, field)
        for country in pycountry.countries
        for field in ("name", "official_name", "common_name")
        if hasattr(country, field)
    }
    # Cover the full country name table and a spread across the subdivision
    # table, including names with commas and non-ASCII characters.
    names.update(subdivision.name for index, subdivision in enumerate(pycountry.subdivisions) if index % 97 == 0)
    assert len(names) > 250
    for name in names:
        assert _has_missing_identity_qualifier("Solstice Point", "Solstice Point, " + name), name


def test_compound_and_parenthetical_geography_preserve_the_complete_base():
    country = pycountry.countries.get(alpha_2="CA").name
    subdivision = pycountry.subdivisions.get(code="CA-BC").name
    for suffix in (f" ({country})", f", {subdivision}, {country}", f" ({subdivision}, {country})"):
        assert _has_missing_identity_qualifier("Solstice Point", "Solstice Point" + suffix)
    assert not _has_missing_identity_qualifier("Solstice Point", "Distinct Point, " + country)


@pytest.mark.parametrize("origin,candidate", [
    ("Solstice Archive", "Solstice Archive"),
    ("ＳＯＬＳＴＩＣＥ Archive", "solstice archive"),
    ("Solstice Archive", "Solstice Archive (complete)"),
    ("Solstice Archive", "Solstice Archive (unknown format)"),
    ("Solstice Archive", "Solstice Archive extended"),
    ("Solstice Archive", "Solstice Archive novel"),
    ("Solstice Archive", "Solstice Archive 123"),
    ("Solstice Archive", "Solstice Archive 20401"),
    ("Solstice Archive", "Solstice Archive2040"),
    ("Solstice Archive", "Solstice Archive II"),
    ("Solstice Archive", "Solstice Archive (1968 novel by Someone)"),
    ("Solstice Archive", "Solstice Archive (unknown format, 1968)"),
    ("Solstice Archive", "Solstice Archive, Fictional Territory"),
    ("Solstice Archive (SA)", "Solstice Archive"),
    ("SA", "Solstice Archive (novel)"),
    ("极光研究所", "Solstice Research Institute"),
    ("Solstice Games 2040", "Summer Games 2040"),
    ("Solstice Archive 1968", "Solstice Archive 1968 (1968)"),
    ("Solstice Archive (novel)", "Solstice Archive (novel) (novel)"),
])
def test_qualifiers_do_not_replace_the_model_for_other_bases_or_unrecognized_suffixes(origin, candidate):
    calls = []

    def judge(prompt):
        calls.append(prompt)
        return {"same_entity": "YES"}

    assert not _has_missing_identity_qualifier(origin, candidate)
    assert _verify_entity_identity(origin, candidate, judge) == "YES"
    assert len(calls) == 1
    pair = json.loads(calls[0].split("====== entity-verification-data ======\n")[1].split("\n====== end-entity-verification-data")[0])
    assert pair == {"origin": origin, "candidate": candidate}


def test_proposal_for_missing_qualifier_keeps_selector_raw_call_but_skips_verifier():
    calls = []

    def judge(prompt):
        calls.append(prompt)
        return {"candidate_id": 0}

    assert _align(["Solstice Archive"], ["Solstice Archive (novel)"], judge, 32, 32000) == {}
    assert len(calls) == 1 and "entity-selection-data" in calls[0]


def test_exact_anchor_is_preserved_without_either_model_stage():
    def judge(prompt):
        pytest.fail("An exact anchor requires no model call")

    assert _align(["Solstice Archive (novel)"], ["Solstice Archive (novel)"], judge, 32, 32000) == {"Solstice Archive (novel)": "Solstice Archive (novel)"}
