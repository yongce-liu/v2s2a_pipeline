"""Smoke test for YAML->Config field filtering (no optimizer run)."""

from __future__ import annotations


def test_filter_config_fields_drops_unknown_keys():
    from physics_opt.config import filter_config_fields

    filtered = filter_config_fields(
        {"sim_dt": 0.005, "not_a_field": 1, "task": "whisking"}
    )
    assert filtered == {"sim_dt": 0.005, "task": "whisking"}
