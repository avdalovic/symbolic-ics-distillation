from __future__ import annotations

from ics_symbolic_distill.detection.selection_guards import state_dependence_for_delta_equation


def test_constant_next_candidate_is_rejected() -> None:
    result = state_dependence_for_delta_equation(
        target="F_PU7",
        equation="41.705242 - F_PU7",
        feature_names=["F_PU7", "P_J415"],
    )
    assert not result.state_dependent
    assert result.reason == "constant_next_state"


def test_zero_delta_is_state_dependent_next_state() -> None:
    result = state_dependence_for_delta_equation(
        target="F_PU7",
        equation="0.0",
        feature_names=["F_PU7", "P_J415"],
    )
    assert result.state_dependent


def test_non_python_identifier_tags_are_supported() -> None:
    result = state_dependence_for_delta_equation(
        target="2_FIT_003_PV",
        equation="7.0 - 2_FIT_003_PV",
        feature_names=["2_FIT_003_PV", "2_PIT_002_PV"],
    )
    assert not result.state_dependent


def test_process_coupling_is_kept() -> None:
    result = state_dependence_for_delta_equation(
        target="LIT101",
        equation="0.192 * (FIT101 - FIT201)",
        feature_names=["LIT101", "FIT101", "FIT201"],
    )
    assert result.state_dependent
