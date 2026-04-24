"""Unit tests for BAEE simulation logic."""

import pytest

from tinker_cookbook.recipes.reasoning_theater.baee import (
    BAEEConfig,
    BAEEReport,
    BAEEResult,
    _regrade_efa_answer,
    simulate_efa_oracle,
    simulate_psc_triggered,
)


def _make_problem(
    idx: int,
    ground_truth: str,
    n_correct: int,
    rollout_correct: bool,
    rollout_len: int,
    efa_results: list[tuple[float, str | None, bool]],
    psc_rates: list[float] | None = None,
) -> dict:
    """Helper to create a mock problem result dict."""
    prefix_results = []
    for i, (frac, answer, correct) in enumerate(efa_results):
        psc_rate = psc_rates[i] if psc_rates and i < len(psc_rates) else 0.5
        prefix_results.append({
            "fraction": frac,
            "prefix_len": int(frac * rollout_len),
            "total_len": rollout_len,
            "efa_answer": answer,
            "efa_correct": correct,
            "atlt_logprob": -2.0,
            "psc_n_correct": int(psc_rate * 16),
            "psc_n_total": 16,
            "psc_agreement_rate": psc_rate,
        })
    return {
        "problem_idx": idx,
        "problem": f"Problem {idx}",
        "level": 3,
        "subject": "Algebra",
        "ground_truth": ground_truth,
        "n_correct_rollouts": n_correct,
        "n_total_rollouts": 4,
        "selected_rollout_len": rollout_len,
        "selected_rollout_correct": rollout_correct,
        "commitment_fraction": None,
        "theater_fraction": None,
        "prefix_results": prefix_results,
    }


# ---------------------------------------------------------------------------
# Re-grading tests
# ---------------------------------------------------------------------------


class TestRegradeEfaAnswer:
    def test_clean_answer(self):
        ans, correct = _regrade_efa_answer("42", "42")
        assert ans == "42"
        assert correct is True

    def test_trailing_brace_dot(self):
        ans, correct = _regrade_efa_answer("42}.", "42")
        assert ans == "42"
        assert correct is True

    def test_trailing_brace(self):
        ans, correct = _regrade_efa_answer("42}", "42")
        assert ans == "42"
        assert correct is True

    def test_none_answer(self):
        ans, correct = _regrade_efa_answer(None, "42")
        assert ans is None
        assert correct is False

    def test_empty_after_strip(self):
        ans, correct = _regrade_efa_answer("}.", "42")
        assert ans is None
        assert correct is False

    def test_wrong_answer(self):
        ans, correct = _regrade_efa_answer("99", "42")
        assert ans == "99"
        assert correct is False


# ---------------------------------------------------------------------------
# EFA Oracle tests
# ---------------------------------------------------------------------------


class TestEFAOracle:
    def test_early_exit_at_first_checkpoint(self):
        results = [_make_problem(
            idx=0, ground_truth="42", n_correct=4, rollout_correct=True, rollout_len=1000,
            efa_results=[
                (0.10, "42", True), (0.25, "42", True),
                (0.50, "42", True), (0.75, "42", True), (0.90, "42", True),
            ],
        )]
        report = simulate_efa_oracle(results)
        assert report.n_problems == 1
        assert report.n_early_exit == 1
        assert report.per_problem[0].exit_fraction == 0.10
        assert report.per_problem[0].savings_fraction == pytest.approx(0.90, abs=0.01)

    def test_no_early_exit(self):
        results = [_make_problem(
            idx=0, ground_truth="42", n_correct=4, rollout_correct=True, rollout_len=1000,
            efa_results=[
                (0.10, "wrong", False), (0.25, "wrong", False),
                (0.50, "wrong", False), (0.75, "wrong", False), (0.90, "wrong", False),
            ],
        )]
        report = simulate_efa_oracle(results)
        assert report.n_early_exit == 0
        assert report.per_problem[0].savings_fraction == 0.0

    def test_unsolvable_skipped(self):
        results = [_make_problem(
            idx=0, ground_truth="42", n_correct=0, rollout_correct=False, rollout_len=1000,
            efa_results=[
                (0.10, None, False), (0.25, None, False),
                (0.50, None, False), (0.75, None, False), (0.90, None, False),
            ],
        )]
        report = simulate_efa_oracle(results)
        assert report.n_problems == 0

    def test_regrade_fixes_trailing(self):
        results = [_make_problem(
            idx=0, ground_truth="42", n_correct=4, rollout_correct=True, rollout_len=1000,
            efa_results=[
                (0.10, "42}.", False),  # Originally wrong due to stripping bug
                (0.25, "42", True),
                (0.50, "42", True), (0.75, "42", True), (0.90, "42", True),
            ],
        )]
        report = simulate_efa_oracle(results)
        assert report.n_early_exit == 1
        assert report.per_problem[0].exit_fraction == 0.10


# ---------------------------------------------------------------------------
# PSC-triggered tests
# ---------------------------------------------------------------------------


class TestPSCTriggered:
    def test_exit_at_high_psc(self):
        """Exit when PSC agreement >= 75%."""
        results = [_make_problem(
            idx=0, ground_truth="42", n_correct=4, rollout_correct=True, rollout_len=1000,
            efa_results=[
                (0.10, "42", True), (0.25, "42", True),
                (0.50, "42", True), (0.75, "42", True), (0.90, "42", True),
            ],
            psc_rates=[0.50, 0.75, 0.90, 1.0, 1.0],  # first >= 0.75 at 25%
        )]
        report = simulate_psc_triggered(results, threshold=0.75)
        assert report.n_problems == 1
        assert report.n_early_exit == 1
        assert report.per_problem[0].exit_fraction == 0.25
        assert report.per_problem[0].savings_fraction == pytest.approx(0.75, abs=0.01)
        assert report.per_problem[0].is_correct is True

    def test_no_exit_low_psc(self):
        """No exit when PSC never reaches threshold."""
        results = [_make_problem(
            idx=0, ground_truth="42", n_correct=4, rollout_correct=True, rollout_len=1000,
            efa_results=[
                (0.10, "42", True), (0.25, "42", True),
                (0.50, "42", True), (0.75, "42", True), (0.90, "42", True),
            ],
            psc_rates=[0.30, 0.40, 0.50, 0.60, 0.70],  # never >= 0.75
        )]
        report = simulate_psc_triggered(results, threshold=0.75)
        assert report.n_early_exit == 0

    def test_wrong_problem_not_triggered(self):
        """PSC on unsolvable problems should be low (simulated here)."""
        results = [_make_problem(
            idx=0, ground_truth="42", n_correct=0, rollout_correct=False, rollout_len=1000,
            efa_results=[
                (0.10, "wrong", False), (0.25, "wrong", False),
                (0.50, "wrong", False), (0.75, "wrong", False), (0.90, "wrong", False),
            ],
            psc_rates=[0.05, 0.10, 0.05, 0.00, 0.00],  # low PSC on wrong
        )]
        report = simulate_psc_triggered(results, threshold=0.75)
        assert report.n_early_exit == 0
        assert report.per_problem[0].is_correct is False

    def test_includes_all_problems(self):
        """PSC-triggered includes unsolvable problems (unlike EFA oracle)."""
        results = [
            _make_problem(
                idx=0, ground_truth="42", n_correct=4, rollout_correct=True, rollout_len=1000,
                efa_results=[(0.10, "42", True)] * 5,
                psc_rates=[0.90] * 5,
            ),
            _make_problem(
                idx=1, ground_truth="7", n_correct=0, rollout_correct=False, rollout_len=500,
                efa_results=[(0.10, "wrong", False)] * 5,
                psc_rates=[0.05] * 5,
            ),
        ]
        report = simulate_psc_triggered(results, threshold=0.75)
        assert report.n_problems == 2  # includes unsolvable
        assert report.n_correct == 1
        assert report.n_early_exit == 1

    def test_preserves_accuracy(self):
        """PSC-triggered should not reduce accuracy."""
        results = [
            # Solvable, PSC triggers
            _make_problem(
                idx=0, ground_truth="42", n_correct=4, rollout_correct=True, rollout_len=1000,
                efa_results=[(0.10, "42", True)] * 5,
                psc_rates=[0.90, 0.95, 1.0, 1.0, 1.0],
            ),
            # Solvable, PSC doesn't trigger — falls back to full CoT
            _make_problem(
                idx=1, ground_truth="7", n_correct=2, rollout_correct=True, rollout_len=500,
                efa_results=[(0.10, "wrong", False)] * 5,
                psc_rates=[0.30, 0.40, 0.50, 0.60, 0.70],
            ),
            # Unsolvable, PSC low
            _make_problem(
                idx=2, ground_truth="99", n_correct=0, rollout_correct=False, rollout_len=800,
                efa_results=[(0.10, None, False)] * 5,
                psc_rates=[0.05] * 5,
            ),
        ]
        report = simulate_psc_triggered(results, threshold=0.75)
        # 2 correct (first solvable + second solvable via full CoT), same as full CoT
        assert report.n_correct == 2
        assert report.accuracy == pytest.approx(2 / 3, abs=0.01)


class TestEmpty:
    def test_empty_efa_oracle(self):
        report = simulate_efa_oracle([])
        assert report.n_problems == 0

    def test_empty_psc(self):
        report = simulate_psc_triggered([])
        assert report.n_problems == 0
