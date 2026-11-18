"""Tests for QuantumMemory-Lab: matrices, channels, echo, code, gates."""
import math

import pytest

import main as m


# ---- matrix helpers + density matrices
def test_dm_trace_and_purity():
    rho = m.dm_from_ket(m.KET_PLUS)
    assert abs(m.mat_trace(rho) - 1) < 1e-12
    assert abs(m.purity(rho) - 1.0) < 1e-12


def test_dm_eigenvalues_pure():
    lo, hi = m.eigenvalues_2x2(m.dm_from_ket(m.KET_TILT))
    assert abs(lo) < 1e-12 and abs(hi - 1.0) < 1e-12


def test_fidelity_self_and_orthogonal():
    for psi in m.STORED_STATES.values():
        assert abs(m.fidelity_pure(m.dm_from_ket(psi), psi) - 1.0) < 1e-12
    assert abs(m.fidelity_pure(m.dm_from_ket(m.KET_0), m.KET_1)) < 1e-12


def test_kron_grows_dimensions():
    rho = m.dm_from_ket(m.KET_0)
    assert len(m.kron(rho, rho)) == 4
    assert len(m.kron(m.kron(rho, rho), rho)) == 8


# ---- Kraus channels
def test_kraus_completeness():
    for ops in (m.amplitude_damping_kraus(37.0), m.phase_damping_kraus(37.0)):
        total = m.mat_scale(ops[0], 0.0)
        for op in ops:
            total = m.mat_add(total, m.mat_mul(m.mat_dagger(op), op))
        for i in range(2):
            for j in range(2):
                assert abs(total[i][j] - (1.0 if i == j else 0.0)) < 1e-12


def test_zero_time_is_identity():
    rho = m.stored_state_after(m.KET_TILT, 0.0)
    assert m.fidelity_pure(rho, m.KET_TILT) > 1 - 1e-12


def test_amplitude_damping_excited_anchor():
    for t in (10.0, 50.0, 100.0):
        rho = m.apply_channel(m.dm_from_ket(m.KET_1),
                              m.amplitude_damping_kraus(t))
        assert abs(m.fidelity_pure(rho, m.KET_1)
                   - math.exp(-t / m.T1_US)) < 1e-9


def test_amplitude_damping_spares_ground():
    rho = m.apply_channel(m.dm_from_ket(m.KET_0),
                          m.amplitude_damping_kraus(500.0))
    assert abs(m.fidelity_pure(rho, m.KET_0) - 1.0) < 1e-12


def test_phase_damping_plus_law():
    for t in (0.0, 20.0, 40.0, 80.0, 160.0):
        rho = m.apply_channel(m.dm_from_ket(m.KET_PLUS),
                              m.phase_damping_kraus(t))
        expected = (1.0 + math.exp(-t / m.T2_US)) / 2.0
        assert abs(m.fidelity_pure(rho, m.KET_PLUS) - expected) < 1e-9


def test_phase_damping_keeps_populations():
    before = m.dm_from_ket(m.KET_TILT)
    after = m.apply_channel(before, m.phase_damping_kraus(45.0))
    assert abs(after[0][0] - before[0][0]) < 1e-12
    assert abs(after[1][1] - before[1][1]) < 1e-12


def test_channel_output_is_physical():
    rho = m.stored_state_after(m.KET_TILT, 63.0)
    assert abs(m.mat_trace(rho) - 1) < 1e-12
    lo, _ = m.eigenvalues_2x2(rho)
    assert lo > -1e-12


def test_combined_decay_monotone():
    times = (0.0, 10.0, 20.0, 40.0, 80.0, 160.0)
    fids = [m.fidelity_pure(m.stored_state_after(m.KET_TILT, t), m.KET_TILT)
            for t in times]
    assert all(a > b for a, b in zip(fids, fids[1:]))


def test_combined_relaxes_to_ground_overlap():
    f = m.fidelity_pure(m.stored_state_after(m.KET_TILT, 320.0), m.KET_TILT)
    assert abs(f - math.cos(m.TILT_THETA / 2) ** 2) < 0.01


def test_t2_within_bound():
    assert 0 < m.T2_US <= 2 * m.T1_US


# ---- spin echo
def test_fid_and_echo_start_at_one():
    assert abs(m.fid_coherence(0.0) - 1.0) < 1e-9
    assert abs(m.echo_coherence(0.0) - 1.0) < 1e-9


def test_echo_rescues_coherence():
    assert m.fid_coherence(40.0) < 0.05
    assert m.echo_coherence(40.0) > 0.95
    assert m.echo_coherence(40.0) > 10 * m.fid_coherence(40.0)


def test_t2_star_short():
    assert 0 < m.t2_star_us() < m.T2_US / 4


def test_fid_gaussian_form():
    expected = math.exp(-(m.SIGMA_DELTA * 10.0) ** 2 / 2.0)
    assert abs(m.fid_coherence(10.0) - expected) < 0.01


# ---- repetition code
def test_patterns_are_eight():
    assert len(m.error_patterns()) == 8


@pytest.mark.parametrize("p", [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.49, 0.5, 0.7, 1.0])
def test_enumeration_matches_polynomial(p):
    poly = m.logical_error_polynomial(p)
    assert abs(m.logical_error_enumerated(p, "bit") - poly) < 1e-12
    assert abs(m.logical_error_enumerated(p, "phase") - poly) < 1e-12


def test_dm_crosscheck():
    for p in (0.1, 0.23):
        assert abs(m.logical_dm_check(p)
                   - m.logical_error_polynomial(p)) < 1e-9


def test_code_wins_only_below_half():
    for p in (0.01, 0.1, 0.25, 0.4, 0.49):
        assert m.logical_error_polynomial(p) < p
    assert abs(m.logical_error_polynomial(0.5) - 0.5) < 1e-12
    for p in (0.51, 0.75, 0.9):
        assert m.logical_error_polynomial(p) > p


def test_logical_fidelity_bounds():
    f = m.logical_fidelity(m.KET_TILT, 0.1)
    assert 0.9 < f < 1.0
    assert abs(m.logical_fidelity(m.KET_TILT, 0.0) - 1.0) < 1e-12


# ---- lifetime + benchmark
def test_lifetime_extension():
    bare = m.time_to_fidelity(0.9, coded=False)
    coded = m.time_to_fidelity(0.9, coded=True)
    assert coded > 2.0 * bare
    assert abs(bare - (-m.TAU_P_US * math.log(0.8))) < 0.01


def test_benchmark_table():
    bench = m.benchmark_table()
    rows = bench["rows"]
    assert rows[0]["bare_fidelity"] == 1.0
    assert all(a["bare_fidelity"] > b["bare_fidelity"]
               for a, b in zip(rows, rows[1:]))
    assert all(r["echo_fidelity"] > r["bare_fidelity"] for r in rows[1:])
    assert rows[1]["coded_fidelity"] > rows[1]["bare_fidelity"]
    assert rows[2]["coded_fidelity"] > rows[2]["bare_fidelity"]


# ---- classifier
def test_classifier_scenarios():
    rep = m.classifier_report()
    assert rep["fresh_store"]["label"] == "healthy"
    assert rep["coded_at_5us"]["label"] == "healthy"
    assert rep["bare_at_20us"]["label"] == "degraded"
    assert rep["coded_at_40us"]["label"] == "degraded"
    assert rep["bare_at_T2"]["label"] == "lost"
    assert rep["bare_at_200us"]["label"] == "lost"


def test_classifier_matches_model():
    for entry in m.classifier_report().values():
        feats = tuple(float(x) for x in entry["features"])
        assert m.classify_memory(*feats) == m.model_label(*feats)


# ---- gates + seats
def test_hold_gates_14():
    assert len(m.HOLD_GATES) == 14
    assert all(v.startswith("HOLD") for v in m.HOLD_GATES.values())


def test_gate_outcomes():
    assert m.gate("hardware_memory_claim") == "HOLD"
    assert m.gate("quantum_repeater_deployment") == "HOLD"
    assert m.gate("fault_tolerant_computation_claim") == "HOLD"
    assert m.gate("run_simulation") == "PROCEED"


def test_seats_cannot_override():
    brains = m.brains_report()
    assert brains["decider"]["can_override_hold"] is False
    assert brains["decider"]["seat"] == "Gemini Argon 4 Extended Thinking"
    assert brains["advisor"]["can_override_hold"] is False
    assert brains["flow"]["rendered"] is False
    assert brains["flow"]["live"] == 0


def test_checks_all_pass():
    checks = m.run_checks()
    assert all(ok for _, ok in checks), [n for n, ok in checks if not ok]
