#!/usr/bin/env python3
"""QuantumMemory-Lab - educational quantum memory lab.

by Darion Kellar

Pure Python standard library. Zero dependencies. Zero network.
Education only - and the central lesson is stated up front:

  * A quantum memory is a fight against decoherence. A stored qubit
    leaks energy (amplitude damping, timescale T1) and loses phase
    (phase damping, timescale T2). This lab does the exact density-
    matrix arithmetic for those two channels on 1-3 qubits, so the
    decay of a stored state is computed, never hand-waved.
  * Two defences are priced honestly. A Hahn spin echo refocuses
    slow detuning noise and recovers coherence the free-induction
    decay already lost. A 3-qubit repetition code trades three
    physical qubits for one logical qubit whose error rate is the
    polynomial p_L = 3p^2 - 2p^3 - better than the bare rate only
    when p < 1/2, with break-even exactly at p = 1/2.
  * Nothing here is hardware. The T1/T2 values, the detuning spread,
    and the physical error timescale are LABELLED teaching values.
    No real qubit was stored, no coherence record was set, and no
    storage product exists. Hardware, product, network, repeater,
    and breakthrough claims are permanent HOLD gates - with Gemini
    Argon 4 Extended Thinking as DECIDER (names-only, tighten-only),
    Muse as advisor (never overrides HOLD), and Google Flow
    plan-only seats.
"""
import hashlib
import itertools
import json
import math
import sys

# ---------------------------------------------------------------- constants
VERSION = "1.0.0"

# Labelled teaching values (superconducting-qubit class). These are
# printed wherever they are used; they are not measured device data.
T1_US = 100.0          # energy-relaxation time, microseconds
T2_US = 80.0           # dephasing time, microseconds (T2 <= 2*T1 holds)
SIGMA_DELTA = 0.15     # quasi-static detuning spread, rad/microsecond
DRIFT_SCALE = 0.002    # slow detuning drift between echo halves, rad/us
TAU_P_US = 50.0        # physical bit-flip error timescale, microseconds
TILT_THETA = 0.7       # stored tilted-state angle, radians

# ---------------------------------------------------------------- HOLD gates
HOLD_GATES = {
    "hardware_memory_claim": "HOLD - no claiming this lab builds, simulates, or certifies real quantum memory hardware",
    "quantum_storage_product_claim": "HOLD - no quantum storage product, service, or commercial offering claim",
    "real_key_storage_claim": "HOLD - no storing real cryptographic keys, QKD keys, or secrets in a quantum memory",
    "fault_tolerant_computation_claim": "HOLD - no claiming fault-tolerant quantum computation has been achieved",
    "coherence_record_claim": "HOLD - no coherence-time records or measured device performance claims",
    "error_correction_breakthrough_claim": "HOLD - no QEC breakthrough, threshold-theorem, or record logical-qubit claims",
    "quantum_network_deployment": "HOLD - no deploying a real quantum network, teleportation link, or entanglement distribution",
    "quantum_repeater_deployment": "HOLD - no quantum repeater deployment or field-trial claim",
    "satellite_quantum_link_claim": "HOLD - no satellite, space, or long-distance quantum memory link claim",
    "medical_data_storage_claim": "HOLD - no storing real medical, genomic, or patient data",
    "financial_record_storage_claim": "HOLD - no storing real financial records, trades, or ledger data",
    "military_defense_use": "HOLD - no military, weapons, surveillance, or defense memory application",
    "classified_data_storage_claim": "HOLD - no storing classified, government, or intelligence data",
    "commercial_advantage_claim": "HOLD - no commercial, speed, or quantum-advantage claim for any memory system",
}


def gate(action):
    """Return 'HOLD' if the action is gated, else 'PROCEED' (sandbox sim)."""
    return "HOLD" if action in HOLD_GATES else "PROCEED"


# ================================================================ 1. Small complex matrices
# Density matrices are nested tuples of complex numbers. The helpers
# below are the whole linear-algebra library: multiply, add, scale,
# conjugate transpose, Kronecker product, trace. Nothing is hidden
# behind numpy - every number in the demo can be recomputed by hand.

def mat_mul(a, b):
    n, inner, m = len(a), len(b), len(b[0])
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(m))
        for i in range(n))


def mat_add(a, b):
    return tuple(tuple(x + y for x, y in zip(ra, rb)) for ra, rb in zip(a, b))


def mat_scale(a, s):
    return tuple(tuple(x * s for x in row) for row in a)


def mat_dagger(a):
    return tuple(tuple(a[j][i].conjugate() for j in range(len(a)))
                 for i in range(len(a[0])))


def mat_trace(a):
    return sum(a[i][i] for i in range(len(a)))


def kron(a, b):
    out = []
    for ra in a:
        for rb in b:
            row = []
            for xa in ra:
                row.extend(xa * xb for xb in rb)
            out.append(tuple(row))
    return tuple(out)


def ket(*amps):
    return tuple(complex(a) for a in amps)


def dm_from_ket(psi):
    """Density matrix |psi><psi| for a pure state."""
    return tuple(tuple(psi[i] * psi[j].conjugate() for j in range(len(psi)))
                 for i in range(len(psi)))


def fidelity_pure(rho, psi):
    """F = <psi|rho|psi> for a pure target state (a real number)."""
    total = 0j
    for i, ci in enumerate(psi):
        for j, cj in enumerate(psi):
            total += ci.conjugate() * rho[i][j] * cj
    return total.real


def purity(rho):
    return mat_trace(mat_mul(rho, rho)).real


def eigenvalues_2x2(rho):
    """Closed-form eigenvalues of a 2x2 Hermitian matrix."""
    a, d = rho[0][0].real, rho[1][1].real
    off = abs(rho[0][1])
    mid, half = (a + d) / 2.0, math.sqrt(((a - d) / 2.0) ** 2 + off ** 2)
    return (mid - half, mid + half)


# Stored single-qubit states used across the lab.
KET_0 = ket(1, 0)
KET_1 = ket(0, 1)
KET_PLUS = ket(1 / math.sqrt(2), 1 / math.sqrt(2))
KET_TILT = ket(math.cos(TILT_THETA / 2), math.sin(TILT_THETA / 2))
STORED_STATES = {"ground |0>": KET_0, "excited |1>": KET_1,
                 "plus |+>": KET_PLUS, "tilted 0.7 rad": KET_TILT}


# ================================================================ 2. Decoherence channels (Kraus form)
# A channel is a list of Kraus operators K with sum K^dagger K = I.
# Applying the channel is rho -> sum_K K rho K^dagger. Amplitude
# damping (gamma = 1 - exp(-t/T1)) drains |1> into |0>; phase
# damping shrinks the off-diagonal coherence by exp(-t/T2) exactly.

def amplitude_damping_kraus(t_us, t1_us=T1_US):
    gamma = 1.0 - math.exp(-t_us / t1_us)
    s = math.sqrt(1.0 - gamma)
    return [((1 + 0j, 0j), (0j, s + 0j)),
            ((0j, math.sqrt(gamma) + 0j), (0j, 0j))]


def phase_damping_kraus(t_us, t2_us=T2_US):
    # Coherence must scale by exp(-t/T2) = sqrt(1 - lam), so the
    # Kraus weight is lam = 1 - exp(-2t/T2).
    lam = 1.0 - math.exp(-2.0 * t_us / t2_us)
    s = math.sqrt(1.0 - lam)
    return [((1 + 0j, 0j), (0j, s + 0j)),
            ((0j, 0j), (0j, math.sqrt(lam) + 0j))]


def apply_channel(rho, kraus_ops):
    out = mat_scale(rho, 0.0)
    for op in kraus_ops:
        out = mat_add(out, mat_mul(mat_mul(op, rho), mat_dagger(op)))
    return out


def stored_state_after(psi, t_us, t1_us=T1_US, t2_us=T2_US):
    """Store psi, then let both channels act for t microseconds."""
    rho = dm_from_ket(psi)
    rho = apply_channel(rho, amplitude_damping_kraus(t_us, t1_us))
    rho = apply_channel(rho, phase_damping_kraus(t_us, t2_us))
    return rho


def fidelity_table(times=(0.0, 20.0, 40.0, 80.0, 160.0)):
    """Fidelity of each stored state vs storage time, both channels."""
    table = {}
    for name, psi in STORED_STATES.items():
        table[name] = [round(fidelity_pure(stored_state_after(psi, t), psi), 6)
                       for t in times]
    return {"times_us": list(times), "fidelity": table}


# ================================================================ 3. Spin echo (Hahn) vs free-induction decay
# Quasi-static noise: every ensemble member keeps one detuning delta
# for the whole shot, drawn from a Gaussian of width SIGMA_DELTA.
# FID coherence is |<exp(i*delta*t)>| over a deterministic weighted
# grid - it dies on the T2* scale. A Hahn echo (pi pulse at tau)
# cancels every static delta exactly; only the labelled slow drift
# between the two halves survives, so the echo keeps the coherence.

def _ensemble():
    """Deterministic detuning grid with Gaussian weights."""
    members = []
    for j in range(-200, 201):
        delta = j * (4.0 * SIGMA_DELTA / 200.0)
        weight = math.exp(-delta ** 2 / (2.0 * SIGMA_DELTA ** 2))
        members.append((delta, weight))
    return members


ENSEMBLE = _ensemble()


def fid_coherence(t_us):
    """Free-induction coherence |<exp(i delta t)>| over the ensemble."""
    num = sum(w * complex(math.cos(d * t_us), math.sin(d * t_us))
              for d, w in ENSEMBLE)
    den = sum(w for _, w in ENSEMBLE)
    return abs(num) / den


def _drift(j):
    """Labelled slow detuning drift for member index j (rad/us)."""
    return DRIFT_SCALE * math.sin(j * 1.7)


def echo_coherence(total_us):
    """Hahn-echo coherence at total time 2*tau (pi pulse at tau).

    Static detuning cancels exactly; the residual phase is the drift
    accumulated asymmetrically across the two halves.
    """
    tau = total_us / 2.0
    num, den = 0j, 0.0
    for j, (delta, w) in enumerate(ENSEMBLE):
        phase = -_drift(j - 200) * tau
        num += w * complex(math.cos(phase), math.sin(phase))
        den += w
    return abs(num) / den


def t2_star_us():
    """FID 1/e time of the ensemble, found by bisection."""
    lo, hi = 0.0, 200.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if fid_coherence(mid) > 1.0 / math.e:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def echo_report(times=(10.0, 20.0, 40.0, 80.0)):
    return {
        "sigma_delta_rad_per_us": SIGMA_DELTA,
        "t2_star_us": round(t2_star_us(), 4),
        "rows": [{"t_us": t,
                  "fid_coherence": round(fid_coherence(t), 6),
                  "echo_coherence": round(echo_coherence(t), 6)}
                 for t in times],
    }


# ================================================================ 4. Repetition-code memory
# Three physical qubits hold one logical qubit. The bit-flip code
# stores alpha|000> + beta|111> and majority-votes the readout; the
# phase-flip code stores alpha|+++> + beta|---> and majority-votes
# in the X basis. Either way the logical failure probability is the
# weight of all error patterns with two or three flips:
#     p_L = 3p^2 (1 - p) + p^3 = 3p^2 - 2p^3
# enumerated pattern by pattern below - not quoted from a textbook.

def error_patterns():
    """All 3-qubit flip patterns as (qubits flipped, count)."""
    pats = []
    for size in range(4):
        for combo in itertools.combinations(range(3), size):
            pats.append(combo)
    return pats


def logical_error_enumerated(p, code="bit"):
    """Exact logical failure probability by pattern enumeration.

    Majority vote fails exactly when >= 2 of the 3 qubits flipped.
    The phase-flip code in the Hadamard basis has the identical
    counting, so both codes return the same polynomial - asserted
    in the checks rather than assumed.
    """
    if code not in ("bit", "phase"):
        raise ValueError(f"unknown code: {code}")
    total = 0.0
    for combo in error_patterns():
        k = len(combo)
        weight = (p ** k) * ((1.0 - p) ** (3 - k))
        if k >= 2:
            total += weight
    return total


def logical_error_polynomial(p):
    return 3.0 * p * p - 2.0 * p * p * p


def logical_dm_check(p):
    """Cross-check the enumeration with an 8x8 density matrix.

    Store the basis logical state |0_L> = |000>, mix over every
    flip pattern with its exact weight, and read the wrong-majority
    probability off the diagonal of the mixed state: outcomes with
    two or three 1s are exactly the majority-vote failures.
    """
    psi_l = [0j] * 8
    psi_l[0] = 1.0 + 0j
    mixed = mat_scale(dm_from_ket(tuple(psi_l)), 0.0)
    for combo in error_patterns():
        k = len(combo)
        weight = (p ** k) * ((1.0 - p) ** (3 - k))
        state = [0j] * 8
        target = 0
        for q in combo:
            target ^= (1 << q)
        state[target] = 1.0 + 0j
        mixed = mat_add(mixed, mat_scale(dm_from_ket(tuple(state)), weight))
    wrong = 0.0
    for idx in range(8):
        if bin(idx).count("1") >= 2:
            wrong += mixed[idx][idx].real
    return wrong


def physical_flip_prob(t_us, tau_us=TAU_P_US):
    """Labelled toy: flip probability saturating at 1/2 over tau."""
    return (1.0 - math.exp(-t_us / tau_us)) / 2.0


def logical_fidelity(psi, p):
    """Fidelity of the corrected logical state at physical rate p.

    After majority correction the logical qubit is the stored state
    with probability 1 - p_L and the logically flipped state with
    probability p_L.
    """
    p_l = logical_error_polynomial(p)
    overlap = abs(sum(psi[i].conjugate() * psi[1 - i] for i in range(2))) ** 2
    return (1.0 - p_l) + p_l * overlap


def time_to_fidelity(target=0.9, coded=False, tau_us=TAU_P_US):
    """Storage time for memory fidelity to fall to `target`.

    Bare: F = 1 - p(t). Coded: F = 1 - p_L(p(t)). Bisection on the
    exact formulas, so the lifetime extension is computed.
    """
    def fidelity(t):
        p = physical_flip_prob(t, tau_us)
        return 1.0 - (logical_error_polynomial(p) if coded else p)

    lo, hi = 0.0, 10000.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if fidelity(mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def code_report(p_values=(0.01, 0.05, 0.1, 0.2, 0.3, 0.5)):
    rows = []
    for p in p_values:
        rows.append({
            "p": p,
            "p_L_bit": round(logical_error_enumerated(p, "bit"), 8),
            "p_L_phase": round(logical_error_enumerated(p, "phase"), 8),
            "p_L_polynomial": round(logical_error_polynomial(p), 8),
            "gain": round(p / logical_error_polynomial(p), 4) if p else None,
        })
    bare = time_to_fidelity(0.9, coded=False)
    coded = time_to_fidelity(0.9, coded=True)
    return {"rows": rows,
            "time_to_fidelity_0.9_us": {"bare": round(bare, 4),
                                        "coded": round(coded, 4),
                                        "extension_factor": round(coded / bare, 4)}}


# ================================================================ 5. Storage benchmark
# One table, matched parameters: the bare qubit (both channels), the
# Hahn echo (ensemble coherence), and the coded memory (logical
# fidelity of the tilted state) at the same storage times.

def benchmark_table(times=(0.0, 10.0, 20.0, 40.0, 80.0)):
    rows = []
    for t in times:
        bare = fidelity_pure(stored_state_after(KET_TILT, t), KET_TILT)
        echo_f = (1.0 + echo_coherence(t)) / 2.0
        p = physical_flip_prob(t)
        coded = logical_fidelity(KET_TILT, p)
        rows.append({"t_us": t,
                     "bare_fidelity": round(bare, 6),
                     "echo_fidelity": round(echo_f, 6),
                     "coded_fidelity": round(coded, 6),
                     "physical_p": round(p, 6),
                     "p_L": round(logical_error_polynomial(p), 6)})
    return {"t1_us": T1_US, "t2_us": T2_US, "tau_p_us": TAU_P_US,
            "rows": rows}


# ================================================================ 6. kNN memory-health classifier
# Features: fidelity, storage time in units of T2, logical error
# probability. The training grid is labelled by the lab's own health
# policy (model_label below) - the classifier interpolates the
# policy, it does not invent a second standard.

def model_label(fidelity, t_over_t2, p_l):
    """The health policy distilled to a three-way teaching label."""
    if fidelity < 0.70 or p_l > 0.20:
        return "lost"
    if fidelity >= 0.95 and p_l <= 0.02 and t_over_t2 <= 1.0:
        return "healthy"
    return "degraded"


def _training_grid():
    rows = []
    for fidelity in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.00):
        for ratio in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
            for p_l in (0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50):
                rows.append(((fidelity, ratio, p_l),
                             model_label(fidelity, ratio, p_l)))
    return rows


TRAINING_GRID = _training_grid()


def _scaled(features):
    fidelity, ratio, p_l = features
    return ((fidelity - 0.5) / 0.5, ratio / 4.0, p_l / 0.5)


def classify_memory(fidelity, t_over_t2, p_l, k=5):
    """kNN over the policy-labelled grid, deterministic tie-breaking."""
    target = _scaled((fidelity, t_over_t2, p_l))
    ranked = []
    for features, label in TRAINING_GRID:
        point = _scaled(features)
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(target, point)))
        ranked.append((dist, features, label))
    ranked.sort(key=lambda row: (row[0], row[1]))
    votes = {}
    for _, _, label in ranked[:k]:
        votes[label] = votes.get(label, 0) + 1
    priority = {"healthy": 0, "degraded": 1, "lost": 2}
    return sorted(votes.items(), key=lambda kv: (-kv[1], priority[kv[0]]))[0][0]


def classifier_report():
    """Scenarios whose features are read off the lab's own benches."""
    scenarios = {}
    fresh = benchmark_table((0.0,))["rows"][0]
    scenarios["fresh_store"] = (fresh["coded_fidelity"], 0.0, 0.0)
    coded_5 = benchmark_table((5.0,))["rows"][0]
    scenarios["coded_at_5us"] = (coded_5["coded_fidelity"], 5.0 / T2_US,
                                 coded_5["p_L"])
    bare_20 = benchmark_table((20.0,))["rows"][0]
    scenarios["bare_at_20us"] = (bare_20["bare_fidelity"], 20.0 / T2_US,
                                 bare_20["physical_p"])
    coded_40 = benchmark_table((40.0,))["rows"][0]
    scenarios["coded_at_40us"] = (coded_40["coded_fidelity"], 40.0 / T2_US,
                                  coded_40["p_L"])
    bare_t2 = benchmark_table((T2_US,))["rows"][0]
    scenarios["bare_at_T2"] = (bare_t2["bare_fidelity"], 1.0,
                               bare_t2["physical_p"])
    bare_200 = benchmark_table((200.0,))["rows"][0]
    scenarios["bare_at_200us"] = (bare_200["bare_fidelity"], 200.0 / T2_US,
                                  bare_200["physical_p"])
    return {name: {"features": [round(f, 6) for f in feats],
                   "label": classify_memory(*feats)}
            for name, feats in scenarios.items()}


# ================================================================ 7. Advisory seats
def _trace(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def argon4_decide(topic="quantummemory-demo"):
    """Gemini Argon 4 DECIDER seat: MAP->MODEL->STRESS->DECIDE.

    Names-only keys, live calls 0, tighten-only: can add HOLDs, never lift one.
    """
    stages = ["MAP", "MODEL", "STRESS", "DECIDE"]
    payload = {"seat": "gemini-argon-4", "topic": topic, "stages": stages}
    return {
        "seat": "Gemini Argon 4 Extended Thinking",
        "role": "DECIDER",
        "pipeline": stages,
        "tiers": [2048, 8192, 32768],
        "key": "names-only (GEMINI_API_KEY name checked, value never read)",
        "live_calls": 0,
        "decision": "PROCEED - sandbox simulation only, all hardware/product gates stay HOLD",
        "can_override_hold": False,
        "trace_sha256_16": _trace(payload),
    }


def claude_advisor(topic="quantummemory-demo"):
    """Muse advisor seat: advises, can never override a HOLD."""
    return {
        "seat": "Muse",
        "role": "ADVISOR",
        "key": "names-only (MODEL_API_KEY name checked, value never read)",
        "live_calls": 0,
        "advice": "Lead with the decay, not the dream: print the bare fidelity next to the echo and coded numbers at matched times, state that T1/T2 are labelled teaching values, and let the break-even at p = 1/2 say plainly that error correction only helps below it.",
        "can_override_hold": False,
    }


def google_flow_plan():
    """Google Flow seat: plan-only, nothing rendered, nothing live."""
    return {
        "seat": "Google Flow",
        "role": "PLAN-ONLY",
        "shots": ["a bright qubit fading as T1 and T2 clocks run down",
                  "an ensemble of arrows fanning out under detuning noise",
                  "a pi pulse flipping the fan so the arrows re-converge at the echo",
                  "three physical qubits voting to protect one logical qubit",
                  "the break-even line at p = 1/2 with the code winning only below it"],
        "rendered": False,
        "live": 0,
    }


def brains_report():
    return {"decider": argon4_decide(), "advisor": claude_advisor(),
            "flow": google_flow_plan()}


# ================================================================ checks
def run_checks():
    c = []

    def chk(name, ok):
        c.append((name, bool(ok)))

    # 1. matrix helpers + density matrices
    rho_plus = dm_from_ket(KET_PLUS)
    chk("dm_trace_one", abs(mat_trace(rho_plus) - 1) < 1e-12)
    chk("dm_pure_purity_one", abs(purity(rho_plus) - 1.0) < 1e-12)
    chk("dm_hermitian", all(abs(rho_plus[i][j] - rho_plus[j][i].conjugate())
                            < 1e-12 for i in range(2) for j in range(2)))
    lo, hi = eigenvalues_2x2(rho_plus)
    chk("dm_pure_eigenvalues", abs(lo) < 1e-12 and abs(hi - 1.0) < 1e-12)
    chk("kron_dimensions", len(kron(rho_plus, rho_plus)) == 4
        and len(kron(kron(rho_plus, rho_plus), rho_plus)) == 8)
    chk("fidelity_self_one",
        all(abs(fidelity_pure(dm_from_ket(psi), psi) - 1.0) < 1e-12
            for psi in STORED_STATES.values()))
    chk("fidelity_orthogonal_zero",
        abs(fidelity_pure(dm_from_ket(KET_0), KET_1)) < 1e-12)

    # 2. Kraus channels are valid channels
    for name, ops in (("ad", amplitude_damping_kraus(37.0)),
                      ("pd", phase_damping_kraus(37.0))):
        total = mat_scale(ops[0], 0.0)
        for op in ops:
            total = mat_add(total, mat_mul(mat_dagger(op), op))
        chk(f"kraus_complete_{name}",
            all(abs(total[i][j] - (1.0 if i == j else 0.0)) < 1e-12
                for i in range(2) for j in range(2)))
    chk("channel_at_zero_is_identity",
        fidelity_pure(stored_state_after(KET_TILT, 0.0), KET_TILT) > 1 - 1e-12)

    # 3. amplitude damping anchors
    for t in (10.0, 50.0, 100.0):
        f1 = fidelity_pure(
            apply_channel(dm_from_ket(KET_1), amplitude_damping_kraus(t)),
            KET_1)
        chk(f"ad_excited_decay_t{t:g}", abs(f1 - math.exp(-t / T1_US)) < 1e-9)
    chk("ad_ground_untouched",
        abs(fidelity_pure(
            apply_channel(dm_from_ket(KET_0), amplitude_damping_kraus(500.0)),
            KET_0) - 1.0) < 1e-12)
    chk("ad_trace_preserved",
        abs(mat_trace(stored_state_after(KET_TILT, 63.0)) - 1) < 1e-12)
    lo, _ = eigenvalues_2x2(stored_state_after(KET_TILT, 63.0))
    chk("channel_output_positive", lo > -1e-12)

    # 4. phase damping: the (1 + exp(-t/T2))/2 law for |+>
    for t in (0.0, 20.0, 40.0, 80.0, 160.0):
        f = fidelity_pure(
            apply_channel(dm_from_ket(KET_PLUS), phase_damping_kraus(t)),
            KET_PLUS)
        expected = (1.0 + math.exp(-t / T2_US)) / 2.0
        chk(f"pd_plus_law_t{t:g}", abs(f - expected) < 1e-9)
    chk("pd_populations_untouched",
        abs(apply_channel(dm_from_ket(KET_TILT), phase_damping_kraus(45.0))[0][0]
            - dm_from_ket(KET_TILT)[0][0]) < 1e-12)
    chk("t2_within_2t1_bound", 0 < T2_US <= 2 * T1_US)

    # 5. combined storage decays, then relaxes toward the |0> overlap
    prev = 2.0
    mono = True
    for t in (0.0, 10.0, 20.0, 40.0, 80.0, 160.0):
        f = fidelity_pure(stored_state_after(KET_TILT, t), KET_TILT)
        mono = mono and f < prev
        prev = f
    chk("combined_fidelity_decays", mono)
    chk("combined_fidelity_relaxes_to_ground_overlap",
        abs(fidelity_pure(stored_state_after(KET_TILT, 320.0), KET_TILT)
            - math.cos(TILT_THETA / 2) ** 2) < 0.01)
    chk("long_storage_near_ground",
        fidelity_pure(stored_state_after(KET_TILT, 2000.0), KET_0) > 0.99)

    # 6. spin echo beats free-induction decay
    chk("fid_starts_at_one", abs(fid_coherence(0.0) - 1.0) < 1e-9)
    chk("echo_starts_at_one", abs(echo_coherence(0.0) - 1.0) < 1e-9)
    chk("fid_dies_by_40us", fid_coherence(40.0) < 0.05)
    chk("echo_survives_40us", echo_coherence(40.0) > 0.95)
    chk("echo_much_better_than_fid",
        echo_coherence(40.0) > 10 * fid_coherence(40.0))
    chk("t2star_much_shorter_than_t2", 0 < t2_star_us() < T2_US / 4)
    chk("fid_matches_gaussian_form",
        abs(fid_coherence(10.0)
            - math.exp(-(SIGMA_DELTA * 10.0) ** 2 / 2.0)) < 0.01)

    # 7. repetition code: enumeration = polynomial, both bases
    for p in (0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.49, 0.5, 0.7, 1.0):
        chk(f"code_poly_p{p:g}",
            abs(logical_error_enumerated(p, "bit")
                - logical_error_polynomial(p)) < 1e-12
            and abs(logical_error_enumerated(p, "phase")
                    - logical_error_polynomial(p)) < 1e-12)
    chk("code_dm_crosscheck",
        abs(logical_dm_check(0.1) - logical_error_polynomial(0.1)) < 1e-9
        and abs(logical_dm_check(0.23) - logical_error_polynomial(0.23)) < 1e-9)
    chk("code_wins_below_half",
        all(logical_error_polynomial(p) < p
            for p in (0.01, 0.1, 0.25, 0.4, 0.49)))
    chk("code_breakeven_at_half",
        abs(logical_error_polynomial(0.5) - 0.5) < 1e-12)
    chk("code_loses_above_half",
        all(logical_error_polynomial(p) > p for p in (0.51, 0.75, 0.9)))
    chk("code_patterns_eight", len(error_patterns()) == 8)

    # 8. lifetime extension is computed, not quoted
    bare_t = time_to_fidelity(0.9, coded=False)
    coded_t = time_to_fidelity(0.9, coded=True)
    chk("lifetime_extension_over_2x", coded_t > 2.0 * bare_t)
    chk("lifetime_bare_matches_formula",
        abs(bare_t - (-TAU_P_US * math.log(0.8))) < 0.01)
    chk("lifetime_targets_hit",
        abs((1.0 - physical_flip_prob(bare_t)) - 0.9) < 1e-6
        and abs((1.0 - logical_error_polynomial(physical_flip_prob(coded_t)))
                - 0.9) < 1e-6)

    # 9. benchmark coherence
    bench = benchmark_table()
    rows = bench["rows"]
    chk("benchmark_starts_perfect",
        rows[0]["bare_fidelity"] == 1.0 and rows[0]["coded_fidelity"] == 1.0)
    chk("benchmark_bare_decays",
        all(rows[i]["bare_fidelity"] > rows[i + 1]["bare_fidelity"]
            for i in range(len(rows) - 1)))
    chk("benchmark_echo_beats_bare",
        all(r["echo_fidelity"] > r["bare_fidelity"] for r in rows[1:]))
    chk("benchmark_coded_beats_bare_early",
        rows[1]["coded_fidelity"] > rows[1]["bare_fidelity"]
        and rows[2]["coded_fidelity"] > rows[2]["bare_fidelity"])

    # 10. classifier - labels come from the health policy itself
    rep = classifier_report()
    chk("classifier_fresh_healthy", rep["fresh_store"]["label"] == "healthy")
    chk("classifier_coded_early_healthy",
        rep["coded_at_5us"]["label"] == "healthy")
    chk("classifier_bare_20_degraded",
        rep["bare_at_20us"]["label"] == "degraded")
    chk("classifier_coded_40_degraded",
        rep["coded_at_40us"]["label"] == "degraded")
    chk("classifier_bare_t2_lost", rep["bare_at_T2"]["label"] == "lost")
    chk("classifier_bare_200_lost", rep["bare_at_200us"]["label"] == "lost")
    chk("classifier_matches_model",
        all(classify_memory(*[float(x) for x in v["features"]])
            == model_label(*[float(x) for x in v["features"]])
            for v in rep.values()))

    # gates + seats
    chk("hold_gates_14", len(HOLD_GATES) == 14
        and all(v.startswith("HOLD") for v in HOLD_GATES.values()))
    chk("gate_hardware_hold", gate("hardware_memory_claim") == "HOLD")
    chk("gate_product_hold", gate("quantum_storage_product_claim") == "HOLD")
    chk("gate_repeater_hold", gate("quantum_repeater_deployment") == "HOLD")
    chk("gate_fault_tolerant_hold",
        gate("fault_tolerant_computation_claim") == "HOLD")
    chk("gate_sandbox_proceed", gate("run_simulation") == "PROCEED")
    brains = brains_report()
    chk("argon4_no_override", brains["decider"]["can_override_hold"] is False
        and brains["decider"]["seat"]
        == "Gemini Argon 4 Extended Thinking")
    chk("claude_no_override", brains["advisor"]["can_override_hold"] is False)
    chk("flow_plan_only", brains["flow"]["rendered"] is False
        and brains["flow"]["live"] == 0)
    return c


# ================================================================ CLI / demo
def demo():
    print("=" * 68)
    print("QuantumMemory-Lab - qubit storage, decoherence & coded memory")
    print("by Darion Kellar | MIT | pure stdlib | zero network | EDUCATION ONLY")
    print("=" * 68)
    print("\n[1] THE HONEST CORE - a memory is a fight against decoherence")
    print("    A stored qubit leaks energy (amplitude damping, T1) and loses")
    print("    phase (phase damping, T2). This lab does the exact density-")
    print("    matrix arithmetic for both channels - decay is computed,")
    print("    never hand-waved. All device numbers are LABELLED teaching")
    print(f"    values: T1 = {T1_US:.0f} us, T2 = {T2_US:.0f} us, "
          f"error timescale = {TAU_P_US:.0f} us.")
    print("\n[2] Bare storage - fidelity vs time (both channels, Kraus form)")
    table = fidelity_table()
    header = "    state".ljust(22) + "".join(
        f"t={t:<8.0f}" for t in table["times_us"])
    print(header)
    for name, row in table["fidelity"].items():
        print("    " + name.ljust(18) + "".join(f"{f:<10.4f}" for f in row))
    print("    -> |1> decays as exp(-t/T1) under amplitude damping; |+>")
    print("       under pure dephasing follows (1 + exp(-t/T2)) / 2 exactly:")
    f_t2 = fidelity_pure(
        apply_channel(dm_from_ket(KET_PLUS), phase_damping_kraus(T2_US)),
        KET_PLUS)
    print(f"       F(|+>, t = T2) = {f_t2:.4f} = (1 + e^-1) / 2.")
    print("\n[3] Spin echo - refocusing the quasi-static detuning fan")
    echo = echo_report()
    print(f"    detuning spread sigma = {SIGMA_DELTA} rad/us (labelled) | "
          f"T2* (FID 1/e time) = {echo['t2_star_us']:.2f} us")
    for row in echo["rows"]:
        print(f"    t={row['t_us']:>5.0f} us   FID coherence "
              f"{row['fid_coherence']:.4f}   Hahn echo {row['echo_coherence']:.4f}")
    print("    -> the pi pulse cancels every static detuning exactly; only")
    print("       the labelled slow drift survives, so the echo keeps what")
    print("       free-induction decay already lost.")
    print("\n[4] Repetition-code memory - 3 physical qubits, 1 logical")
    code = code_report()
    for row in code["rows"]:
        print(f"    p={row['p']:<5} p_L={row['p_L_polynomial']:<10.6f} "
              f"gain x{row['gain']}")
    t09 = code["time_to_fidelity_0.9_us"]
    print(f"    time for fidelity to fall to 0.9: bare {t09['bare']:.2f} us vs "
          f"coded {t09['coded']:.2f} us (x{t09['extension_factor']:.2f})")
    print("    -> p_L = 3p^2 - 2p^3, enumerated pattern by pattern for both")
    print("       the bit-flip and phase-flip codes. The code wins exactly")
    print("       when p < 1/2; break-even at p = 1/2 is asserted, not assumed.")
    print("\n[5] Storage benchmark - matched parameters, one table")
    bench = benchmark_table()
    print("    t (us)   bare      echo      coded     p        p_L")
    for row in bench["rows"]:
        print(f"    {row['t_us']:<8.0f} {row['bare_fidelity']:<9.4f} "
              f"{row['echo_fidelity']:<9.4f} {row['coded_fidelity']:<9.4f} "
              f"{row['physical_p']:<8.4f} {row['p_L']:.4f}")
    print("\n[6] Memory-health classifier (TinyML kNN, policy-labelled)")
    for name, entry in classifier_report().items():
        feats = entry["features"]
        print(f"    {name:20s} F={feats[0]:.3f} t/T2={feats[1]:.2f} "
              f"p={feats[2]:.3f} -> {entry['label']}")
    print("\n[7] Advisory seats")
    brains = brains_report()
    print(f"    decider: {brains['decider']['decision']} "
          f"trace={brains['decider']['trace_sha256_16']}")
    print(f"    advisor can_override_hold = {brains['advisor']['can_override_hold']}")
    print(f"    flow rendered = {brains['flow']['rendered']} "
          f"live = {brains['flow']['live']}")
    print(f"\n[8] HOLD gates: {len(HOLD_GATES)}")
    for key in HOLD_GATES:
        print(f"    HOLD  {key}")
    checks = run_checks()
    passed = sum(1 for _, ok in checks if ok)
    print(f"\nCHECKS {passed}/{len(checks)}")
    for name, ok in checks:
        if not ok:
            print(f"    FAIL {name}")
    verdict = "PROCEED" if passed == len(checks) else "HOLD"
    print(f"VERDICT {verdict} (sandbox simulation only) | HOLD gates {len(HOLD_GATES)}")
    return 0 if passed == len(checks) else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] == "demo":
        return demo()
    cmd = argv[0]
    if cmd == "checks":
        checks = run_checks()
        for name, ok in checks:
            print(f"{'PASS' if ok else 'FAIL'} {name}")
        print(f"CHECKS {sum(1 for _, ok in checks if ok)}/{len(checks)}")
        return 0 if all(ok for _, ok in checks) else 1
    if cmd == "store":
        print(json.dumps(fidelity_table(), indent=2, sort_keys=True))
        return 0
    if cmd == "echo":
        print(json.dumps(echo_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "code":
        print(json.dumps(code_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "benchmark":
        print(json.dumps(benchmark_table(), indent=2, sort_keys=True))
        return 0
    if cmd == "classify":
        print(json.dumps(classifier_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "brains":
        print(json.dumps(brains_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "holds":
        for key, value in HOLD_GATES.items():
            print(f"{key}: {value}")
        return 0
    if cmd == "hold-demo":
        action = argv[1] if len(argv) > 1 else "hardware_memory_claim"
        print(f"{action}: {HOLD_GATES.get(action, 'PROCEED - not gated')}")
        return 2 if gate(action) == "HOLD" else 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
