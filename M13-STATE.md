# M13 — the state-compression ladder: how many dynamical degrees
# of freedom does a neuron's computation actually need?

Origin: collaborator proposal (2026-09-01) — approximate the
*computationally relevant* state of biological units with compact
learned dynamical systems, compose them, allow heterogeneous
timescales. Sharpened here into the program's form: a measured
ladder with a known-answer calibration rung.

The program already owns pieces of this question: WKV = 2 counters,
longhorn = 1 diagonal memory/channel, M10's clock law (17 params of
phase beat every oscillator bank), M12's five-instances law
("conflicting temporal configurations demand independent state
variables; parameter search cannot substitute for state"). M13 asks
the question at the single-unit level, quantitatively.

## Design (rung 1 — calibration on known dimension)

Teacher: standard squid-axon Hodgkin–Huxley (V, m, h, n — true
dynamical dimension 4). Internal integration dt=0.01 ms
(exponential Euler on gates), recorded at 0.1 ms. Drive:
Ornstein-Uhlenbeck current (per-sequence mean 0-8, sigma 0.5-4,
tau 3-50 ms) plus 0-3 random step pulses (-3 to +12, 20-200 ms),
clipped to [-5, 20] uA/cm^2 — spans silent, fluctuation-driven and
mean-driven regimes. 256 train / 32 val / 32 test sequences of
1000 ms.

Surrogates: k-state GRU cell (input = scaled current, hidden = k,
linear readout to normalized voltage), k in {1, 2, 4, 8}. TBPTT
chunks of 1000 steps (100 ms), Adam, spike-region samples
(V > -20 mV) weighted 4x in the MSE. Sub-1k parameters at k=8 —
these are units, not networks.

Metrics:
1. voltage RMSE (mV, test);
2. spike-timing F1, +-2 ms tolerance (spikes = upward 0 mV
   crossings, 2 ms dedup, same detector on teacher and surrogate);
3. behavioral signatures OUT OF DISTRIBUTION of the training
   drive: f-I curve (HH is type II — discontinuous onset near
   ~50 Hz at rheobase ~6.2), and the anodal-break rebound spike
   (release from I=-3 hyperpolarization fires with NO positive
   drive).

## Preregistered predictions (written before any training)

P1. Both fit metrics improve sharply k=1 -> 2 -> 4 and saturate at
    k=4: the ladder should recover the teacher's true dimension
    (k=8 ~ k=4 within noise).
P2. k=1 fails qualitatively, not just quantitatively: no rebound
    spike, distorted f-I onset (type II onset needs the
    subthreshold resonance a single state cannot express). A
    1-state unit is a committer, not a neuron (cf. M12).
P3. k=2 captures most of the spike-timing F1 (the Izhikevich
    claim, learned rather than hand-derived) but underfits
    voltage shape and the rebound amplitude.
P4. The k=4 surrogate at 0.1 ms is a ~10x cheaper-per-step drop-in
    for the 0.01 ms teacher (and needs no gating lookups) — the
    "compact digital approximation" quantified.

## Later rungs (preregistered direction, not yet run)

R2: hand-designed classical arms fitted to the same data by CMA-ES
    (LIF-1, Izhikevich-2, AdEx-2) vs the learned ladder at equal
    state count — does learning beat 50 years of hand design at
    matching a *specific* teacher?
R3: adapting teacher (HH + slow M-current, dimension 5) — does the
    ladder's saturation point track the added slow variable? This
    is the instrument test: detect unknown effective dimension.
R4: composition — a ring/chain of k-state surrogates vs one
    (nk)-state monolith at matched total state: where does
    locality + heterogeneity beat lumping (the collaborator's
    mixture-of-dynamics idea, and M12's five-instances law run
    forward).
R5: task rung — budget-matched sequence tasks (our M9/M10 suites)
    with 1/2/4/8-state units: which states pay per byte on *tasks*
    rather than on imitation.

Honesty notes: rung-1 saturation at k=4 validates the method, not
the hypothesis that richer units help on tasks (that is R5). A
GRU's gates give it timescale flexibility a plain RNN lacks; the
ladder measures state COUNT, with cell class held fixed.

## Run 1 (2026-09-01): GATE FAILED — recipe, not dimension, was measured

Instrument gate (implicit until now, explicit from here on): the
ladder is only interpretable if the LARGEST k fits the teacher well
(target: spike F1 > 0.9, RMSE < 5 mV). Run 1, 12 epochs, linear
readout, pure weighted-MSE: k=1 RMSE 17.9/F1 0.00, k=2 18.8/0.00,
k=4 15.0/0.51, k=8 20.2/0.23. Non-monotone in k, no surrogate
fires on constant drive (all f-I RMSE = 53 Hz = the never-spikes
score), zero rebounds. MSE alone prefers blurred spikes; the
optimizer, not the state count, is binding. No prediction can be
scored from this run.

Recipe v2 (declared before running): memoryless MLP readouts
(k->32->1; adds no state, ladder still measures k), auxiliary
spike head trained with BCE on +-0.3 ms spike indicators
(training aid; PRIMARY F1 still scored by the same 0 mV voltage-
crossing detector as the teacher), spike-region weight 10x,
30 epochs, cosine lr 3e-3 -> 3e-4.

## Related work (added 2026-09-01, user-supplied)

1. Wan, Karniadakis & Stinis, "From LIF to QIF: toward
   differentiable spiking neurons for scientific ML" (npj AI,
   2026; s44387-026-00121-2). QIF's quadratic subthreshold keeps
   spiking smooth enough for backprop — the mirror image of M12's
   HEU non-differentiability blocker. Hook -> R2/R5: add QIF as a
   1-state arm; it varies NONLINEARITY at fixed k, orthogonal to
   our k-ladder (GRU fixed). Sharpens P2: QIF onset is type-I-like;
   a 1-state unit of any nonlinearity should still fail HH's
   type II onset and rebound (both need a second, resonant state).
2. Tandale & Stoffel, "Meta-learning hybrid spiking networks as
   physics-based nonlinear solvers" (npj Unconventional Computing,
   2026; s44335-025-00048-y). LIF-gated graded outputs (spike
   mechanism as learned selectivity/regularization), Loihi-2
   deployment. Precedent that spiking dynamics pay as parameter
   efficiency in trained solvers; adjacent to our rwkv-spiking
   line more than to rung 1.
3. Freddi et al., "Mean-field criticality in spiking networks for
   reservoir computing" (Sci Reports, 2025; s41598-025-18004-y).
   Closed-form critical coupling <W>_crit for LIF reservoirs —
   principled fixed-physics initialization, no tuning. Hook -> R4:
   when composing many k-state units, their formula is a candidate
   operating point; test edge-of-chaos vs off-critical composition
   with our stability metrics (M10-style fixed dynamics + trained
   readouts is exactly their regime).
