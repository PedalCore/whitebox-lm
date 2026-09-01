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

## Run 2 (2026-09-01): GATE FAILED AGAIN — optimization variance
## dominates; declaring recipe v3 (teacher forcing)

v2 results: k=1 RMSE 25.7/F1 0.00, k=2 25.4/0.00, k=4 23.2/0.00
(WORSE than its own v1 run, 0.51 — same recipe, different seed
behavior), k=8 17.7/0.67. Still no constant-drive firing, no
rebound. Diagnosis: free-run regression onto sharp rare events —
mistimed spikes cost double (miss + false alarm), so the optimizer
suppresses spiking; run-to-run variance confirms a hard loss
landscape, not a capacity limit.

Recipe v3 (declared before running): autoregressive observable
feedback with scheduled sampling — input becomes [I_t, v_{t-1}],
where v_{t-1} is the TEACHER's voltage with probability 1-eps and
the model's own (detached) prediction otherwise; eps ramps 0 -> 1
over the first 60% of epochs; evaluation is ALWAYS full free-run
from rest. This is standard teacher forcing from the dynamical-
system-reconstruction literature, and the same fix (scheduled
sampling) that repaired M10-P calibration.

HONEST STATE ACCOUNTING: the fed-back voltage is itself a state
variable. Surrogate total state = k + 1. The ladder therefore
sweeps k in {1,2,3,4,8} (total 2,3,4,5,9), and P1 is restated:
saturation at TOTAL state 4, i.e. k=3. All other predictions
carry over with k read as total-state minus one.

## R2 declared in detail (2026-09-01, before running; v3 still going)

v3 interim: k=1,2,3 all F1 0.00 free-run (the eps -> 1 anneal
collapses every arm to silence so far). Whatever k=8 does, R2 runs
next: classical hand-designed units fitted to the SAME train
sequences by CMA-ES (loss = (1 - F1_2ms) + 0.05 * subthreshold
RMSE; 16 fitting seqs, 4 restarts), scored on the same test set +
signatures. Arms: LIF (1 state + reset), Izhikevich (2), AdEx (2).
Spikes for classical arms are their explicit reset events (their
mechanism, honestly theirs); voltage RMSE via affine map, reported
but secondary.

R2 predictions:
R2a. Fitted Izhikevich-2 beats every learned arm's current F1 by a
     wide margin — the bottleneck in runs 1-3 is TRAINABILITY of
     spiking dynamics under gradient descent, not state count.
R2b. LIF shows zero rebound spikes STRUCTURALLY (passive leak
     cannot overshoot on release); at least one of the 2-state
     arms recovers the anodal-break rebound.
R2c. No classical arm matches HH's type II f-I discontinuity as
     well as its overall F1 (Izhikevich fitted for timing tends
     type I unless parameters land in the resonator regime).

## Round 1 synthesis (2026-09-02): three learned recipes, one
## classical round — the honest scorecard

v3 final: ALL arms (k=1,2,3,4,8; total state 2-9) F1 0.00 in
free-run. Notably WORSE than v2 at k=8 (0.67 -> 0.00): detached
scheduled sampling taught reliance on teacher feedback, then
removed its quality with no gradient path to learn self-
correction — the documented bias of detached scheduled sampling.

R2 classical (CMA-ES, same data): LIF-1 F1 0.279 / f-I 107.5 /
rebound 0; Izhikevich-2 0.335 / 44.7 / 0; AdEx-2 0.368 / 49.0 / 0.

Prediction scoring:
- P1-P4: UNSCOREABLE — the instrument gate (largest k at F1>0.9)
  never passed in three recipes. The ladder has not yet measured
  HH's dimension.
- R2a: CONFIRMED with a caveat — every designed arm beats every
  learned arm (0.28-0.37 vs 0.00), but the designed ceiling is
  ~0.37, not mastery.
- R2b: HALF-FAILED — LIF's zero rebound is structural as
  predicted, but NEITHER 2-state arm recovered the rebound:
  timing-optimal parameters sit in the fast-spiking regime, not
  the resonator regime. Timing fit and signature fit are in
  tension at 2 states.
- R2c: CONFIRMED — no arm matches type II; Izhikevich's 44.7 is
  the only f-I score better than never-firing (53).

Two real findings survive the failed gate:
1. TRAINABILITY, not capacity, binds the learned arms: 9 total
   states + 3 recipes < 1 designed state + parameter search.
2. STATE, not design, binds the classical arms: AdEx and
   Izhikevich converge to fitting losses within 1e-4 of each
   other — a 2-state class ceiling (~F1 0.35) for fluctuation-
   driven HH timing, echoing the five-instances law from the
   designed side.

Recipe v4 (declared before running): return to the v2
configuration (no feedback input — the only arm ever to fire in
free-run), epochs 20 -> 60, ks {2,3,4,8}, seed 0, same gate.
If v4 also fails the gate, rung 1 concludes as a negative result
with the two findings above as its product, and R3+ proceed with
designed/hybrid arms.

## CORRECTIONS (2026-09-02, after collaborator review) — published,
## not patched

1. P1's framing was WRONG, not merely unscoreable. Canonical HH
   has 4 explicit state variables, but 4 is not its minimal
   observable/computational dimension: 2-D reductions (m -> m_inf(V)
   slaving, h/n collapse — Krinsky-Kokoz, Rinzel, the FitzHugh-
   Nagumo lineage) reproduce much of its behavior. The ladder
   measures the MINIMAL REALIZATION for a model class x input
   distribution x loss x observables — saturation at 2 would not
   be instrument failure, saturation at 4 would not prove
   dimension recovery. All rungs reread accordingly.
2. RETRACTED: "a 2-state class ceiling, not optimizer luck" (also
   posted publicly; correction posted to the project). Two 2-state
   families at similar losses do not bound the class of all
   2-state systems — objective, drive distribution, CMA budget,
   parameterization, and local optima are all uncontrolled.
   Replacement: two quite different canonical 2-state models
   converged to remarkably similar performance under the same
   criterion, SUGGESTING but not establishing a representational
   rather than model-specific bottleneck.
3. Downgraded: "you can't parameter-search your way out of
   missing state" is a hypothesis these results suggest (and M12
   demonstrated in ITS setting), not something demonstrated here.
4. Sharpened reading of runs 1-3: they measured capacity x
   optimization x rollout-stability x loss-geometry, entangled.
   Scheduled sampling itself has known consistency problems.
   Rather than v5/v6/v7 on the same entangled problem, redesign.

## REDESIGN — three compressions, experiment ladder A-D
## (declared before running)

A. MECHANISTIC: remove partial observability. Supervision on the
   full teacher state (V,m,h,n); model = encoder E(s_0) -> z_0,
   SAME GRU cell class as runs 1-4 (input = current only), decoder
   z -> (V,m,h,n). Implementation gate: k=8 must reach near-
   perfect rollout (V-RMSE < 2 mV, F1 > 0.95) — if it cannot,
   the problem is implementation, full stop. Then sweep
   k = 1,2,3,4,8. Expectations: k=1 fails; k=2 tracks the known
   2-D reduction (good V/spike behavior, imperfect gate
   trajectories); k >= 4 near-perfect. If A's gate passes where
   B's failed, partial observability + loss geometry — not the
   cell class — was the binding constraint of runs 1-4.
B. OBSERVABLE: current (+ voltage history) -> future voltage.
   Runs v1-v4 belong here; delay embeddings are the classical
   fix and a future arm.
C. FUNCTIONAL: behaviors only (spike timing, f-I, rebound,
   adaptation) — closest to "does extra state buy computation".
D. COMPUTATIONAL: downstream task performance per unit state
   (the R5 task rung, renamed).

The physical -> observable -> behavioral -> computational
progression is the project's actual question; rung-1's "knee at
k=4" was a special case and is retired as a headline claim.

## Experiment A interim + wording correction (2026-09-02)

Results so far (20 epochs, seed 0): k=8 F1 0.712 / V-RMSE 17.9 /
f-I 44.7 (fires under constant drive); k=4 0.468 / 21.7 / 53.0
(fires ONLY under fluctuating drive); k=3 0.000. Gate not passed.

WORDING CORRECTED (per review): the k=8 sub-gate result is "not
evidence for insufficient latent dimensionality or partial
observability" — NOT "a training/dynamics-class problem, not
capacity." With k=8 > 4 explicit teacher states and full
supervision, the unresolved bottleneck is among: optimization,
transition-class expressivity of the discrete GRU map,
discretization, rollout stability, loss scaling. "Capacity" also
includes the GRU transition's own expressive capacity.

Confounds logged before 2/1 land: (a) k=8's F1 was 0.00 through
ep9, then 0.60/0.67/0.70 — a late sudden transition, still
climbing at ep20; the 8-vs-4 gap partly reads as "larger models
cross the spiking transition earlier at fixed budget"; (b) single
seed — earlier non-monotone runs mandate multi-seed replication
before any capacity-law claim. What IS noteworthy: k=8 crossed a
QUALITATIVE boundary k=4 has not (sustained constant-drive
firing), suggesting extra state buys dynamical regime, not just
voltage error. Restrained summary: a provisional state-dependent
performance hierarchy under a fixed learned dynamics class —
not a measurement of minimal realization.

Normalization audit (per review): per-variable normalized stds
V 0.171 / m 0.211 / h 0.140 / n 0.116 — no variable dominates the
uniform state loss. Caveat stands: dynamical sensitivity is not
uniform (I_Na ~ m^3 h), so state-loss fidelity does not imply
dynamical fidelity.

## Experiment A0 declared (before running): the diagnostic fork

Above A in the ladder: NO latent, NO encoder/decoder, NO
recurrence. Plain MLP transition F(V,m,h,n,I) -> next state
(residual, 0.1 ms flow map), trained on all teacher transitions
as iid pairs. Two evaluations, separated:
1. teacher-forced ONE-STEP error — if not excellent, something
   basic is wrong (scaling/optimizer/architecture/loss);
2. AUTONOMOUS rollout from rest — recursive self-feeding.
Fork: one-step bad -> function-approximation problem; one-step
good + rollout bad -> stability/accumulated error; both good ->
the GRU formulation was the problem and the ladder rebuilds on
this transition class. Secondary mode: learn the analytic vector
field (RHS at recorded states) and integrate with Euler substeps
— separates vector-field learning from discretized transitions
(the CfC/continuous-time question, testable cheaply).

## Experiment A final + A0 fork results (2026-09-02)

A (full-state supervision, 20 ep, seed 0): k=1 F1 0.000 / k=2
0.000 / k=3 0.000 / k=4 0.468 / k=8 0.712. Monotone, with two
QUALITATIVE transitions: spiking appears between k=3 and k=4;
autonomous constant-drive firing appears between k=4 and k=8.
Restrained reading (per review): a provisional state-dependent
performance hierarchy under a fixed learned dynamics class — NOT
a minimal-realization measurement (gate unpassed; single seed;
k=8's late learning transition confounds budget with capacity).

A0 fork (no latents, no recurrence, MLP 128x128, 8 ep):
- step mode (0.1 ms flow map): one-step rel. RMSE V 0.37 / m 0.20
  / h 0.14 / n 0.06 — NOT excellent; rollout explodes (56.7 mV,
  F1 0.005). The flow map's curvature concentrates in the thin
  spike-upstroke region that iid sampling barely weights.
- deriv mode (analytic HH RHS, Euler substeps): one-step rel.
  RMSE 0.10 / 0.05 / 0.06 / 0.02 — 4x better (smoother target);
  rollout STILL fails (84 mV, F1 0.01) but differently: fires
  spuriously (4 false rebounds) and saturates the clamps rather
  than going silent. 5-10% vector-field error destroys the limit
  cycle.

Fork verdict so far: "one-step decent, rollout bad" — between the
review's branches. Before attributing to dynamical stability, the
one-step fit must be pushed to excellent (a closed-form smooth
RHS should fit to <1%).

A0b declared (before running): deriv mode, 40 epochs, width 256,
spike-region sample weighting. Question: does rollout fidelity
improve CONTINUOUSLY with vector-field precision, or is there a
precision cliff below which the limit cycle cannot be maintained?
Either answer is informative: continuous -> budget problem;
cliff -> quantifies the precision a learned dynamical
approximation of a neuron actually needs (directly relevant to
the substrate question: hand-designed dynamics carry their
qualitative regime for free; learned ones must buy it with
precision).
