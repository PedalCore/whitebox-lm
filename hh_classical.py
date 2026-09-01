"""M13 rung 2 — classical hand-designed units fitted to the HH
teacher by CMA-ES: LIF (1 state + reset), Izhikevich (2), AdEx (2).

Spikes are the models' explicit reset events. Fitting loss =
(1 - F1_2ms) + 0.05 * subthreshold-RMSE/10mV on 16 train seqs;
CMA-ES sigma0 0.5, 4 restarts x 60 iters, pop 16. Scored on the
full test split + the two OOD signatures.

python3 -m whitebox.hh_classical
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from whitebox.hh_teacher import spikes_from_v                  # noqa

OUT = pathlib.Path('whitebox/runs/m13')
DT = 0.1                    # ms — classical arms run at recorded dt
NFIT = 16


# ---- models: vectorized over (P population, B sequences) ----
# Each returns (spike raster list per (p,b), subthreshold V in mV).

def sim_lif(theta, I):
    """theta (P,5): log tau, EL, R, thresh-offset, log t_ref."""
    P, (B, T) = len(theta), I.shape
    tau = 20.0 * np.exp(theta[:, 0])[:, None]
    EL = -65.0 + 10.0 * theta[:, 1][:, None]
    R = 2.0 * np.exp(theta[:, 2])[:, None]
    thr = EL + 15.0 + 5.0 * theta[:, 3][:, None]
    tref = 2.0 * np.exp(np.clip(theta[:, 4], -2, 2))[:, None]
    v = np.broadcast_to(EL, (P, B)).copy()
    refr = np.zeros((P, B))
    V = np.empty((P, B, T), np.float32)
    S = np.zeros((P, B, T), bool)
    for t in range(T):
        act = refr <= 0
        v = np.where(act, v + DT * (-(v - EL) + R * I[:, t]) / tau, v)
        spk = act & (v >= thr)
        S[:, :, t] = spk
        v = np.where(spk, EL, v)
        refr = np.where(spk, tref, refr - DT)
        V[:, :, t] = v
    return S, V


def sim_izhi(theta, I):
    """theta (P,6): a,b,c,d (offsets), input gain, input offset."""
    P, (B, T) = len(theta), I.shape
    a = 0.02 * np.exp(theta[:, 0])[:, None]
    b = (0.2 + 0.1 * theta[:, 1])[:, None]
    c = (-65.0 + 5.0 * theta[:, 2])[:, None]
    d = (6.0 * np.exp(theta[:, 3]))[:, None]
    g = (10.0 * np.exp(theta[:, 4]))[:, None]
    off = (5.0 * theta[:, 5])[:, None]
    v = np.full((P, B), -65.0)
    u = b * v
    V = np.empty((P, B, T), np.float32)
    S = np.zeros((P, B, T), bool)
    for t in range(T):
        Iin = g * I[:, t] + off
        v = v + DT * (0.04 * v * v + 5.0 * v + 140.0 - u + Iin)
        u = u + DT * a * (b * v - u)
        v = np.clip(v, -120.0, 40.0)
        spk = v >= 30.0
        S[:, :, t] = spk
        v = np.where(spk, c, v)
        u = np.where(spk, u + d, u)
        V[:, :, t] = v
    return S, V


def sim_adex(theta, I):
    """theta (P,7): gL, DeltaT, VT, tau_w, a, b, input gain."""
    P, (B, T) = len(theta), I.shape
    C = 1.0
    gL = 0.05 * np.exp(theta[:, 0])[:, None]
    dT = 2.0 * np.exp(np.clip(theta[:, 1], -1.5, 1.5))[:, None]
    VT = (-50.0 + 5.0 * theta[:, 2])[:, None]
    tw = 50.0 * np.exp(theta[:, 3])[:, None]
    aa = 0.02 * np.exp(np.clip(theta[:, 4], -3, 3))[:, None]
    bb = 0.5 * np.exp(np.clip(theta[:, 5], -3, 3))[:, None]
    g = np.exp(theta[:, 6])[:, None]
    EL, Vr = -65.0, -58.0
    v = np.full((P, B), EL)
    w = np.zeros((P, B))
    V = np.empty((P, B, T), np.float32)
    S = np.zeros((P, B, T), bool)
    for t in range(T):
        ex = gL * dT * np.exp(np.clip((v - VT) / dT, -20.0, 20.0))
        v = v + DT * (-gL * (v - EL) + ex - w + g * I[:, t]) / C
        w = w + DT * (aa * (v - EL) - w) / tw
        spk = v >= 0.0
        S[:, :, t] = spk
        v = np.where(spk, Vr, np.clip(v, -120.0, 20.0))
        w = np.where(spk, w + bb, w)
        V[:, :, t] = v
    return S, V


MODELS = {'lif': (sim_lif, 5), 'izhikevich': (sim_izhi, 6),
          'adex': (sim_adex, 7)}


def f1_raster(S_pb, true_times, dt=DT, tol=2.0):
    """S_pb (T,) bool raster vs teacher spike times (ms)."""
    sp = np.flatnonzero(S_pb) * dt
    used = np.zeros(len(true_times), bool)
    tp = fp = 0
    for t in sp:
        j = np.flatnonzero(~used & (np.abs(true_times - t) <= tol))
        if len(j):
            used[j[0]] = True
            tp += 1
        else:
            fp += 1
    fn = int((~used).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-9)


def fit(name, d, seed=0):
    import cma
    simf, nd = MODELS[name]
    Ifit = d['train_I'][:NFIT].astype(np.float64)
    true = [spikes_from_v(v) for v in d['train_V'][:NFIT]]
    Vt = d['train_V'][:NFIT]
    sub = Vt < -50.0

    def loss(thetas):
        th = np.asarray(thetas)
        S, V = simf(th, Ifit)
        out = []
        for p in range(len(th)):
            f1 = np.mean([f1_raster(S[p, b], true[b])
                          for b in range(NFIT)])
            # affine-map subthreshold voltage, then RMSE
            x, y = V[p][sub], Vt[sub]
            A = np.vstack([x, np.ones_like(x)]).T
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            rmse = float(np.sqrt(np.mean((A @ coef - y) ** 2)))
            out.append((1.0 - f1) + 0.05 * rmse / 10.0)
        return out

    best, bl = None, np.inf
    for r in range(4):
        es = cma.CMAEvolutionStrategy(
            np.zeros(nd), 0.5, {'popsize': 16, 'maxiter': 60,
                                'seed': seed + 42 + r,
                                'verbose': -9})
        while not es.stop():
            xs = es.ask()
            ls = loss(xs)
            es.tell(xs, ls)
            i = int(np.argmin(ls))
            if ls[i] < bl:
                bl, best = ls[i], np.array(xs[i])
        print(f'{name} restart {r + 1}: best loss {bl:.4f}',
              flush=True)
    return best, bl


def score(name, theta, d):
    simf, _ = MODELS[name]
    th = theta[None]
    S, _ = simf(th, d['test_I'].astype(np.float64))
    true = [spikes_from_v(v) for v in d['test_V']]
    f1 = float(np.mean([f1_raster(S[0, b], true[b])
                        for b in range(len(true))]))
    amps = d['fi_amps']
    T = int(1200.0 / DT)
    Ifi = np.repeat(amps[:, None], T, 1)
    Sfi, _ = simf(th, Ifi)
    rate = Sfi[0, :, 2000:].sum(1).astype(float)
    fi_rmse = float(np.sqrt(np.mean((rate - d['fi_rate']) ** 2)))
    T2 = int(400.0 / DT)
    Ir = np.zeros((1, T2))
    Ir[0, :T2 // 2] = -3.0
    Sr, _ = simf(th, Ir)
    reb = int(Sr[0, 0, T2 // 2:].sum())
    return dict(arm=name, spike_f1=round(f1, 3),
                fi_rmse_hz=round(fi_rmse, 1), rebound_spikes=reb,
                theta=[round(float(x), 3) for x in theta])


def main():
    d = dict(np.load(OUT / 'hh_data.npz'))
    results = []
    for name in MODELS:
        theta, bl = fit(name, d)
        r = score(name, theta, d)
        print('RESULT', json.dumps(r), flush=True)
        results.append(r)
    json.dump(results, open(OUT / 'classical_results.json', 'w'),
              indent=1)
    print('=== R2 DONE ===', flush=True)


if __name__ == '__main__':
    main()
