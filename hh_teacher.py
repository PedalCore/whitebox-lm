"""M13 rung 1 — Hodgkin-Huxley teacher (true dimension 4).

Standard squid-axon HH, exponential-Euler gates at dt=0.01 ms,
recorded every 0.1 ms. Drive = OU current + random step pulses.
Writes whitebox/runs/m13/hh_data.npz (train/val/test V + I traces)
and prints the two OOD behavioral signatures (f-I curve, anodal-
break rebound) for later surrogate comparison.

python3 -m whitebox.hh_teacher
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

OUT = pathlib.Path('whitebox/runs/m13')
DT = 0.01            # ms, internal
REC_EVERY = 10       # record at 0.1 ms
T_MS = 1000.0
C, GNA, GK, GL = 1.0, 120.0, 36.0, 0.3
ENA, EK, EL = 50.0, -77.0, -54.387


def _safe_exp_ratio(x, scale):
    """x / (1 - exp(-x/scale)), with the x->0 limit handled."""
    x = np.asarray(x, float)
    return np.where(np.abs(x) < 1e-6, scale,
                    x / (1.0 - np.exp(-x / scale)))


def rates(V):
    am = 0.1 * _safe_exp_ratio(V + 40.0, 10.0)
    bm = 4.0 * np.exp(-(V + 65.0) / 18.0)
    ah = 0.07 * np.exp(-(V + 65.0) / 20.0)
    bh = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    an = 0.01 * _safe_exp_ratio(V + 55.0, 10.0)
    bn = 0.125 * np.exp(-(V + 65.0) / 80.0)
    return am, bm, ah, bh, an, bn


def init_state(B):
    V = np.full(B, -65.0)
    am, bm, ah, bh, an, bn = rates(V)
    return [V, am / (am + bm), ah / (ah + bh), an / (an + bn)]


def simulate(I_of_t, B):
    """I_of_t: (B, N_internal) current. Returns recorded V (B, N/10)."""
    V, m, h, n = init_state(B)
    N = I_of_t.shape[1]
    rec = np.empty((B, N // REC_EVERY), np.float32)
    for t in range(N):
        am, bm, ah, bh, an, bn = rates(V)
        # exponential Euler on gates
        for g, a, b in ((0, am, bm), (1, ah, bh), (2, an, bn)):
            x = (m, h, n)[g]
            tau = 1.0 / (a + b)
            inf = a * tau
            x[:] = inf + (x - inf) * np.exp(-DT / tau)
        ina = GNA * m ** 3 * h * (V - ENA)
        ik = GK * n ** 4 * (V - EK)
        il = GL * (V - EL)
        V += DT * (I_of_t[:, t] - ina - ik - il) / C
        if t % REC_EVERY == REC_EVERY - 1:
            rec[:, t // REC_EVERY] = V
    return rec


def make_drive(B, N, rng):
    """OU + step pulses, per-sequence regimes. (B, N) at dt=DT."""
    mu = rng.uniform(0.0, 8.0, B)
    sig = rng.uniform(0.5, 4.0, B)
    tau = rng.uniform(3.0, 50.0, B)
    I = np.empty((B, N), np.float32)
    x = mu.copy()
    k = np.exp(-DT / tau)
    s = sig * np.sqrt(1.0 - k ** 2)
    for t in range(N):
        x = mu + (x - mu) * k + s * rng.standard_normal(B)
        I[:, t] = x
    for b in range(B):
        for _ in range(rng.integers(0, 4)):
            t0 = rng.integers(0, N - 2000)
            dur = rng.integers(2000, 20000)
            I[b, t0:t0 + dur] += rng.uniform(-3.0, 12.0)
    return np.clip(I, -5.0, 20.0)


def spikes_from_v(v, dt_ms=0.1):
    """Upward 0 mV crossings, 2 ms dedup. v: (T,). -> times (ms)."""
    up = np.flatnonzero((v[1:] >= 0.0) & (v[:-1] < 0.0)) + 1
    out, last = [], -1e9
    for i in up:
        t = i * dt_ms
        if t - last >= 2.0:
            out.append(t)
            last = t
    return np.array(out)


def fi_curve():
    amps = np.arange(0.0, 15.1, 0.5)
    N = int(1200.0 / DT)
    I = np.repeat(amps[:, None], N, 1).astype(np.float32)
    v = simulate(I, len(amps))
    rate = [len(spikes_from_v(v[i, 2000:])) / 1.0 for i in
            range(len(amps))]
    return amps, np.array(rate)


def rebound():
    N = int(400.0 / DT)
    I = np.zeros((1, N), np.float32)
    I[0, :int(200.0 / DT)] = -3.0
    v = simulate(I, 1)[0]
    post = spikes_from_v(v[2000:])
    return v, post


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    N = int(T_MS / DT)
    splits = {}
    for name, B in (('train', 256), ('val', 32), ('test', 32)):
        I = make_drive(B, N, rng)
        V = simulate(I, B)
        Irec = I[:, REC_EVERY - 1::REC_EVERY].astype(np.float32)
        splits[name + '_I'] = Irec
        splits[name + '_V'] = V
        ns = np.mean([len(spikes_from_v(V[b])) for b in range(B)])
        print(f'{name}: {B} seqs, mean {ns:.1f} spikes/s', flush=True)
    amps, rate = fi_curve()
    vreb, reb = rebound()
    onset = amps[np.argmax(rate > 0)] if (rate > 0).any() else np.nan
    f_at = rate[np.argmax(rate > 0)]
    print(f'f-I: rheobase ~{onset} uA/cm^2, onset rate {f_at:.0f} Hz '
          f'(type II expects discontinuous ~50)', flush=True)
    print(f'anodal-break rebound spikes after release: {len(reb)} '
          f'at {np.round(reb, 1)} ms', flush=True)
    np.savez_compressed(OUT / 'hh_data.npz', fi_amps=amps,
                        fi_rate=rate, rebound_v=vreb, **splits)
    print('wrote', OUT / 'hh_data.npz', flush=True)


if __name__ == '__main__':
    main()
