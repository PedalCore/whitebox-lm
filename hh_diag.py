"""M13 experiment A0 — the diagnostic fork. No latent, no
encoder/decoder, no recurrence.

Mode 'step': MLP F(V,m,h,n,I) -> next normalized state (residual,
0.1 ms flow map), trained on all teacher transitions as iid pairs.
Mode 'deriv': MLP learns the ANALYTIC vector field (HH RHS at
recorded states, per-ms units) and rollout integrates it with 10
Euler substeps per 0.1 ms record step.

Reports (1) teacher-forced one-step RMSE per variable and (2)
autonomous rollout from rest: V-RMSE, spike F1, f-I, rebound.

python3 -m whitebox.hh_diag [--mode step|deriv] [--epochs 8]
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from whitebox.hh_teacher import (C, DT, ENA, EK, EL, GK, GL, GNA,  # noqa
                                 REC_EVERY, init_state, rates,
                                 spikes_from_v)
from whitebox.hh_surrogate import spike_f1                         # noqa

OUT = pathlib.Path('whitebox/runs/m13')
VS, VOFF = 100.0, 65.0
IS = 10.0
SUB = 10                     # Euler substeps per record step (deriv)


def norm_state(V, G):
    return np.stack([(V + VOFF) / VS, G[0], G[1], G[2]],
                    axis=-1).astype(np.float32)


def hh_rhs(S, I):
    """Analytic HH RHS at normalized states. S (...,4) norm, I raw.
    Returns d(norm state)/dt in per-ms units."""
    V = S[..., 0] * VS - VOFF
    m, h, n = S[..., 1], S[..., 2], S[..., 3]
    am, bm, ah, bh, an, bn = rates(V)
    dV = (I - GNA * m ** 3 * h * (V - ENA) - GK * n ** 4 * (V - EK)
          - GL * (V - EL)) / C
    return np.stack([dV / VS, am * (1 - m) - bm * m,
                     ah * (1 - h) - bh * h,
                     an * (1 - n) - bn * n], axis=-1)


class F(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 128), nn.Tanh(),
                                 nn.Linear(128, 128), nn.Tanh(),
                                 nn.Linear(128, 4))

    def forward(self, s, i):
        return self.net(torch.cat([s, i], -1))


def rollout(model, I_mv, s0, mode, dev, bs=32):
    model.eval()
    outs = []
    dt_sub = DT * REC_EVERY / SUB          # 0.01 ms
    with torch.no_grad():
        for b0 in range(0, len(I_mv), bs):
            Ib = torch.tensor(I_mv[b0:b0 + bs] / IS,
                              dtype=torch.float32, device=dev)
            s = torch.tensor(np.repeat(s0[None], len(Ib), 0),
                             dtype=torch.float32, device=dev)
            traj = []
            for t in range(Ib.shape[1]):
                i_t = Ib[:, t:t + 1]
                if mode == 'step':
                    s = s + model(s, i_t)
                else:
                    for _ in range(SUB):
                        s = s + dt_sub * model(s, i_t)
                s = torch.clamp(s, -0.5, 1.5)
                traj.append(s)
            outs.append(torch.stack(traj, 1).cpu().numpy())
    return np.concatenate(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='step',
                    choices=['step', 'deriv'])
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--dev', default='cpu')
    args = ap.parse_args()
    dev = args.dev
    d = dict(np.load(OUT / 'hh_data_full.npz'))
    Str = norm_state(d['train_V'], d['train_G'])
    Ste = norm_state(d['test_V'], d['test_G'])
    # transition pairs
    X = Str[:, :-1].reshape(-1, 4)
    Inow = (d['train_I'][:, :-1].reshape(-1, 1) / IS)
    if args.mode == 'step':
        Y = (Str[:, 1:] - Str[:, :-1]).reshape(-1, 4)
    else:
        Y = hh_rhs(Str[:, :-1],
                   d['train_I'][:, :-1]).reshape(-1, 4)
    X = torch.tensor(X)
    Inow = torch.tensor(Inow, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    scale = Y.std(0, keepdim=True) + 1e-8      # per-var whitening
    torch.manual_seed(0)
    model = F().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-4)
    N = len(X)
    for ep in range(args.epochs):
        perm = torch.randperm(N)
        tot = cnt = 0.0
        for b0 in range(0, N, 4096):
            idx = perm[b0:b0 + 4096]
            x, i, y = (X[idx].to(dev), Inow[idx].to(dev),
                       Y[idx].to(dev))
            loss = (((model(x, i) - y) / scale.to(dev))
                    ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
            cnt += len(idx)
        sched.step()
        print(f'{args.mode} ep{ep + 1}: whitened one-step MSE '
              f'{tot / cnt:.6f}', flush=True)
    # (1) teacher-forced one-step error on test, per variable
    Xte = torch.tensor(Ste[:, :-1].reshape(-1, 4))
    Ite = torch.tensor(d['test_I'][:, :-1].reshape(-1, 1) / IS,
                       dtype=torch.float32)
    if args.mode == 'step':
        Yte = torch.tensor((Ste[:, 1:] - Ste[:, :-1])
                           .reshape(-1, 4))
    else:
        Yte = torch.tensor(hh_rhs(Ste[:, :-1],
                                  d['test_I'][:, :-1])
                           .reshape(-1, 4), dtype=torch.float32)
    with torch.no_grad():
        pe = []
        for b0 in range(0, len(Xte), 65536):
            pe.append(model(Xte[b0:b0 + 65536].to(dev),
                            Ite[b0:b0 + 65536].to(dev)).cpu())
        pred = torch.cat(pe)
    one = torch.sqrt(((pred - Yte) ** 2).mean(0)).numpy()
    rel = one / Yte.std(0).numpy()
    print('one-step RMSE (V,m,h,n):',
          np.round(one, 6).tolist(), flush=True)
    print('one-step RMSE / target std:',
          np.round(rel, 4).tolist(), flush=True)
    # (2) autonomous rollout from rest
    V0, m0, h0, n0 = init_state(1)
    rest = np.array([(V0[0] + VOFF) / VS, m0[0], h0[0], n0[0]],
                    np.float32)
    tr = rollout(model, d['test_I'], rest, args.mode, dev)
    v_pred = tr[..., 0] * VS - VOFF
    v_true = Ste[..., 0] * VS - VOFF
    v_rmse = float(np.sqrt(np.mean((v_pred - v_true) ** 2)))
    f1 = spike_f1(v_true, v_pred)
    amps = d['fi_amps']
    T = int(1200.0 / (DT * REC_EVERY))
    v_fi = rollout(model, np.repeat(amps[:, None], T, 1), rest,
                   args.mode, dev)[..., 0] * VS - VOFF
    rate = np.array([len(spikes_from_v(x[2000:])) for x in v_fi])
    fi_rmse = float(np.sqrt(np.mean((rate - d['fi_rate']) ** 2)))
    T2 = int(400.0 / (DT * REC_EVERY))
    I2 = np.zeros((1, T2))
    I2[0, :T2 // 2] = -3.0
    v_r = rollout(model, I2, rest, args.mode, dev)[0, :, 0] \
        * VS - VOFF
    reb = len(spikes_from_v(v_r[T2 // 2:]))
    res = dict(mode=args.mode, one_step_rel=np.round(rel, 4)
               .tolist(), v_rmse_mv=round(v_rmse, 2),
               spike_f1=round(f1, 3), fi_rmse_hz=round(fi_rmse, 1),
               rebound_spikes=reb)
    print('RESULT', json.dumps(res), flush=True)
    json.dump(res, open(OUT / f'diag_{args.mode}.json', 'w'))


if __name__ == '__main__':
    main()
