"""M13 experiment A — MECHANISTIC compression: full-state
supervision with a k-latent bottleneck.

Model: encoder E(s_0) -> tanh -> z_0 (k,); latent dynamics = the
SAME GRU cell class as runs 1-4, driven by current only; decoder
D(z) -> (V, m, h, n). Autonomous rollout from E(rest); supervision
on all four teacher states. Implementation gate: k=8 must reach
near-perfect rollout (V-RMSE < 2 mV, F1 > 0.95) — otherwise the
problem is implementation, not capacity. Sweep k = 1,2,3,4,8.

python3 -m whitebox.hh_mech [--ks 1,2,3,4,8] [--epochs 20]
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from whitebox.hh_teacher import DT, REC_EVERY, spikes_from_v   # noqa
from whitebox.hh_surrogate import spike_f1                     # noqa

OUT = pathlib.Path('whitebox/runs/m13')
CHUNK = 1000
VS, VOFF = 100.0, 65.0
IS = 10.0


def norm_state(V, G):
    """V (B,T) mV, G (3,B,T) gates -> (B,T,4) normalized."""
    return np.stack([(V + VOFF) / VS, G[0], G[1], G[2]],
                    axis=-1).astype(np.float32)


class Mech(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(4, 64), nn.Tanh(),
                                 nn.Linear(64, k), nn.Tanh())
        self.cell = nn.GRUCell(1, k)
        self.dec = nn.Sequential(nn.Linear(k, 64), nn.Tanh(),
                                 nn.Linear(64, 4))
        self.k = k

    def forward(self, i_seq, z=None, s0=None):
        """i_seq (B,T) normalized current; z carried latent or s0
        (B,4) to encode. Returns (states (B,T,4), z)."""
        if z is None:
            z = self.enc(s0)
        zs = []
        for t in range(i_seq.shape[1]):
            z = self.cell(i_seq[:, t:t + 1], z)
            zs.append(z)
        Z = torch.stack(zs, 1)
        return self.dec(Z), z


def rollout(model, I_mv, s0, dev, bs=32):
    """Free-run from encoded s0. Returns decoded states numpy."""
    model.eval()
    outs = []
    with torch.no_grad():
        for b0 in range(0, len(I_mv), bs):
            x = torch.tensor(I_mv[b0:b0 + bs] / IS,
                             dtype=torch.float32, device=dev)
            s = torch.tensor(s0[b0:b0 + bs], dtype=torch.float32,
                             device=dev)
            y, _ = model(x, s0=s)
            outs.append(y.cpu().numpy())
    return np.concatenate(outs, 0)


REST = None     # (4,) normalized rest state, set in main


def eval_split(model, I, S, dev):
    pred = rollout(model, I, S[:, 0], dev)
    v_pred = pred[..., 0] * VS - VOFF
    v_true = S[..., 0] * VS - VOFF
    v_rmse = float(np.sqrt(np.mean((v_pred - v_true) ** 2)))
    g_rmse = float(np.sqrt(np.mean((pred[..., 1:] - S[..., 1:])
                                   ** 2)))
    f1 = spike_f1(v_true, v_pred)
    return v_rmse, g_rmse, f1


def eval_signatures(model, d, dev):
    amps = d['fi_amps']
    T = int(1200.0 / (DT * REC_EVERY))
    I = np.repeat(amps[:, None], T, 1)
    s0 = np.repeat(REST[None], len(amps), 0)
    v = rollout(model, I, s0, dev)[..., 0] * VS - VOFF
    rate = np.array([len(spikes_from_v(x[2000:])) for x in v])
    fi_rmse = float(np.sqrt(np.mean((rate - d['fi_rate']) ** 2)))
    T2 = int(400.0 / (DT * REC_EVERY))
    I2 = np.zeros((1, T2))
    I2[0, :T2 // 2] = -3.0
    v2 = rollout(model, I2, REST[None], dev)[0, :, 0] * VS - VOFF
    return fi_rmse, len(spikes_from_v(v2[T2 // 2:]))


def train_one(k, d, Str, Sval, Ste, dev, epochs, use_wandb=False):
    torch.manual_seed(0)
    model = Mech(k).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=3e-4)
    Itr = torch.tensor(d['train_I'] / IS, dtype=torch.float32)
    Ytr = torch.tensor(Str)
    W = torch.ones(Ytr.shape[:2])
    W[Ytr[..., 0] > 0.45] = 10.0            # spike-region weight
    nprm = sum(p.numel() for p in model.parameters())
    B, T = Itr.shape
    for ep in range(epochs):
        perm = torch.randperm(B)
        tot = cnt = 0.0
        for b0 in range(0, B, 32):
            idx = perm[b0:b0 + 32]
            z = None
            for c0 in range(0, T, CHUNK):
                x = Itr[idx, c0:c0 + CHUNK].to(dev)
                y = Ytr[idx, c0:c0 + CHUNK].to(dev)
                w = W[idx, c0:c0 + CHUNK].to(dev)
                if z is None:
                    pred, z = model(x, s0=y[:, 0])
                else:
                    pred, z = model(x, z=z.detach())
                loss = (((pred - y) ** 2).mean(-1) * w).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * x.numel()
                cnt += x.numel()
        sched.step()
        vr, gr, f1 = eval_split(model, d['val_I'], Sval, dev)
        print(f'k={k} ep{ep + 1}: train {tot / cnt:.5f} val '
              f'V-RMSE {vr:.2f} mV gates {gr:.3f} F1 {f1:.2f}',
              flush=True)
        if use_wandb:
            import wandb
            wandb.log({f'mechk{k}/val_vrmse': vr,
                       f'mechk{k}/val_f1': f1, 'epoch': ep + 1})
    vr, gr, f1 = eval_split(model, d['test_I'], Ste, dev)
    fi_rmse, reb = eval_signatures(model, d, dev)
    res = dict(k=k, params=nprm, v_rmse_mv=round(vr, 2),
               gate_rmse=round(gr, 4), spike_f1=round(f1, 3),
               fi_rmse_hz=round(fi_rmse, 1), rebound_spikes=reb)
    print('RESULT', json.dumps(res), flush=True)
    torch.save(model.state_dict(), OUT / f'mech_k{k}.pt')
    return res


def main():
    global REST
    ap = argparse.ArgumentParser()
    ap.add_argument('--ks', default='8,4,3,2,1')   # gate arm first
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--dev', default='cpu')
    args = ap.parse_args()
    d = dict(np.load(OUT / 'hh_data_full.npz'))
    Str = norm_state(d['train_V'], d['train_G'])
    Sval = norm_state(d['val_V'], d['val_G'])
    Ste = norm_state(d['test_V'], d['test_G'])
    from whitebox.hh_teacher import init_state
    V0, m0, h0, n0 = init_state(1)
    REST = np.array([(V0[0] + VOFF) / VS, m0[0], h0[0], n0[0]],
                    np.float32)
    use_wandb = False
    try:
        import wandb
        wandb.init(project='m13-state', name='expA-mech',
                   config=vars(args))
        use_wandb = True
    except Exception as e:
        print('wandb off:', e, flush=True)
    results = [train_one(int(k), d, Str, Sval, Ste, args.dev,
                         args.epochs, use_wandb)
               for k in args.ks.split(',')]
    json.dump(results, open(OUT / 'mech_results.json', 'w'),
              indent=1)
    print('=== EXP-A DONE ===', flush=True)


if __name__ == '__main__':
    main()
