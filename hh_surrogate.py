"""M13 rung 1 — learned k-state surrogates of the HH teacher.

k-state GRU cell + linear voltage readout, k in {1,2,4,8}. TBPTT
over 1000-step (100 ms) chunks. Voltage MSE with 4x weight on
spike-region samples (teacher V > -20 mV). Reports voltage RMSE,
spike-timing F1 (+-2 ms), and the two OOD signatures (f-I curve
error, anodal-break rebound) per k.

python3 -m whitebox.hh_surrogate [--ks 1,2,4,8] [--epochs 12]
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

OUT = pathlib.Path('whitebox/runs/m13')
CHUNK = 1000
VS, VOFF = 100.0, 65.0        # V_norm = (V + 65) / 100
IS = 10.0                     # I_norm = I / 10


class Surrogate(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.cell = nn.GRUCell(1, k)
        self.out = nn.Linear(k, 1)
        self.k = k

    def forward(self, i_seq, h=None):
        """i_seq (B,T) normalized current -> v (B,T) normalized."""
        B, T = i_seq.shape
        if h is None:
            h = i_seq.new_zeros(B, self.k)
        vs = []
        for t in range(T):
            h = self.cell(i_seq[:, t:t + 1], h)
            vs.append(self.out(h))
        return torch.cat(vs, 1), h


def run_full(model, I_mv, dev, bs=32):
    """I_mv (B,T) raw current -> predicted V in mV (numpy)."""
    model.eval()
    outs = []
    with torch.no_grad():
        for b0 in range(0, len(I_mv), bs):
            x = torch.tensor(I_mv[b0:b0 + bs] / IS,
                             dtype=torch.float32, device=dev)
            v, _ = model(x)
            outs.append(v.cpu().numpy() * VS - VOFF)
    return np.concatenate(outs, 0)


def spike_f1(v_true, v_pred, tol=2.0):
    tp = fp = fn = 0
    for a, b in zip(v_true, v_pred):
        st, sp = spikes_from_v(a), spikes_from_v(b)
        used = np.zeros(len(st), bool)
        for t in sp:
            j = np.flatnonzero(~used & (np.abs(st - t) <= tol))
            if len(j):
                used[j[0]] = True
                tp += 1
            else:
                fp += 1
        fn += int((~used).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-9)


def eval_signatures(model, d, dev):
    """OOD: f-I curve RMSE (Hz) + rebound spike count."""
    amps = d['fi_amps']
    T = int(1200.0 / (DT * REC_EVERY))
    I = np.repeat(amps[:, None], T, 1)
    v = run_full(model, I, dev)
    rate = np.array([len(spikes_from_v(x[2000:])) for x in v])
    fi_rmse = float(np.sqrt(np.mean((rate - d['fi_rate']) ** 2)))
    T2 = int(400.0 / (DT * REC_EVERY))
    I2 = np.zeros((1, T2))
    I2[0, :T2 // 2] = -3.0
    v2 = run_full(model, I2, dev)[0]
    reb = len(spikes_from_v(v2[T2 // 2:]))
    return fi_rmse, reb


def train_one(k, d, dev, epochs, seed=0, use_wandb=False):
    torch.manual_seed(seed)
    model = Surrogate(k).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    Itr = torch.tensor(d['train_I'] / IS, dtype=torch.float32)
    Vtr = torch.tensor((d['train_V'] + VOFF) / VS,
                       dtype=torch.float32)
    W = torch.where(Vtr > 0.45, 4.0, 1.0)      # V > -20 mV
    nprm = sum(p.numel() for p in model.parameters())
    B, T = Itr.shape
    for ep in range(epochs):
        perm = torch.randperm(B)
        tot = cnt = 0.0
        for b0 in range(0, B, 32):
            idx = perm[b0:b0 + 32]
            h = None
            for c0 in range(0, T, CHUNK):
                x = Itr[idx, c0:c0 + CHUNK].to(dev)
                y = Vtr[idx, c0:c0 + CHUNK].to(dev)
                w = W[idx, c0:c0 + CHUNK].to(dev)
                v, h = model(x, None if h is None else h.detach())
                loss = ((v - y) ** 2 * w).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * x.numel()
                cnt += x.numel()
        va = run_full(model, d['val_I'], dev)
        rmse = float(np.sqrt(np.mean((va - d['val_V']) ** 2)))
        print(f'k={k} ep{ep + 1}: train {tot / cnt:.5f} '
              f'val-RMSE {rmse:.2f} mV', flush=True)
        if use_wandb:
            import wandb
            wandb.log({f'k{k}/train_loss': tot / cnt,
                       f'k{k}/val_rmse': rmse, 'epoch': ep + 1})
    te = run_full(model, d['test_I'], dev)
    rmse = float(np.sqrt(np.mean((te - d['test_V']) ** 2)))
    f1 = spike_f1(d['test_V'], te)
    fi_rmse, reb = eval_signatures(model, d, dev)
    res = dict(k=k, params=nprm, test_rmse_mv=round(rmse, 2),
               spike_f1=round(f1, 3), fi_rmse_hz=round(fi_rmse, 1),
               rebound_spikes=reb)
    print('RESULT', json.dumps(res), flush=True)
    torch.save(model.state_dict(), OUT / f'surrogate_k{k}.pt')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ks', default='1,2,4,8')
    ap.add_argument('--epochs', type=int, default=12)
    args = ap.parse_args()
    dev = ('mps' if torch.backends.mps.is_available() else 'cpu')
    d = dict(np.load(OUT / 'hh_data.npz'))
    use_wandb = False
    try:
        import wandb
        wandb.init(project='m13-state', name='rung1-ladder',
                   config=vars(args))
        use_wandb = True
    except Exception as e:
        print('wandb off:', e, flush=True)
    teacher_reb = len(spikes_from_v(
        d['rebound_v'][int(200.0 / (DT * REC_EVERY)):]))
    print(f'teacher rebound spikes: {teacher_reb}', flush=True)
    results = [train_one(int(k), d, dev, args.epochs,
                         use_wandb=use_wandb)
               for k in args.ks.split(',')]
    json.dump(results, open(OUT / 'ladder_results.json', 'w'),
              indent=1)
    print('\n=== LADDER ===', flush=True)
    for r in results:
        print(r, flush=True)


if __name__ == '__main__':
    main()
