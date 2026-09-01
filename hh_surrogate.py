"""M13 rung 1 — learned k-state surrogates of the HH teacher.

Recipe v3. k-state GRU cell, k in {1,2,3,4,8}, plus autoregressive
observable feedback (input = [I_t, v_{t-1}]) with scheduled
sampling: teacher voltage early, own detached prediction late (eps
ramps 0 -> 1 over the first 60% of epochs); evaluation is ALWAYS
full free-run from rest. HONEST STATE ACCOUNTING: fed-back voltage
is a state variable — total state = k + 1, so saturation at HH's
true dimension 4 predicts a knee at k=3. Memoryless MLP readouts
for voltage and an auxiliary spike head (training aid — primary F1
scored by 0 mV voltage crossings, same detector as the teacher).
Weighted voltage MSE (10x on V > -20 mV) + 0.5*BCE on +-0.3 ms
spike indicators. Instrument gate: interpret the ladder only if
the largest k fits well (F1 > 0.9, RMSE < 5 mV).

python3 -m whitebox.hh_surrogate [--ks 1,2,3,4,8] [--epochs 20]
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
    """k persistent states; readouts are memoryless MLPs (add no
    state, so the ladder still measures k)."""

    def __init__(self, k, fb=True):
        super().__init__()
        self.fb = fb                   # v4: fb=False = v2 recipe
        self.cell = nn.GRUCell(2, k)   # [current, v_feedback]
        self.out = nn.Sequential(nn.Linear(k, 32), nn.Tanh(),
                                 nn.Linear(32, 1))
        self.spk = nn.Sequential(nn.Linear(k, 32), nn.Tanh(),
                                 nn.Linear(32, 1))
        self.k = k

    def forward(self, i_seq, v_teach=None, eps=1.0, h=None,
                v_prev=None):
        """i_seq (B,T) normalized current. Autoregressive observable
        feedback: input_t = [I_t, v_{t-1}], where v_{t-1} is the
        teacher's voltage w.p. 1-eps, else the model's own detached
        prediction. eps=1 (or v_teach=None) = full free-run.
        Returns (v, spike_logit, h, last_v)."""
        B, T = i_seq.shape
        if h is None:
            h = i_seq.new_zeros(B, self.k)
        if v_prev is None:
            v_prev = i_seq.new_zeros(B, 1)          # rest = 0 norm
        hs, vs = [], []
        for t in range(T):
            if not self.fb:
                v_fb = v_prev * 0.0
            elif v_teach is not None and eps < 1.0:
                use_own = (torch.rand(B, 1, device=i_seq.device)
                           < eps).float()
                prev_t = (v_teach[:, t - 1:t] if t > 0 else v_prev)
                v_fb = use_own * v_prev + (1 - use_own) * prev_t
            else:
                v_fb = v_prev
            h = self.cell(
                torch.cat([i_seq[:, t:t + 1], v_fb], 1), h)
            v_prev = self.out(h).detach()
            hs.append(h)
            vs.append(v_prev)
        H = torch.stack(hs, 1)
        return (self.out(H).squeeze(-1), self.spk(H).squeeze(-1),
                h, vs[-1])


def run_full(model, I_mv, dev, bs=32):
    """I_mv (B,T) raw current -> predicted V in mV (numpy)."""
    model.eval()
    outs = []
    with torch.no_grad():
        for b0 in range(0, len(I_mv), bs):
            x = torch.tensor(I_mv[b0:b0 + bs] / IS,
                             dtype=torch.float32, device=dev)
            v, _, _, _ = model(x)
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


def train_one(k, d, dev, epochs, seed=0, use_wandb=False, fb=True):
    torch.manual_seed(seed)
    model = Surrogate(k, fb=fb).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=3e-4)
    Itr = torch.tensor(d['train_I'] / IS, dtype=torch.float32)
    Vtr = torch.tensor((d['train_V'] + VOFF) / VS,
                       dtype=torch.float32)
    W = torch.where(Vtr > 0.45, 10.0, 1.0)     # V > -20 mV
    S = torch.zeros_like(Vtr)                  # spike indicator
    for b in range(len(d['train_V'])):
        for t in spikes_from_v(d['train_V'][b]):
            i = int(t / 0.1)
            S[b, max(i - 3, 0):i + 4] = 1.0
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.0,
                                                       device=dev))
    nprm = sum(p.numel() for p in model.parameters())
    B, T = Itr.shape
    for ep in range(epochs):
        eps = min(1.0, ep / max(1, int(0.6 * epochs)))
        perm = torch.randperm(B)
        tot = cnt = 0.0
        for b0 in range(0, B, 32):
            idx = perm[b0:b0 + 32]
            h = vlast = None
            for c0 in range(0, T, CHUNK):
                x = Itr[idx, c0:c0 + CHUNK].to(dev)
                y = Vtr[idx, c0:c0 + CHUNK].to(dev)
                w = W[idx, c0:c0 + CHUNK].to(dev)
                sy = S[idx, c0:c0 + CHUNK].to(dev)
                v, sl, h, vlast = model(
                    x, v_teach=y, eps=eps,
                    h=None if h is None else h.detach(),
                    v_prev=vlast)
                loss = ((v - y) ** 2 * w).mean() + 0.5 * bce(sl, sy)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * x.numel()
                cnt += x.numel()
        sched.step()
        va = run_full(model, d['val_I'], dev)
        rmse = float(np.sqrt(np.mean((va - d['val_V']) ** 2)))
        vf1 = spike_f1(d['val_V'], va)
        print(f'k={k} ep{ep + 1} eps={eps:.2f}: train '
              f'{tot / cnt:.5f} val-RMSE {rmse:.2f} mV '
              f'F1 {vf1:.2f}', flush=True)
        if use_wandb:
            import wandb
            wandb.log({f'k{k}/train_loss': tot / cnt,
                       f'k{k}/val_rmse': rmse,
                       f'k{k}/val_f1': vf1, 'epoch': ep + 1})
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
    ap.add_argument('--ks', default='1,2,3,4,8')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--no-fb', action='store_true')
    args = ap.parse_args()
    dev = ('mps' if torch.backends.mps.is_available() else 'cpu')
    d = dict(np.load(OUT / 'hh_data.npz'))
    use_wandb = False
    try:
        import wandb
        wandb.init(project='m13-state',
                   name='rung1-ladder-v4' if args.no_fb else 'rung1-ladder-v3',
                   config=vars(args))
        use_wandb = True
    except Exception as e:
        print('wandb off:', e, flush=True)
    teacher_reb = len(spikes_from_v(
        d['rebound_v'][int(200.0 / (DT * REC_EVERY)):]))
    print(f'teacher rebound spikes: {teacher_reb}', flush=True)
    results = [train_one(int(k), d, dev, args.epochs,
                         use_wandb=use_wandb, fb=not args.no_fb)
               for k in args.ks.split(',')]
    json.dump(results, open(OUT / 'ladder_results.json', 'w'),
              indent=1)
    print('\n=== LADDER ===', flush=True)
    for r in results:
        print(r, flush=True)


if __name__ == '__main__':
    main()
