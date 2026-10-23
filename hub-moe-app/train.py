"""Train the MoE regressor on the Hugging Face Open AI Landscape dataset.

    python train.py --data data/models.csv --epochs 80
"""
import argparse, json, copy
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import train_test_split

from hubmoe.features import featurize, Encoder, NUM_COLS, CAT_COLS
from hubmoe.model import MoERegressor
from hubmoe.metrics import regression_report
from hubmoe.baseline import run_baselines


def to_t(*arrs):
    return [torch.from_numpy(a) for a in arrs]


@torch.no_grad()
def predict(model, xc, xn, bs=2048):
    model.eval(); ps, ws = [], []
    for i in range(0, len(xc), bs):
        p, _, w = model(xc[i:i + bs], xn[i:i + bs]); ps.append(p); ws.append(w)
    return torch.cat(ps).numpy(), torch.cat(ws).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/models.csv")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--experts", type=int, default=6)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--aux", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.out); out.mkdir(exist_ok=True)

    df = pd.read_csv(a.data)
    y = np.log10(df["downloads"].to_numpy(np.float64))
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.2, random_state=a.seed)
    va, te = train_test_split(tmp, test_size=0.5, random_state=a.seed)

    feats = featurize(df)
    enc = Encoder().fit(feats.iloc[tr])           # fit ONLY on train
    xc, xn = enc.transform(feats)
    y_mu, y_sd = y[tr].mean(), y[tr].std()
    yz = ((y - y_mu) / y_sd).astype(np.float32)

    # ---- baselines (honest reference point) ----
    base = run_baselines(xc[tr], xn[tr], y[tr], xc[te], xn[te], y[te])

    # ---- MoE ----
    XC, XN, Y = to_t(xc, xn, yz)
    model = MoERegressor(enc.cardinalities, len(NUM_COLS), n_experts=a.experts, top_k=a.top_k)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    loss_fn = nn.HuberLoss(delta=1.0)
    best, best_state, patience, bad = 1e9, None, 15, 0
    tr_t = torch.from_numpy(tr)
    for ep in range(a.epochs):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t))]
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            p, aux, _ = model(XC[b], XN[b])
            loss = loss_fn(p, Y[b]) + a.aux * aux
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        pv, _ = predict(model, XC[va], XN[va])
        vloss = float(np.sqrt(np.mean((pv - yz[va]) ** 2)))
        if vloss < best - 1e-4:
            best, best_state, bad = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if ep % 5 == 0:
            print(f"epoch {ep:3d}  val RMSE(z)={vloss:.4f}  best={best:.4f}")
        if bad >= patience:
            print("early stop at epoch", ep); break
    model.load_state_dict(best_state)

    pv, _ = predict(model, XC[va], XN[va]); pv = pv * y_sd + y_mu
    pt, wt = predict(model, XC[te], XN[te]); pt = pt * y_sd + y_mu
    moe = regression_report(y[te], pt)
    resid = y[va] - pv
    q05, q95 = np.quantile(resid, [0.05, 0.95])

    # ---- routing analysis: which expert handles which task? ----
    _, w_all = predict(model, XC, XN)
    r = pd.DataFrame({"pipeline_tag": feats["pipeline_tag"], "expert": w_all.argmax(1)})
    top_tags = r.pipeline_tag.value_counts().head(12).index
    routing = pd.crosstab(r[r.pipeline_tag.isin(top_tags)].pipeline_tag,
                          r[r.pipeline_tag.isin(top_tags)].expert, normalize="index").round(3)
    routing.to_csv(out / "routing_by_task.csv")

    torch.save(model.state_dict(), out / "moe.pt")
    meta = {"encoder": enc.to_dict(), "y_mu": float(y_mu), "y_sd": float(y_sd),
            "resid_q05": float(q05), "resid_q95": float(q95),
            "config": {"n_experts": a.experts, "top_k": a.top_k},
            "metrics": {"moe": moe, **base}, "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("\n=== TEST METRICS (log10 downloads) ===")
    print(pd.DataFrame(meta["metrics"]).T.round(3).to_string())
    print("\nExpert routing by task (rows sum to 1):\n", routing.to_string())


if __name__ == "__main__":
    main()
