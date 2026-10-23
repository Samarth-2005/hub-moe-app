import json
from pathlib import Path
import numpy as np, pandas as pd, torch
from .features import featurize, Encoder, NUM_COLS
from .model import MoERegressor


class Predictor:
    def __init__(self, artifact_dir="artifacts"):
        d = Path(artifact_dir)
        self.meta = json.loads((d / "meta.json").read_text())
        self.enc = Encoder.from_dict(self.meta["encoder"])
        cfg = self.meta["config"]
        self.model = MoERegressor(self.enc.cardinalities, len(NUM_COLS),
                                  n_experts=cfg["n_experts"], top_k=cfg["top_k"])
        self.model.load_state_dict(torch.load(d / "moe.pt", map_location="cpu"))
        self.model.eval()

    @property
    def vocab(self):
        return self.enc.vocab

    def predict(self, pipeline_tag, library, author, name, created_at):
        row = pd.DataFrame([{"model_id": f"{author}/{name}", "pipeline_tag": pipeline_tag,
                             "library": library, "author": author,
                             "created_at": str(created_at)}])
        xc, xn = self.enc.transform(featurize(row))
        with torch.no_grad():
            p, _, w = self.model(torch.from_numpy(xc), torch.from_numpy(xn))
        y = float(p[0]) * self.meta["y_sd"] + self.meta["y_mu"]
        lo, hi = y + self.meta["resid_q05"], y + self.meta["resid_q95"]
        return {"log10": y, "downloads": 10 ** y, "low": 10 ** lo, "high": 10 ** hi,
                "expert_weights": w[0].numpy()}
