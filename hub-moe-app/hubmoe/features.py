"""Feature engineering shared by training AND serving (prevents train/serve skew)."""
import re
import numpy as np
import pandas as pd

SNAPSHOT = pd.Timestamp("2026-10-11")      # date the dataset was fetched
LEGACY_DATE = "2022-03-02"                 # created_at is floored to this date in the dataset
OTHER = "__other__"
CAT_COLS = ["pipeline_tag", "library", "author"]
NUM_COLS = ["log_age_days", "is_legacy_date", "log_params_b", "has_params",
            "name_len", "is_gguf", "is_instruct", "is_quant"]

_PARAM_RE = re.compile(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s?([bm])(?![a-z])")


def _params_in_billions(name: str):
    m = _PARAM_RE.search(name.lower())
    if not m:
        return np.nan
    v = float(m.group(1))
    return v if m.group(2) == "b" else v / 1000.0


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    """df needs: model_id, pipeline_tag, library, author, created_at."""
    out = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        out[c] = df[c].fillna("unknown").astype(str)
    name = df["model_id"].astype(str).str.split("/").str[-1]
    lname = name.str.lower()
    created = pd.to_datetime(df["created_at"])
    age = (SNAPSHOT - created).dt.days.clip(lower=0)
    out["log_age_days"] = np.log1p(age)
    out["is_legacy_date"] = (df["created_at"].astype(str) == LEGACY_DATE).astype(float)
    p = name.map(_params_in_billions)
    out["has_params"] = p.notna().astype(float)
    out["log_params_b"] = np.log10(p.clip(lower=1e-3)).fillna(0.0)
    out["name_len"] = np.log1p(name.str.len())
    out["is_gguf"] = lname.str.contains("gguf").astype(float)
    out["is_instruct"] = lname.str.contains(r"instruct|chat|-it\b").astype(float)
    out["is_quant"] = lname.str.contains("awq|gptq|int4|int8|4bit|8bit|fp8|bnb|mlx").astype(float)
    return out


class Encoder:
    """Maps categoricals to ids (0 = rare/unseen) and standardises numerics."""

    def __init__(self, min_count=None):
        self.min_count = min_count or {"pipeline_tag": 3, "library": 5, "author": 8}
        self.vocab, self.mean, self.std = {}, None, None

    def fit(self, feats: pd.DataFrame):
        for c in CAT_COLS:
            vc = feats[c].value_counts()
            keep = vc[vc >= self.min_count[c]].index.tolist()
            self.vocab[c] = [OTHER] + [k for k in keep if k != OTHER]
        self.mean = feats[NUM_COLS].mean().to_dict()
        self.std = feats[NUM_COLS].std().replace(0, 1.0).fillna(1.0).to_dict()
        return self

    @property
    def cardinalities(self):
        return [len(self.vocab[c]) for c in CAT_COLS]

    def transform(self, feats: pd.DataFrame):
        cats = []
        for c in CAT_COLS:
            idx = {v: i for i, v in enumerate(self.vocab[c])}
            cats.append(feats[c].map(lambda v: idx.get(v, 0)).to_numpy(np.int64))
        xc = np.stack(cats, 1)
        xn = np.stack([(feats[c].to_numpy(float) - self.mean[c]) / self.std[c]
                       for c in NUM_COLS], 1).astype(np.float32)
        return xc, xn

    def to_dict(self):
        return {"vocab": self.vocab, "mean": self.mean, "std": self.std, "min_count": self.min_count}

    @classmethod
    def from_dict(cls, d):
        e = cls(d["min_count"]); e.vocab, e.mean, e.std = d["vocab"], d["mean"], d["std"]
        return e
