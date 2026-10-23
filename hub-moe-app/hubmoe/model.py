import torch
import torch.nn as nn
import torch.nn.functional as F

EMB_DIMS = [16, 16, 32]          # pipeline_tag, library, author


class MoERegressor(nn.Module):
    """Sparse Mixture-of-Experts regressor for tabular Hub metadata.

    embeddings + numerics -> shared trunk h
    gate(h) -> softmax over E experts -> keep top-k, renormalise
    h' = LayerNorm(h + sum_k w_k * Expert_k(h))     (residual MoE block)
    head(h') -> predicted log10(downloads)

    Auxiliary load-balancing loss (Switch-Transformer style) stops all
    samples collapsing onto one expert.
    """

    def __init__(self, cardinalities, n_num, d_model=64, n_experts=6, top_k=2,
                 hidden=128, dropout=0.1):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.embs = nn.ModuleList([nn.Embedding(c, e) for c, e in zip(cardinalities, EMB_DIMS)])
        self.trunk = nn.Sequential(nn.Linear(sum(EMB_DIMS) + n_num, d_model),
                                   nn.GELU(), nn.LayerNorm(d_model))
        self.gate = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(hidden, d_model))
            for _ in range(n_experts)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, xc, xn):
        e = [emb(xc[:, i]) for i, emb in enumerate(self.embs)]
        h = self.trunk(torch.cat(e + [xn], dim=1))                 # (B, d)
        probs = F.softmax(self.gate(h), dim=-1)                    # (B, E)
        _, topi = probs.topk(self.top_k, dim=-1)
        mask = torch.zeros_like(probs).scatter_(1, topi, 1.0)
        w = probs * mask
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)            # sparse routing weights
        outs = torch.stack([ex(h) for ex in self.experts], dim=1)  # (B, E, d)
        h = self.norm(h + (w.unsqueeze(-1) * outs).sum(1))
        # load-balancing aux loss: E * sum_e (fraction routed to e) * (mean prob of e)
        f = mask.mean(0) / self.top_k
        aux = self.n_experts * (f * probs.mean(0)).sum()
        return self.head(h).squeeze(-1), aux, w
