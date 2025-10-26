import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationalRescorerMLP(nn.Module):
    """Simple MLP-based rescorer for detections.

    Input: node_feats tensor (N, D) where D=6 by default -> [score, cx, cy, w, h, area]
    Output: rescore tensor (N,) with values in (0,1).
    """
    def __init__(self, node_feat_dim: int = 6, hidden: int = 64):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.hidden = hidden
        # Produce a single logit per node (no sigmoid here). Training script will
        # choose BCEWithLogitsLoss so that numerical stability and pos_weight can
        # be applied. For inference callers that expect probabilities, use
        # predict() which applies sigmoid.
        self.net = nn.Sequential(
            nn.Linear(node_feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, max(8, hidden//2)),
            nn.ReLU(),
            nn.Linear(max(8, hidden//2), 1)
        )

    def forward(self, node_feats: torch.Tensor, boxes: torch.Tensor = None, labels: torch.Tensor = None, scores: torch.Tensor = None):
        # node_feats: (N, D)
        out = self.net(node_feats)
        return out.squeeze(-1)

    def predict(self, node_feats: torch.Tensor, device='cpu'):
        self.eval()
        with torch.no_grad():
            nf = node_feats.to(device)
            # return probabilities
            return torch.sigmoid(self.forward(nf)).detach().cpu()


class GraphAttentionRescorer(nn.Module):
    """Light-weight GAT-like rescorer.

    - Uses a small node MLP to project node features to embeddings.
    - Computes pairwise attention weights via scaled dot-product of embeddings and aggregates neighbor features.
    - Produces a final sigmoid score per node.

    This is intentionally small to keep CPU/GPU cost low for inference/training in our experiments.
    """
    def __init__(self, node_feat_dim: int = 6, emb_dim: int = 64, num_heads: int = 2):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.emb_dim = emb_dim
        self.num_heads = max(1, num_heads)

        # Node projection
        self.proj = nn.Sequential(
            nn.Linear(node_feat_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        # Final scorer MLP
        # output a single logit per node; apply sigmoid during inference if needed
        self.scorer = nn.Sequential(
            nn.Linear(emb_dim + node_feat_dim, emb_dim//2),
            nn.ReLU(),
            nn.Linear(emb_dim//2, 1)
        )

    def forward(self, node_feats: torch.Tensor, boxes: torch.Tensor = None, labels: torch.Tensor = None, scores: torch.Tensor = None):
        # node_feats: (N, D)
        if node_feats is None or node_feats.shape[0] == 0:
            return node_feats.new_zeros((0,))

        N = node_feats.shape[0]
        device = node_feats.device

        h = self.proj(node_feats)  # (N, emb_dim)

        # compute attention matrix via scaled dot-product (N,N)
        scale = max(1.0, (self.emb_dim ** 0.5))
        attn_logits = torch.matmul(h, h.t()) / scale

        # Mask self-attention slightly (optional)
        attn_logits = attn_logits - torch.eye(N, device=device) * 1e9

        attn = torch.softmax(attn_logits, dim=1)  # rows sum to 1

        # aggregate neighbor embeddings
        agg = torch.matmul(attn, h)  # (N, emb_dim)

        # combine aggregated embedding with original node features
        combined = torch.cat([agg, node_feats], dim=1)
        out = self.scorer(combined).squeeze(-1)
        return out


def create_rescorer(model_type: str = 'mlp', **kwargs):
    """Factory: returns a rescorer instance by model_type ('mlp' or 'gat')."""
    model_type = (model_type or 'mlp').lower()
    if model_type == 'gat' or model_type == 'graph':
        return GraphAttentionRescorer(**kwargs)
    else:
        return RelationalRescorerMLP(**kwargs)


# Backwards-compatibility wrapper expected by older loader code
class RelationalRescorer(nn.Module):
    """Compatibility wrapper named `RelationalRescorer`.

    Older evaluation/training code imports `RelationalRescorer` and instantiates it
    with arguments like `node_feat_dim` and `hidden`. To remain compatible with
    both the MLP and the GAT checkpoints we expose a wrapper that contains
    both submodules (MLP -> `net`, GAT -> `proj`/`scorer`). When called, it will
    attempt to run the GAT path first and fall back to the MLP path if that
    fails. This allows tolerant checkpoint loading where the checkpoint may
    contain either MLP or GAT keys.
    """
    def __init__(self, node_feat_dim: int = 6, hidden: int = 64, emb_dim: int = 64, num_heads: int = 2):
        super().__init__()
        # Create both options. Checkpoint loading will populate whichever keys match.
        self.node_feat_dim = node_feat_dim

        # MLP option (kept under attribute name 'net' to match older checkpoints)
        # logits-producing MLP (no sigmoid)
        self.net = nn.Sequential(
            nn.Linear(node_feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, max(8, hidden//2)),
            nn.ReLU(),
            nn.Linear(max(8, hidden//2), 1)
        )

        # GAT-like option (attributes match GraphAttentionRescorer)
        self.proj = nn.Sequential(
            nn.Linear(node_feat_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        # scorer outputs a logit; sigmoid applied in predict()
        self.scorer = nn.Sequential(
            nn.Linear(emb_dim + node_feat_dim, emb_dim//2),
            nn.ReLU(),
            nn.Linear(emb_dim//2, 1)
        )

    def forward(self, node_feats: torch.Tensor, boxes: torch.Tensor = None, labels: torch.Tensor = None, scores: torch.Tensor = None):
        # Try GAT-style forward first, fallback to MLP if it fails
        if node_feats is None or node_feats.shape[0] == 0:
            return node_feats.new_zeros((0,))

        try:
            # attempt GAT path
            h = self.proj(node_feats)
            scale = max(1.0, (h.shape[1] ** 0.5))
            attn_logits = torch.matmul(h, h.t()) / scale
            N = node_feats.shape[0]
            device = node_feats.device
            attn_logits = attn_logits - torch.eye(N, device=device) * 1e9
            attn = torch.softmax(attn_logits, dim=1)
            agg = torch.matmul(attn, h)
            combined = torch.cat([agg, node_feats], dim=1)
            out = self.scorer(combined).squeeze(-1)
            return out
        except Exception:
            # fallback to MLP
            out = self.net(node_feats)
            return out.squeeze(-1)

    def predict(self, node_feats: torch.Tensor, device='cpu'):
        """Return probabilities (sigmoid applied) for callers that expect rescores in (0,1)."""
        self.eval()
        with torch.no_grad():
            nf = node_feats.to(device)
            logits = self.forward(nf)
            return torch.sigmoid(logits).detach().cpu()
