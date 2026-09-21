import torch
import torch.nn as nn

class RerankLinear(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, pair_feats: torch.Tensor) -> torch.Tensor:
        return self.fc(pair_feats).squeeze(-1)

class RerankMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 512, dropout: float = 0.1, num_layers: int = 1):
        super().__init__()
        layers = []
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        for _ in range(max(0, num_layers - 1)):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._dropout = dropout

    def forward(self, pair_feats: torch.Tensor) -> torch.Tensor:
        return self.net(pair_feats).squeeze(-1)

class RerankInteractionMLP(nn.Module):
    def __init__(self, t_dim: int, i_dim: int, proj_dim: int = 256, hidden_dim: int = 512, dropout: float = 0.1, use_bilinear: bool = True):
        super().__init__()
        self.t_dim = t_dim
        self.i_dim = i_dim
        self.t_proj = nn.Linear(t_dim, proj_dim)
        self.i_proj = nn.Linear(i_dim, proj_dim)
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.Sigmoid(),
        )
        self.use_bilinear = use_bilinear
        if use_bilinear:
            self.bilinear = nn.Linear(proj_dim * proj_dim, proj_dim)
        feat_dim = proj_dim * 6 + 2
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pair_feats: torch.Tensor) -> torch.Tensor:
        t = pair_feats[:, :self.t_dim]
        i = pair_feats[:, self.t_dim:self.t_dim + self.i_dim]
        tp = self.t_proj(t)
        ip = self.i_proj(i)
        gi = torch.cat([tp, ip], dim=-1)
        g = self.gate(gi)
        diff = tp - ip
        adiff = torch.abs(diff)
        prod = tp * ip
        dot = (tp * ip).sum(dim=-1, keepdim=True)
        cos = torch.nn.functional.cosine_similarity(tp, ip, dim=-1).unsqueeze(-1)
        if self.use_bilinear:
            bi = torch.bmm(tp.unsqueeze(2), ip.unsqueeze(1)).view(tp.size(0), -1)
            bi = self.bilinear(bi)
        else:
            bi = torch.zeros_like(tp)
        inter = torch.cat([tp, ip, diff, adiff, prod, bi, dot, cos], dim=-1)
        inter = inter * torch.cat([g, g, g, g, g, g, torch.ones_like(dot), torch.ones_like(cos)], dim=-1)
        return self.net(inter).squeeze(-1)
