import torch, math
import torch.nn.functional as F
from einops import rearrange, repeat

# -----------------------------------------------------------------------------
# (a)    3‑D State‑Fusion  ——>   feature‑wise dilated 3‑D depthwise conv
# -----------------------------------------------------------------------------
class StateFusion3D(torch.nn.Module):
    """
    Structure‑aware fusion for a 3‑D tensor h (B, D, T, H, W).
    Runs three dilated 3‑D DW‑conv kernels (1×,3×,5×) and learns a
    convex combination   α₀·k₁ + α₁·k₃ + α₂·k₅   (α≥0, Σα=1).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.k1 = torch.nn.Conv3d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.k3 = torch.nn.Conv3d(dim, dim, 3, padding=3, dilation=3, groups=dim, bias=False)
        self.k5 = torch.nn.Conv3d(dim, dim, 3, padding=5, dilation=5, groups=dim, bias=False)
        self.alpha = torch.nn.Parameter(torch.ones(3))        # learnable weights

    def forward(self, h):                                    # h: (B,D,T,H,W)
        w = F.softmax(self.alpha, dim=0)
        z = 0
        for wi, ki in zip(w, self.convs):
            # true RF  = dilation*(k−1)+1  (Conv3d formula)
            rf = [(k-1)*d + 1 for k,d in zip(ki.kernel_size, ki.dilation)]
            if all(s >= r for s, r in zip(h.shape[-3:], rf)):     # fits?
                z = z + wi * ki(h)
        return z

# -----------------------------------------------------------------------------
# (b)    helper that converts (B,T,H,W,C)  <‑‑>  (B,C,T,H,W)
# -----------------------------------------------------------------------------
def to_channels_first(x):   # (B,T,H,W,C) -> (B,C,T,H,W)
    return rearrange(x, 'b t h w c -> b c t h w')

def to_channels_last(x):    # (B,C,T,H,W) -> (B,T,H,W,C)
    return rearrange(x, 'b c t h w -> b t h w c')
