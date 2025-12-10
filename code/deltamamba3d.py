import torch, math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from mamba_ssm import Mamba
from .spatialmamba import StructureAwareSSM           # <- original 2‑D SSM
from .utils3d import StateFusion3D, to_channels_first, to_channels_last
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
# -------------------------------------------------------------------------
# 2‑>3‑D SSM (single scan still along flattened voxels, but keeping 3‑D conv)
# -------------------------------------------------------------------------



class StateFusion3D(nn.Module):
    def __init__(self, dim: int,
                 ks=((3,3,3), (3,5,5), (1,5,5)),
                 groupwise: bool = True):
        super().__init__()
        
        groups = dim if groupwise else 1
        convs = []
        for kt, kh, kw in ks:
            pad = (kt//2, kh//2, kw//2)
            convs.append(nn.Conv3d(dim, dim,
                                   kernel_size=(kt,kh,kw),
                                   padding=pad, dilation=1,
                                   groups=groups, bias=False))
        self.convs = nn.ModuleList(convs)
        self.alpha = nn.Parameter(torch.ones(len(convs)))

    def forward(self, h):  # (B,C,T,H,W)
        w = F.softmax(self.alpha, dim=0)
        out = 0
        for wi, conv in zip(w, self.convs):
            out = out + wi * conv(h)
        return out
'''  
class StateFusion3D(nn.Module):
    """
    Multi-scale depth-wise 3-D conv fusion without changing spatial size.
    ks = (1,3,5)   means:
        k=1 :    1×1×1   (no shrink)
        k=3 :    3×3×3
        k=5 :    3×3×3  with dilation=2  (effective 5×5×5)
    """
    def __init__(self, dim: int, ks=(1, 3, 5), groupwise=True):
        super().__init__()
        groups = dim if groupwise else 1
        convs  = []
        for k in ks:
            if k == 1:
                convs.append(
                    nn.Conv3d(dim, dim, kernel_size=1,
                              padding=0, dilation=1,
                              groups=groups, bias=False))
            else:
                d = k // 2                       #  k=3→1, k=5→2, ...
                convs.append(
                    nn.Conv3d(dim, dim, kernel_size=3,
                              padding=d, dilation=d,
                              groups=groups, bias=False))
        self.convs = nn.ModuleList(convs)
        self.alpha = nn.Parameter(torch.ones(len(convs)))

    def forward(self, h):                         # h: (B,D,T,H,W)
        w = F.softmax(self.alpha, dim=0)
        z = 0
        for wi, ki in zip(w, self.convs):
            z = z + wi * ki(h)                   # every kernel now size-safe
        return z
# -------------------------------------------------------------------------
'''
class StructureAwareSSM3D(nn.Module):
    r"""
    3-D extension of the 2-D Structure-Aware Spatial-Mamba block.

    Input / Output tensor shape (channels-first): **[B, C, T, H, W]**

    Pipeline:
        1) 1 × 1 × 1 point-wise Conv       (channel expansion  C → 2C)
        2) 3-D depth-wise Conv             (local context)
        3) Mamba SSM mixer                (unidirectional scan over T·H·W tokens)
        4) (optional) StateFusion3D        (multi-scale refinement in 3-D space)
        5) 1 × 1 × 1 projection back to C
    """

    def __init__(
        self,
        d_model:    int,
        d_state:    int  = 16,
        dt_rank:    str | int = "auto",
        dt_init:    str  = "random",
        dropout:    float = 0.0,
        act_layer        = nn.SiLU,
        fuse:       bool  = True,
        fuse_kwargs: dict | None = None,
    ):
        super().__init__()

        C_in   = d_model          # original channels
        C_exp  = 2 * C_in         # channels after expand/dw-conv

        # 1) C → 2C
        self.expand = nn.Conv3d(C_in, C_exp, kernel_size=1, bias=False)

        # 2) depth-wise 3 × 3 conv
        self.dwconv = nn.Conv3d(
            C_exp, C_exp,
            kernel_size=(1, 3, 3), padding=(0, 1, 1),
            groups=C_exp, bias=False)

        # ─────────── Mamba pieces ─────────── #
        if dt_rank == "auto":
            dt_rank = max(4, d_state // 2)
        self.dt_rank = dt_rank
        self.d_state = d_state
        self.drop_path = DropPath(0.1)
        # **FIX**: in_features must be C_exp, not C_in
        self.x_proj   = nn.Linear(C_exp,
                                  dt_rank + 2 * d_state,
                                  bias=True)
        self.dt_projs = nn.Linear(dt_rank, C_exp, bias=True)

        self.A_logs = nn.Parameter(torch.zeros(C_exp, d_state))
        self.Ds     = nn.Parameter(torch.ones(C_exp))
        
        self.C_proj = nn.Linear(d_state, C_exp, bias=False)
        self.state_fusion3D = (
            StateFusion3D(C_exp, **(fuse_kwargs or {}))
            if fuse else nn.Identity()
        )
        self.selective_scan = selective_scan_fn
        self.project = nn.Conv3d(C_exp, C_in, kernel_size=1, bias=False)

        self.act  = act_layer()
        self.drop = nn.Dropout3d(0.2)
        self.tau_min = 1                        # scalar > 0
        self.alpha   = nn.Parameter(torch.tensor(0.5))
        
    # ------------------------------------------------------------------ #
    #  Mamba SSM mix over flattened spatio-temporal tokens               #
    # ------------------------------------------------------------------ #
    def ssm(self, x: torch.Tensor, dt: torch.Tensor | None = None, gamma=1.0) -> torch.Tensor:
    # x : (B, C_exp, T, H, W)   -- C_exp = 2 * d_model (e.g. 1536)
        B, C_exp, T, H, W = x.shape
        L = T * H * W                      # sequence length

        # 1) Flatten spatio-temporal grid → sequence
        xs = x.view(B, C_exp, L).transpose(1, 2).contiguous()   # (B, L, C_exp)
        x_dbl = self.x_proj(xs)
        # 2) Point-wise projection: (Δt |  B  |  C)
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )    # (B, L, d_state) each
        dts = self.dt_projs(dts)
        # b) Δt comes from the caller (shape-check later)
        if dt is None:
            raise ValueError("Pass Δt tensor when using continuous-time Mamba")
        # 1) flatten dt to match L  (=T*H*W)
        dt_flat = dt.reshape(x.size(0), -1, 1)            # (B, L, 1)
        # 2) copy along dt_rank
        dt_rep  = dt_flat.repeat(1, 1, self.dt_rank)      # (B, L, dt_rank)
        # 3) linear projection into the channel space
        dtf     = self.dt_projs(dt_rep)  
        dtf = dtf * (1 + gamma * dts/12)
        # 3) Selective scan (kernel from mamba-ssm ≥ 1.1.x)
        h = self.selective_scan(
            xs.transpose(1, 2),           # u  : (B, C_exp, L)
            dtf.transpose(1, 2),          # dt : (B, C_exp, L)
            -torch.exp(self.A_logs),      # A  : (C_exp, d_state)
            Bs.transpose(1, 2),           # B  : (B, d_state, L)
            Cs.transpose(1, 2),           # C  : (B, d_state, L)  -- **must NOT be None**
            self.Ds,                      # D  : (C_exp)
            z=None,
            delta_bias=self.dt_projs.bias,
            delta_softplus=True,
            return_last_state=False,
        )                                 # (B, C_exp, L)

        # 4) Gate projection   (16 → 1536)
        C_gate = self.C_proj(Cs).transpose(1, 2)       # (B, C_exp, L)

        # 5) Restore 5-D layout
        h      = h     .reshape(B, C_exp, T, H, W)
        C_gate = C_gate.reshape(B, C_exp, T, H, W)

        # 6) Optional multi-scale fusion + residual
        #h = self.state_fusion3D(h)
        y = h * C_gate + x * self.Ds.view(1, -1, 1, 1, 1)
        return y                                         # (B, 2C, T, H, W)

    # ------------------------------------------------------------------ #
    #  Forward                                                           #
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, dt:torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        B, C = x.shape[:2]

        # ---- local conv + activation --------------------------------- #
        x = self.expand(x)
        if x.shape[-2] >= 3 and x.shape[-1] >= 3:
            x = self.dwconv(x)
        x = self.act(x)

        # Remember spatial dims for reshape later
        T_orig, H_orig, W_orig = x.shape[-3:]

        # ---- SSM mixer ---------------------------------------------- #
        x     = self.ssm(x, dt=dt)
        x = self.drop(x)
        x = self.drop_path(x)
        # ---- projection back to C ------------------------------------ #
        x = self.project(x)                              # (B, C, T, H, W)
        return x

class STMambaBlock3D(nn.Module):
    """
    Residual block = LN → 3-D Spatial-Temporal-Mamba → LN → MLP,
    with optional multi-scale state fusion inside the Mamba mixer.

    Expected tensor layout in/out:  (B, T, H, W, C)  (channels-last)
    """
    def __init__(self,
                 dim: int,
                 drop_path: float = 0.,
                 mlp_ratio: float = 4.,
                 d_state: int = 16,
                 dt_init: str = "random",
                 attn_drop: float = 0.,
                 *,
                 # ——— new flags ———
                 fuse: bool = True,
                 fuse_kwargs: dict | None = None,
                 **kw):
        """
        Parameters
        ----------
        dim          : channel dimension of the block
        drop_path    : stochastic-depth probability
        mlp_ratio    : hidden = mlp_ratio × dim
        d_state      : Mamba state size
        dt_init      : 'random' | 'constant'  (see mamba-ssm)
        attn_drop    : dropout inside Mamba
        fuse         : turn multi-scale StateFusion3D on / off
        fuse_kwargs  : dict forwarded to StateFusion3D
                       e.g. {'ks':(1,3,5),'groupwise':True}
        kw           : any other kwargs you want to pass to StructureAwareSSM3D
        """
        super().__init__()

        kw.pop("dropout", None)           # avoid duplicate key

        # —— branch A: 3-D Mamba mixer ————————————————
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.mamba = StructureAwareSSM3D(
            d_model     = dim,
            d_state     = d_state,
            dt_init     = dt_init,
            dropout     = attn_drop,
            fuse        = fuse,
            fuse_kwargs = fuse_kwargs,
            **kw)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # —— branch B: MLP —————————————————————
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden     = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

    # ------------------------------------------------------------------
    def forward(self, x, visits, padding_mask=None):       # x: (B,T,H,W,C)
        # Mixer branch
        
        B, T, H, W, C = x.shape          # remember x is channels-last here
        deltas = visits.view(B, 1, T, 1, 1).expand(-1, 1, -1, H, W)
        y = to_channels_first(x)                   # (B,C,T,H,W)
        y = self.mamba(y, dt = deltas) 
                                            # StructureAwareSSM3D
        y = to_channels_last(y)                    # (B,T,H,W,C)
        y = self.norm1(y)
        
        #print(x.shape, y.shape)
        x = x + self.drop_path(y)

        # Feed-forward branch
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
# -------------------------------------------------------------------------
# A very thin 3‑D backbone that mirrors the 2‑D Spatial‑Mamba hierarchy.
# You can stack arbitrarily many STMambaBlock3D's exactly as before.
# -------------------------------------------------------------------------
class SpatialTemporalMamba(nn.Module):
    """
    Input  : B × T × C × H × W   (channels‑first)  OR
             B × T × H × W × C   (channels‑last, set channels_last=True)
    Output : pooled feature vector (B, C_out)
    """
    def __init__(self, in_chans=3, embed_dim=96, depths=[2,2,6,2],
                 channels_last=False,  **kw):
        super().__init__()
        self.channels_last = channels_last
        # shallow stem on each frame separately
        self.stem = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=(1,4,4), stride=(1,2,2))
        self.blocks = nn.ModuleList()
        dpr = torch.linspace(0., 0.2, sum(depths)).tolist()
        cur = 0
        for stage, d in enumerate(depths):
            for i in range(d):
                self.blocks.append(
                    STMambaBlock3D(
                dim       = embed_dim,          # ← ALWAYS 768
                drop_path = dpr[cur + i],
                **kw
            ))
            cur += d
            '''
            if stage < len(depths)-1:
                self.blocks.append(
                    nn.Conv3d(embed_dim*(2**stage),
                              embed_dim*(2**(stage+1)),
                              kernel_size=(1,2,2), stride=(1,2,2)))
            '''
        #self.norm = nn.LayerNorm(embed_dim*(2**(len(depths)-1)))
        self.norm = nn.LayerNorm(embed_dim) 
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self, x, padding_mask=None):
        if self.channels_last:                    # (B,T,H,W,C) -> (B,C,T,H,W)
            x = to_channels_first(x)
        #print(x.shape)
        x = self.stem(x)                          # (B,C,T,H',W')
        # bring to channels‑last for LayerNorm‑friendly shape (B,T,H,W,C)
        #print(x.shape)
        x = to_channels_last(x)
        #print(x.shape)
        for blk in self.blocks:
            if isinstance(blk, STMambaBlock3D):
                x = blk(x, padding_mask)
            else:                                 # downsample conv3d
                if self.channels_last:
                    x = to_channels_first(x)
                    x = blk(x)
                    x = to_channels_last(x)
                else:
                    x = blk(x)
        if self.channels_last:
            x = to_channels_first(x)
        x = self.pool(x)                          # B,C,1,1,1
        x = x.flatten(1)
        return self.norm(x)
