"""Single-frame 2D DINO baseline (self-distillation with no labels).

ORIGINAL REPOS: facebookresearch/dino (Caron et al., ICCV 2021) and the newer
facebookresearch/dinov3 (`external/dinov3`). This is a faithful from-scratch RE-IMPLEMENTATION,
single-GPU (distributed code stripped), with the backbone PARAM-MATCHED to the campaign ViT
(embed_dim 256 / depth 6 / heads 8 / patch 8, ~the same ViT-Tiny used by MAE & I-JEPA, Hard-Rule #4).

Pieces reproduced faithfully:
  - DINOHead: 3-layer MLP -> L2-normalize bottleneck -> prototype layer (ported verbatim from
    external/dinov3/dinov3/layers/dino_head.py).
  - DINOLoss: teacher centering + sharpening, cross-entropy student||teacher over crop pairs
    (ported from external/dinov3/dinov3/loss/dino_clstoken_loss.py; all_reduce removed).
  - Teacher = EMA of student (backbone + head); teacher sees the 2 global crops, student all crops.

DELIBERATE physics adaptation: multi-crop is GEOMETRIC ONLY — random-resized sub-windows of the
field rescaled to the model resolution (per-sample affine grid_sample). NO photometric jitter / flips
(they would corrupt physical field semantics: sign of velocity components, vorticity handedness). All
crops are rendered at the model resolution (single ViT with fixed pos-embed), so the global/local
distinction is one of crop AREA (scale), not token count.

Downstream probe representation: teacher backbone patch tokens, mean-pooled (`encode`) — identical
contract to mae/ijepa so the frozen-probe harness treats all three the same.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from benchmarks.jepa.ijepa2d import ViT2D                      # shared param-matched backbone


# ----- DINOHead (ported verbatim from external/dinov3/dinov3/layers/dino_head.py) -----
def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    layers = [nn.Linear(in_dim, hidden_dim, bias=bias), nn.GELU()]
    for _ in range(nlayers - 2):
        layers += [nn.Linear(hidden_dim, hidden_dim, bias=bias), nn.GELU()]
    layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
    return nn.Sequential(*layers)


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = _build_mlp(max(nlayers, 1), in_dim, bottleneck_dim, hidden_dim=hidden_dim)
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = F.normalize(x, dim=-1, p=2, eps=eps)
        return self.last_layer(x)


# ----- DINOLoss (ported from external/dinov3/.../dino_clstoken_loss.py; distributed stripped) -----
class DINOLoss(nn.Module):
    def __init__(self, out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def update_center(self, teacher_output):                   # teacher_output: (T*B, K)
        c = teacher_output.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(c * (1 - self.center_momentum))

    def forward(self, student_logits, teacher_probs):
        """student_logits: (S,B,K) raw; teacher_probs: (T,B,K) softmaxed. CE over all student-crop x
        teacher-crop pairs EXCLUDING the same-crop diagonal — student global-crop i vs teacher crop i is
        the identical view (trivial). Faithful to DINO (Caron et al.) / dinov3 `ignore_diagonal`."""
        S, B, K = student_logits.shape
        T = teacher_probs.shape[0]
        student = F.log_softmax(student_logits.float() / self.student_temp, dim=-1)
        ce = -torch.einsum("s b k, t b k -> s t", student, teacher_probs)        # (S,T) summed over B,K
        m = min(S, T)
        ce = torch.diagonal_scatter(ce, ce.new_zeros(m))                          # drop trivial same-crop pairs
        return ce.sum() / (B * (S * T - m))


# ----- geometric multi-crop (per-sample affine, no photometric augs) -----
def _affine_crops(x, n, smin, smax, armin=0.75, armax=1.333):
    """n independent random-resized sub-windows of x (B,C,H,W), each rendered back to (H,W)."""
    B, C, H, W = x.shape
    out = []
    for _ in range(n):
        a = smin + (smax - smin) * torch.rand(B, device=x.device)               # area scale per sample
        r = torch.exp(math.log(armin) + (math.log(armax) - math.log(armin)) * torch.rand(B, device=x.device))
        fw = (a * r).sqrt().clamp(max=1.0); fh = (a / r).sqrt().clamp(max=1.0)   # box fractions
        cx = (2 * torch.rand(B, device=x.device) - 1) * (1 - fw)                 # center in [-(1-fw),(1-fw)]
        cy = (2 * torch.rand(B, device=x.device) - 1) * (1 - fh)
        theta = torch.zeros(B, 2, 3, device=x.device)
        theta[:, 0, 0] = fw; theta[:, 0, 2] = cx
        theta[:, 1, 1] = fh; theta[:, 1, 2] = cy
        grid = F.affine_grid(theta, (B, C, H, W), align_corners=False)
        out.append(F.grid_sample(x, grid, align_corners=False, padding_mode="reflection"))
    return out


def multicrop(x, n_global=2, n_local=4, global_scale=(0.4, 1.0), local_scale=(0.05, 0.4)):
    g = _affine_crops(x, n_global, *global_scale)
    l = _affine_crops(x, n_local, *local_scale)
    return g, l                                                # (globals, locals)


class DINO2D(nn.Module):
    """Student + EMA teacher ViT2D backbones, each with a DINOHead; owns the DINOLoss (center buffer)."""
    def __init__(self, img_size=128, patch_size=8, in_chans=4, embed_dim=256, depth=6, num_heads=8,
                 out_dim=4096, hidden_dim=2048, bottleneck_dim=256,
                 n_global=2, n_local=4, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student = ViT2D(img_size, patch_size, in_chans, embed_dim, depth, num_heads)
        self.teacher = ViT2D(img_size, patch_size, in_chans, embed_dim, depth, num_heads)
        self.student_head = DINOHead(embed_dim, out_dim, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)
        self.teacher_head = DINOHead(embed_dim, out_dim, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)
        self.teacher.load_state_dict(self.student.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for p in list(self.teacher.parameters()) + list(self.teacher_head.parameters()):
            p.requires_grad_(False)
        self.dino_loss = DINOLoss(out_dim, student_temp, center_momentum)
        self.n_global, self.n_local = n_global, n_local
        self.embed_dim = embed_dim
        self.teacher_temp = 0.04                                # set per-epoch by the trainer

    @torch.no_grad()
    def update_target(self, tau):                              # EMA student -> teacher (backbone + head)
        for pe, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.mul_(tau).add_(pe.data, alpha=1 - tau)
        for pe, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            pt.mul_(tau).add_(pe.data, alpha=1 - tau)

    def _head(self, net, head, crops):                         # list of (B,C,H,W) -> (n,B,K)
        z = [head(net(c).mean(dim=1)) for c in crops]          # mean-pool patch tokens -> head
        return torch.stack(z, 0)

    def dino_step(self, x):
        """One DINO loss on a batch (does centering update). Returns scalar loss."""
        globals_, locals_ = multicrop(x, self.n_global, self.n_local)
        with torch.no_grad():
            t_logits = self._head(self.teacher, self.teacher_head, globals_)        # (G,B,K) teacher: globals only
            t_probs = self.dino_loss.softmax_center_teacher(t_logits, self.teacher_temp)
        s_logits = self._head(self.student, self.student_head, globals_ + locals_)  # (G+L,B,K)
        loss = self.dino_loss(s_logits, t_probs)
        self.dino_loss.update_center(t_logits.reshape(-1, t_logits.size(-1)))
        return loss

    @torch.no_grad()
    def encode(self, imgs):
        """Frozen probe representation: teacher backbone patch tokens, mean-pooled."""
        return self.teacher(imgs).mean(dim=1)                  # (B, embed_dim)


def dino2d_physics(img_size=128, in_chans=4, patch_size=8, embed_dim=256, depth=6, num_heads=8,
                   out_dim=4096, n_local=4, **kw):
    """Param-matched (~ViT-Tiny encoder) single-frame 2D DINO. Backbone dims mirror the campaign
    MAE/I-JEPA so the three encoders are the same ViT (the head/teacher are extra, not counted in
    the encoder budget)."""
    return DINO2D(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                  depth=depth, num_heads=num_heads, out_dim=out_dim, n_local=n_local, **kw)
