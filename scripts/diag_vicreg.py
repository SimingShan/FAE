"""Diagnose why their VICReg loss is flat. Run ~60 steps of THEIR VICReg + THEIR loader, log the
loss COMPONENTS (repr/std/cov), the projector-output feature std (collapse indicator), and the
input view-difference (is the Lie augmentation producing two DIFFERENT views?). Faithful config."""
import os, sys, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external/SSLForPDEs"))
from navier_stokes import VICReg
from utils import get_loader_ns, off_diagonal

DEVICE = "cuda"
DATA = os.path.expanduser("~/scratch/ns_data")
BS, NF = 64, 512
STRENGTHS = [0.0, 1.0, 1.0, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01]   # tsl_t,tsl_x,tsl_y,scale,rot,lin_x,lin_y,quad_x,quad_y (paper defaults)


def main():
    lr = float(sys.argv[1]) if len(sys.argv) > 1 else 3e-4
    model = VICReg(sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, batch_size=BS, mlp="512-512-512", n_time_steps=16).to(DEVICE)
    loader = get_loader_ns(DATA, BS, steps=2, order=2, num_workers=4, strengths=STRENGTHS, mode="train", crop_size=(16, 128, 128), dataset_size=26624)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"VICReg diag  lr={lr}  bs={BS}  (faithful strengths)", flush=True)
    print(f"{'step':>4} {'loss':>7} {'25*repr':>8} {'25*std':>7} {'cov':>7} {'featstd':>8} {'viewdiff':>9}", flush=True)
    model.train()
    for i, (x, y, _) in enumerate(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        vd = (x - y).abs().mean().item()
        px = model.projector(model.backbone(x)); py = model.projector(model.backbone(y))
        repr_loss = F.mse_loss(px, py)
        pxc = px - px.mean(0); pyc = py - py.mean(0)
        sx = torch.sqrt(pxc.var(0) + 1e-4); sy = torch.sqrt(pyc.var(0) + 1e-4)
        std_loss = torch.mean(F.relu(1 - sx)) / 2 + torch.mean(F.relu(1 - sy)) / 2
        cov = (off_diagonal((pxc.T @ pxc) / (BS - 1)).pow(2).sum() + off_diagonal((pyc.T @ pyc) / (BS - 1)).pow(2).sum()) / NF
        loss = 25 * repr_loss + 25 * std_loss + 1 * cov
        opt.zero_grad(); loss.backward(); opt.step()
        if i % 5 == 0:
            print(f"{i:>4} {loss.item():>7.2f} {25*repr_loss.item():>8.3f} {25*std_loss.item():>7.3f} "
                  f"{cov.item():>7.3f} {sx.mean().item():>8.4f} {vd:>9.4f}", flush=True)
        if i >= 60:
            break


if __name__ == "__main__":
    main()
