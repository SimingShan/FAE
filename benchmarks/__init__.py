"""SSL benchmark encoders (MAE, I-JEPA) + a factory shared by training and eval."""
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(method, resolution=224, in_chans=4, norm_pix=False,
                embed_dim=None, depth=None, patch_size=None, num_heads=None):
    """Build an MAE or I-JEPA encoder. resolution = int (square) or (H, W) (rectangular)."""
    vit = {k: v for k, v in (("embed_dim", embed_dim), ("depth", depth), ("patch_size", patch_size), ("num_heads", num_heads)) if v}
    if method == "mae":
        from benchmarks.mae.mae import mae_physics
        return mae_physics(img_size=resolution, in_chans=in_chans, norm_pix_loss=norm_pix, **vit).to(DEVICE)
    if method == "dino":
        from benchmarks.dino.dino2d import dino2d_physics
        return dino2d_physics(img_size=resolution, in_chans=in_chans, **vit).to(DEVICE)
    from benchmarks.jepa.ijepa2d import ijepa2d_physics
    return ijepa2d_physics(img_size=resolution, in_chans=in_chans, **vit).to(DEVICE)
