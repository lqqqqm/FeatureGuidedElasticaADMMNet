from __future__ import annotations

import torch

from fg_elastica_inpaint.losses import InpaintingLoss
from fg_elastica_inpaint.models import FeatureGuidedElasticaADMMNet



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    model = FeatureGuidedElasticaADMMNet(
        image_size=64,
        K=2,
        transformer_depth=1,
        enable_p_correction=False,
        enable_n_correction=False,
    ).to(device)
    criterion = InpaintingLoss(K=2, lambda_perc=0.0, use_perceptual=False).to(device)

    gt = torch.randn(2, 3, 64, 64, device=device).clamp(-1, 1)
    M = (torch.rand(2, 1, 64, 64, device=device) > 0.4).float()
    I_m = gt * M

    outputs = model(I_m, M)
    loss_dict = criterion(outputs, gt, M)
    loss = loss_dict["total"]
    loss.backward()

    print("Smoke test passed.")
    print("pred:", outputs["pred"].shape)
    print({k: float(v.detach() if hasattr(v, 'detach') else v) for k, v in loss_dict.items()})


if __name__ == "__main__":
    main()
