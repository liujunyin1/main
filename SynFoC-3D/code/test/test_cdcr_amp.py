import os
import sys

import pytest

torch = pytest.importorskip("torch")
from torch.cuda.amp import autocast


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from regularizers.cdcr_3d import cdcr_loss_3d  # noqa: E402


def test_cdcr_loss_amp_extreme_logits_stays_finite():
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)

    base = torch.tensor(
        [
            [
                [[1000.0, -1000.0], [1200.0, -1200.0]],
                [[-1500.0, 1500.0], [-1300.0, 1300.0]],
            ],
            [
                [[-900.0, 900.0], [-1100.0, 1100.0]],
                [[800.0, -800.0], [950.0, -950.0]],
            ],
        ],
        dtype=torch.float32,
        device=device,
    )
    sam_logits = base.unsqueeze(0)
    vnet_logits = (-base).unsqueeze(0)

    autocast_kwargs = {"enabled": True}
    if device_type != "cuda":
        autocast_kwargs["device_type"] = device_type

    with autocast(**autocast_kwargs):
        sam_probs = torch.softmax(sam_logits, dim=1)
        vnet_probs = torch.softmax(vnet_logits, dim=1)
        loss = cdcr_loss_3d(sam_probs, vnet_probs)

    assert torch.isfinite(loss).item(), "CDCR loss should remain finite under autocast with extreme logits"