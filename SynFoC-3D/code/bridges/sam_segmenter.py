# -*- coding: utf-8 -*-
import torch.nn as nn


class SAM2DSegmenter(nn.Module):
    """Wrap the SAM model to expose a logits-style interface."""

    def __init__(self, sam_model: nn.Module, image_size: int, multimask_output: bool):
        super().__init__()
        self.sam = sam_model
        self.image_size = int(image_size)
        self.multimask_output = bool(multimask_output)

    def forward(self, x):
        outputs = self.sam(x, self.multimask_output, self.image_size)
        if isinstance(outputs, dict):
            return outputs["masks"]
        return outputs
