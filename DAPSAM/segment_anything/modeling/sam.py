# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .memory.memory_prompt import PrototypePromptGenerate
from .prompt_encoder import PromptEncoder
from .cls_head import SAMAdvancedClsHead

class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        # prompt_encoder: PromptEncoder,
        cls_num_classes: int,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        # self.prompt_encoder = prompt_encoder
        self.prompt_generator = PrototypePromptGenerate()
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        # <--- [新增] 分类头 (仿照 unet.py 的结构) --->
        # SAM 默认的图像特征通道数是 256
        self.cls_decoder = SAMAdvancedClsHead(
            in_channels=256,
            hidden_dim=256,  # 可以根据需要调大，比如 512
            num_classes=cls_num_classes
        )
        # ---------------------------------------------

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(self, batched_input, multimask_output, image_size, need_fp=False):
        if isinstance(batched_input, list):
            outputs = self.forward_test(batched_input, multimask_output)
        else:
            outputs = self.forward_train(batched_input, multimask_output, image_size, need_fp)
        return outputs

    
    def forward_train(self, batched_input, multimask_output, image_size, need_fp=False):
        input_images = self.preprocess(batched_input)
        image_embeddings = self.image_encoder(input_images)

        # 1. Feature Perturbation (FP)
        if need_fp:
            # 仿照 unet 的逻辑，在 batch 维度拼接：原始特征 和 Dropout 扰动后的特征
            image_embeddings = torch.cat((image_embeddings, nn.Dropout2d(0.5)(image_embeddings)), dim=0)

        # 2. 生成 prompt (会根据当前 image_embeddings 的 batch_size 自动生成对应的 prompt)
        sparse_embeddings, prompt = self.prompt_generator(image_embeddings)

        # 获取密集位置编码，确保 batch_size 对齐
        image_pe = self.prompt_generator.get_dense_pe()
        if image_pe.size(0) != image_embeddings.size(0):
            image_pe = image_pe.expand(image_embeddings.size(0), -1, -1, -1)

        # 3. 分割解码器
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=prompt,
            multimask_output=multimask_output
        )

        # 4. 恢复原图尺寸的 mask
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )

        # 5. 分类解码器
        # 自适应池化 image_embeddings [B, 256, H, W] -> [B, 256]
        # image_embedding_pool = F.adaptive_avg_pool2d(image_embeddings, 1).view(image_embeddings.size(0), -1)
        # cls_logits = self.cls_decoder(image_embedding_pool)
        cls_logits = self.cls_decoder(image_embeddings)
        # 6. 整理输出格式，完美兼容你的代码逻辑
        if need_fp:
            # 将在 batch 维度翻倍的特征 chunk(2) 切分为 (preds, preds_fp)
            low_res_masks, low_res_masks_fp = low_res_masks.chunk(2)
            cls_logits, cls_logits_fp = cls_logits.chunk(2)
            # (可选) 如果你后续需要原尺寸 mask 或 iou，也可以 chunk 它们
            masks, masks_fp = masks.chunk(2)
            iou_predictions, iou_predictions_fp = iou_predictions.chunk(2)

            outputs = {
                'masks': (masks, masks_fp),
                'iou_predictions': (iou_predictions, iou_predictions_fp),
                'low_res_logits': (masks, masks_fp),
                'cls_logits': (cls_logits, cls_logits_fp)
            }
        else:
            outputs = {
                'masks': masks,
                'iou_predictions': iou_predictions,
                'low_res_logits': masks,
                'cls_logits': cls_logits
            }

        return outputs

    @torch.no_grad()
    def forward_test(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

