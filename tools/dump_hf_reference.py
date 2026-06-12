#!/usr/bin/env python3
"""Dump HF/mlx-vlm reference artifacts for the Swift parity gate.

Per case: templated+expanded input_ids (HF AutoProcessor), position_ids
(HF get_rope_index), pixel_values (HF processor), ViT merged features
(mlx-vlm VisionModel, fp32, CPU stream — deterministic), and mlx-vlm's
greedy answer text for informal comparison.

Run (uses the lance-mlx venv for transformers/mlx_vlm):
    cd /Volumes/DEV_ARCHIVE/lance-mlx && HF_HUB_DISABLE_XET=1 uv run python \
        ~/Development/MLXEngine/qwen25vl-mlx-swift/tools/dump_hf_reference.py --case 02
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from PIL import Image

IMAGES = Path("/Volumes/DEV_ARCHIVE/lance-mlx/tests/fixtures/images")
WEIGHTS = Path("/Volumes/DEV_VOL1/VideoResearch/qwen25vl-mlx-models/Qwen2.5-VL-3B-Instruct-bf16")
OUT = Path("/Volumes/DEV_VOL1/VideoResearch/qwen25vl-mlx-models/parity")

QUESTIONS = {
    "01": "Is the largest segment greater than sum of all the other segments?",
    "02": "What percentage of respondents want better border security?",
    "03": "What is the license plate number of the car?",
    "04": "According to the data from the proprietary market research, how much amount was spent on the promotional meetings and events during 1998?",
    "05": "What is the appearance of the Colosseum in Rome, Italy?",
    "06": "How does a total solar eclipse look like from Earth?",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="02")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    mx.set_default_device(mx.cpu)  # deterministic op-parity reference

    image = Image.open(IMAGES / f"image-understanding-case-{args.case}.png").convert("RGB")
    question = QUESTIONS[args.case]

    # --- HF processor: template + ids + pixels --------------------------------
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": question}],
    }]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(images=image, text=text, return_tensors="np")
    input_ids = np.asarray(inputs["input_ids"][0]).astype(np.int32)
    pixel_values = np.asarray(inputs["pixel_values"]).astype(np.float32)
    grid = np.asarray(inputs["image_grid_thw"]).astype(np.int32)
    print(f"template:\n{text!r}")
    print(f"ids: {len(input_ids)} tokens, grid {grid.tolist()}, pixels {pixel_values.shape}")

    # --- HF get_rope_index -----------------------------------------------------
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
    cfg = json.loads((WEIGHTS / "config.json").read_text())
    hf_cfg = Qwen2_5_VLConfig(**{k: v for k, v in cfg.items()
                                 if k not in ("architectures", "torch_dtype",
                                              "transformers_version", "quantization")})
    shim = Qwen2_5_VLModel.__new__(Qwen2_5_VLModel)
    shim.config = hf_cfg
    tids = torch.from_numpy(input_ids.astype(np.int64))[None]
    image_pad = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    mm_types = (tids == image_pad).int() * 1  # 1 = image
    pos, _ = Qwen2_5_VLModel.get_rope_index(
        shim, input_ids=tids, mm_token_type_ids=mm_types,
        image_grid_thw=torch.from_numpy(grid.astype(np.int64)),
        video_grid_thw=None, second_per_grid_ts=None,
        attention_mask=torch.ones_like(tids),
    )
    position_ids = pos.numpy().astype(np.int32)  # (3, 1, T)

    # --- mlx-vlm ViT features, fp32 CPU ----------------------------------------
    import inspect
    from mlx_vlm.models.qwen2_5_vl.config import VisionConfig
    from mlx_vlm.models.qwen2_5_vl.vision import VisionModel
    fields = set(inspect.signature(VisionConfig).parameters)
    vc = dict(cfg["vision_config"])
    if "in_chans" in vc:
        vc["in_channels"] = vc.pop("in_chans")
    vision = VisionModel(VisionConfig(**{k: v for k, v in vc.items() if k in fields}))
    vit_weights = {}
    for f in sorted(WEIGHTS.glob("*.safetensors")):
        for k, v in mx.load(str(f)).items():
            if k.startswith("vision_tower."):
                vit_weights[k.removeprefix("vision_tower.")] = v
    vit_weights = vision.sanitize(vit_weights)
    vit_weights = {k: v.astype(mx.float32) for k, v in vit_weights.items()}
    vision.load_weights(list(vit_weights.items()))
    mx.eval(vision.parameters())
    feats = vision(mx.array(pixel_values), mx.array(grid))
    mx.eval(feats)
    print(f"vit features: {feats.shape} (fp32, cpu)")

    out = OUT / f"case{args.case}.safetensors"
    mx.save_safetensors(str(out), {
        "input_ids": mx.array(input_ids),
        "position_ids": mx.array(position_ids),
        "pixel_values": mx.array(pixel_values),
        "image_grid_thw": mx.array(grid),
        "vit_features_fp32_cpu": feats.astype(mx.float32),
    })
    (OUT / f"case{args.case}.template.txt").write_text(text)
    print(f"saved → {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
