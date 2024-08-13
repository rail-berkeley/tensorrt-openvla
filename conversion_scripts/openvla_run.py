import argparse
import json
import os
import sys
from io import BytesIO
from pathlib import Path
import time

import requests

# isort: off
import torch
import numpy as np
import tensorrt as trt
# isort: on

from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors import safe_open
from torchvision import transforms
from transformers import (AutoConfig, AutoProcessor, AutoTokenizer, AutoModelForVision2Seq)

from transformers.dynamic_module_utils import get_class_from_dynamic_module

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm import logger
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.runtime import ModelRunner, Session, TensorInfo, StoppingCriteria

import copy
from io import BytesIO
from PIL import Image
import requests
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--save-dir', default="./save_dir", type=str)
parser.add_argument('--ckpts-dir', default="./ckpts", type=str)

args = parser.parse_args()



device = "cuda"
engine_dir = os.path.join(args.ckpts_dir, "openvla_engine")
save_dir = args.save_dir

print("Loading compiled LLM")
runtime_rank = tensorrt_llm.mpi_rank()
runner_kwargs = dict(
        engine_dir=engine_dir,
        rank=runtime_rank,
    )
runner = ModelRunner.from_dir(**runner_kwargs)
print("Done!")

print("Loading tokenizer...")
path_to_converted_ckpt = "Embodied-CoT/ecot-openvla-7b-bridge"
processor = AutoProcessor.from_pretrained(path_to_converted_ckpt, trust_remote_code=True)
tokenizer = processor.tokenizer
pad_id = tokenizer.pad_token_id
end_id = tokenizer.eos_token_id
print("Done!")

print("Loading vision backbone...")
pretrained_model_name_or_path = "Embodied-CoT/ecot-openvla-7b-bridge"
config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True
            )

vision_backbone_class_ref = config.auto_map[AutoModelForVision2Seq.__name__].replace("OpenVLAForActionPrediction", "PrismaticVisionBackbone")
vision_backbone_class = get_class_from_dynamic_module(
    vision_backbone_class_ref, pretrained_model_name_or_path
)
vision_backbone = vision_backbone_class(
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
        )
vision_backbone.load_state_dict(torch.load(os.path.join(save_dir, "vision_backbone.pth"), map_location=device))
vision_backbone.featurizer.to(device, torch.bfloat16)
print("Successful!")

# Get class for projector using HF utils, then load
print("Loading projector...")
proj_class_ref = config.auto_map[AutoModelForVision2Seq.__name__].replace("OpenVLAForActionPrediction", "PrismaticProjector")
proj_class = get_class_from_dynamic_module(
    proj_class_ref, pretrained_model_name_or_path
)
projector = proj_class(
    config.use_fused_vision_backbone,
    vision_dim=vision_backbone.embed_dim,
    llm_dim=config.text_config.hidden_size,
)
projector.load_state_dict(torch.load(os.path.join(save_dir, "projector.pth"), map_location=device))
projector.to(device, torch.bfloat16)
print("Successful!")

print("Getting image...")
url = 'https://raw.githubusercontent.com/MichalZawalski/embodied-CoT/main/test_obs.png'
page = requests.get(url)
image = Image.open(BytesIO(page.content))
print("Done!")

def parse_input(tokenizer,
                input_text,
                prompt_template=None,
                add_special_tokens=True,
                max_input_length=512,
                pad_id=None):
    if pad_id is None:
        pad_id = tokenizer.pad_token_id

    batch_input_ids = []
    for curr_text in input_text:
        if prompt_template is not None:
            curr_text = prompt_template.format(input_text=curr_text)
        input_ids = tokenizer.encode(curr_text,
                                        add_special_tokens=add_special_tokens,
                                        truncation=True,
                                        max_length=max_input_length)
        batch_input_ids.append(input_ids)
    
    base_vocab_size = 32064
    for i in range(len(batch_input_ids)):
        batch_input_ids[i] = batch_input_ids[i][:1] + \
            list(range(base_vocab_size, base_vocab_size + 256)) + \
            batch_input_ids[i][1:]

    batch_input_ids = [
        torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
    ]
    return batch_input_ids

print("Running inference...")
start = time.time()

prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: What action should the robot take to place the watermelon on the towel? ASSISTANT: TASK:"
batch_input_ids = parse_input(tokenizer, [prompt])

with torch.no_grad():
    pixel_values = processor("", image)["pixel_values"].to(device, dtype=torch.bfloat16)
    patch_features = vision_backbone(pixel_values)
    projected_patch_embeddings = projector(patch_features)

    outputs = runner.generate(
        batch_input_ids=batch_input_ids,
        encoder_input_ids=None,
        max_new_tokens=400,
        end_id=end_id,
        pad_id=pad_id,
        stop_words_list=[[[1, 2,]],],
        early_stopping=1,
        do_sample=False,
        random_seed=0,
        prompt_table=projected_patch_embeddings,
        prompt_vocab_size=256,
        output_sequence_lengths=True,
        return_dict=True,)
        #return_all_generated_tokens=True)
    torch.cuda.synchronize()

print(f"Done! Time elapsed: {time.time() - start}.")

output_ids = outputs['output_ids']
output_text = tokenizer.decode(output_ids.reshape(-1))

print(output_text)
print(output_ids)