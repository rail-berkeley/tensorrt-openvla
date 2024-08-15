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
from PIL import Image, ImageDraw, ImageFont
import requests
import argparse
import textwrap
import enum

from model import TRTOpenVLA

parser = argparse.ArgumentParser()

parser.add_argument('--save-dir', default="./save_dir", type=str)
parser.add_argument('--ckpts-dir', default="./ckpts", type=str)

args = parser.parse_args()

device = "cuda"
engine_dir = os.path.join(args.ckpts_dir, "openvla_engine")
save_dir = args.save_dir

processor = AutoProcessor.from_pretrained("Embodied-CoT/ecot-openvla-7b-bridge", trust_remote_code=True)
vla = TRTOpenVLA(save_dir, engine_dir, "Embodied-CoT/ecot-openvla-7b-bridge", device=device)

print("Getting image...")
url = 'https://raw.githubusercontent.com/MichalZawalski/embodied-CoT/main/test_obs.png'
page = requests.get(url)
image = Image.open(BytesIO(page.content))
print("Done!")

print("Running inference...")
start = time.time()

prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: What action should the robot take to place the watermelon on the towel? ASSISTANT: TASK:"
inputs = processor(prompt, image)

with torch.no_grad():
    action, generated_ids = vla.get_action(inputs, unnorm_key="bridge_orig") 
    torch.cuda.synchronize()

print(f"Done! Time elapsed: {time.time() - start}.")

print(action)
print(processor.decode(generated_ids))