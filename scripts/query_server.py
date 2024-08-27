import requests
from io import BytesIO
from PIL import Image

import json_numpy, json
json_numpy.patch()
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--host', default="0.0.0.0", type=str)
parser.add_argument('--port', default=8000, type=int)
parser.add_argument('--instruction', type=str, default="place the watermelon on the towel")
args = parser.parse_args()

print("Getting image...")
url = 'https://raw.githubusercontent.com/MichalZawalski/embodied-CoT/main/test_obs.png'
page = requests.get(url)
image = Image.open(BytesIO(page.content))
print("Done!")

instruction = args.instruction
prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: What action should the robot take to {instruction}? ASSISTANT:"

# Below prompt is used for base OpenVLA, as downloaded from Huggingface
# prompt = f"In: What action should the robot take to {instruction}?\nOut:"

url = f"http://{args.host}:{args.port}/act"

output = requests.post(
    url,
    json={
            "image": json_numpy.dumps(np.array(image, dtype=np.uint8)), 
            "instruction": prompt,
            "unnorm_key": "bridge_orig",
            "return_ids": False
        }
).json()

print(json.loads(output))

