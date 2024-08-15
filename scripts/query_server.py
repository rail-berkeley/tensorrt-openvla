import requests
from io import BytesIO
from PIL import Image

import json_numpy, json
json_numpy.patch()
import numpy as np

print("Getting image...")
url = 'https://raw.githubusercontent.com/MichalZawalski/embodied-CoT/main/test_obs.png'
page = requests.get(url)
image = Image.open(BytesIO(page.content))
print("Done!")

prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: What action should the robot take to place the watermelon on the towel? ASSISTANT: TASK:"

url = "http://0.0.0.0:8000/act"

output = requests.post(
    url,
    json={
        "encoded": json_numpy.dumps({
            "image": np.array(image, dtype=np.uint8), 
            "instruction": prompt,
            "unnorm_key": "bridge_orig",
            "return_ids": False
        })
    }
).json()

print(json.loads(output))

