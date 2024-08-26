import os

import json_numpy

json_numpy.patch()

import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
from model import TRTOpenVLA
from transformers import AutoProcessor
import argparse

class TRTOpenVLAServer:
    def __init__(self, save_dir, engine_dir, hf_name, device="cuda"):
        self.hf_name = hf_name
        self.device = device
        self.trt_openvla = TRTOpenVLA(save_dir, engine_dir, self.hf_name, device)
        self.processor = AutoProcessor.from_pretrained(self.hf_name, trust_remote_code=True)

    def predict_action(self, payload: Dict[str, Any]) -> str:
        try:
            # payload = json.loads(payload)

            # Parse payload components
            image, instruction = json_numpy.loads(payload["image"]), payload["instruction"]
            unnorm_key = payload.get("unnorm_key", None)

            # Run VLA Inference
            inputs = self.processor(instruction, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)
            action, generated_ids = self.trt_openvla.get_action(inputs, unnorm_key=unnorm_key, do_sample=False)

            if payload["return_ids"]:
                generated = np.array(generated_ids.cpu())
            else:
                generated = self.processor.decode(generated_ids, skip_special_tokens=True)
            return JSONResponse(json_numpy.dumps({"action": action, "generated": generated}))
        
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)


def deploy(args) -> None:
    server = TRTOpenVLAServer(args.save_dir, args.engine_dir, args.hf_name)
    server.run(args.host, port=args.port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', default="./save_dir", type=str)
    parser.add_argument('--engine-dir', default="./ckpts/openvla-engine", type=str)
    parser.add_argument('--hf-name', default="Embodied-CoT/ecot-openvla-7b-bridge", type=str)
    parser.add_argument('--host', default="0.0.0.0", type=str)
    parser.add_argument('--port', default=8000, type=int)
    args = parser.parse_args()
    deploy(args)