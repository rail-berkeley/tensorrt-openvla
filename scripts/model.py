import os

import torch
from transformers import (AutoConfig, AutoProcessor, AutoModelForVision2Seq)

from transformers.dynamic_module_utils import get_class_from_dynamic_module

import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

import requests
import argparse

class TRTOpenVLA:
    def __init__(self, save_dir, engine_dir, hf_name, device="cuda"):
        self.engine_dir = engine_dir
        self.save_dir = save_dir
        self.hf_name = hf_name

        self._setup_llm_runner()
        self.processor = AutoProcessor.from_pretrained(self.hf_name, 
                                                       trust_remote_code=True)
        self._setup_vision_backbone_and_projector()

        
    def _setup_llm_runner(self):
        runtime_rank = tensorrt_llm.mpi_rank()
        runner_kwargs = dict(
                engine_dir=self.engine_dir,
                rank=runtime_rank,
            )
        self.llm_runner = ModelRunner.from_dir(**runner_kwargs)

    def _setup_vision_backbone_and_projector(self):
        config = AutoConfig.from_pretrained(
                        self.hf_name,
                        trust_remote_code=True
                    )

        vision_backbone_class_ref = config.auto_map[AutoModelForVision2Seq.__name__].replace("OpenVLAForActionPrediction", "PrismaticVisionBackbone")
        vision_backbone_class = get_class_from_dynamic_module(
            vision_backbone_class_ref, self.hf_name
        )
        self.vision_backbone = vision_backbone_class(
                    config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
                )
        self.vision_backbone.load_state_dict(torch.load(os.path.join(self.save_dir, "vision_backbone.pth"), map_location=self.device))
        self.vision_backbone.featurizer.to(self.device, torch.bfloat16)
        print("Successful!")

        # Get class for projector using HF utils, then load
        print("Loading projector...")
        proj_class_ref = config.auto_map[AutoModelForVision2Seq.__name__].replace("OpenVLAForActionPrediction", "PrismaticProjector")
        proj_class = get_class_from_dynamic_module(
            proj_class_ref, self.hf_name
        )
        self.projector = proj_class(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )
        self.projector.load_state_dict(torch.load(os.path.join(self.save_dir, "projector.pth"), 
                                             map_location=self.device))
        self.projector.to(self.device, torch.bfloat16)

    def parse_input(self,
                    input_text,
                    add_special_tokens=True,
                    max_input_length=512
                   ):
        tokenizer = self.processor.tokenizer
        batch_input_ids = []
        for curr_text in input_text:
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
    
    def generate(self, 
                 prompt, 
                 image,
                 max_new_tokens=400,
                 do_sample=False,
                 random_seed=0,
                 **kwargs):
        pixel_values = self.processor("", image)["pixel_values"].to(self.device, dtype=torch.bfloat16)
        patch_features = self.vision_backbone(pixel_values)
        projected_patch_embeddings = self.projector(patch_features)
        batch_input_ids = self.parse_input([prompt])

        end_id = self.processor.tokenizer.eos_token_id
        pad_id = self.processor.tokenizer.pad_token_id

        outputs = self.runner.generate(
            batch_input_ids=batch_input_ids,
            encoder_input_ids=None,
            max_new_tokens=max_new_tokens,
            end_id=end_id,
            pad_id=pad_id,
            random_seed=random_seed,
            do_sample=do_sample,
            prompt_table=projected_patch_embeddings,
            prompt_vocab_size=256,
            **kwargs
        )
        torch.cuda.synchronize()
        return outputs
    
    def get_action(self, instruction, image, **kwargs):
        prompt = f"A chat between a curious user and an artificial intelligence assistant." +\
            f" The assistant gives helpful, detailed, and polite answers to the user's questions." + \
            f" USER: What action should the robot take to {instruction.lower()}? ASSISTANT: TASK:"
        
        outputs = self.generate(prompt, 
                                image,
                                max_new_tokens=400,
                                do_sample=False,
                                random_seed=0,
                                return_dict=True
                                **kwargs)
        
        output_ids = outputs['output_ids']
        output_text = self.processor.tokenizer.decode(output_ids.reshape(-1), skip_special_tokens=True)
        raise NotImplementedError



        