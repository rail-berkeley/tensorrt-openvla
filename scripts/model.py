import os
import numpy as np

import torch
from transformers import (AutoConfig, AutoProcessor, AutoModelForVision2Seq)

from transformers.dynamic_module_utils import get_class_from_dynamic_module

import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

# TODO: Refactor to directly subclass OpenVLAForActionPrediction

class TRTOpenVLA:
    def __init__(self, save_dir, engine_dir, hf_name, device="cuda"):
        self.engine_dir = engine_dir
        self.save_dir = save_dir
        self.hf_name = hf_name
        self.device = device
        
        self.config = AutoConfig.from_pretrained(
                        self.hf_name,
                        trust_remote_code=True
                    )
        
        self._setup_action_info()
        self._setup_llm_runner()
        self._setup_vision_backbone_and_projector()

        self.end_id = 2 # TODO: Change how to extract EOS token ID
        self.pad_id = self.config.pad_token_id

    def _setup_action_info(self):
        self.norm_stats = self.config.norm_stats
        self.bins = np.linspace(-1, 1, self.config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of
        
    def _setup_llm_runner(self):
        runtime_rank = tensorrt_llm.mpi_rank()
        runner_kwargs = dict(
                engine_dir=self.engine_dir,
                rank=runtime_rank,
            )
        self.llm_runner = ModelRunner.from_dir(**runner_kwargs)

    def _setup_vision_backbone_and_projector(self):

        vision_backbone_class_ref = self.config.auto_map[AutoModelForVision2Seq.__name__].replace("OpenVLAForActionPrediction", 
                                                                                                  "PrismaticVisionBackbone")
        vision_backbone_class = get_class_from_dynamic_module(
            vision_backbone_class_ref, self.hf_name
        )
        self.vision_backbone = vision_backbone_class(
                    self.config.use_fused_vision_backbone, self.config.image_sizes, 
                    self.config.timm_model_ids, self.config.timm_override_act_layers
                )
        self.vision_backbone.load_state_dict(torch.load(os.path.join(self.save_dir, "vision_backbone.pth"), map_location=self.device))
        self.vision_backbone.featurizer.to(self.device, torch.bfloat16)
        print("Successful!")

        # Get class for projector using HF utils, then load
        print("Loading projector...")
        proj_class_ref = self.config.auto_map[AutoModelForVision2Seq.__name__].replace("OpenVLAForActionPrediction", "PrismaticProjector")
        proj_class = get_class_from_dynamic_module(
            proj_class_ref, self.hf_name
        )
        self.projector = proj_class(
            self.config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=self.config.text_config.hidden_size,
        )
        self.projector.load_state_dict(torch.load(os.path.join(self.save_dir, "projector.pth"), 
                                             map_location=self.device))
        self.projector.to(self.device, torch.bfloat16)

    def parse_input(self, input_ids):
        base_vocab_size = 32064
        batch_size = input_ids.shape[0]
        
        virt_tokens = torch.tensor([list(range(base_vocab_size, base_vocab_size + 256))] * batch_size, dtype=torch.int32)
        batch_input_ids = torch.cat([input_ids[:, :1], virt_tokens.to(input_ids.device), input_ids[:, 1:]], dim=1)

        # for i in range(len(batch_input_ids)):
        #     batch_input_ids[i] = batch_input_ids[i][:1] + \
        #         list(range(base_vocab_size, base_vocab_size + 256)) + \
        #         batch_input_ids[i][1:]

        # batch_input_ids = [
        #     torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
        # ]
        return batch_input_ids
    
    def generate(self, 
                 inputs,
                 max_new_tokens=400,
                 do_sample=False,
                 random_seed=0,
                 **kwargs):
        
        pixel_values = inputs["pixel_values"].to(self.device, dtype=torch.bfloat16)
        patch_features = self.vision_backbone(pixel_values)
        projected_patch_embeddings = self.projector(patch_features)
        batch_input_ids = self.parse_input(inputs["input_ids"])

        outputs = self.llm_runner.generate(
            batch_input_ids=batch_input_ids,
            encoder_input_ids=None,
            max_new_tokens=max_new_tokens,
            end_id=self.end_id,
            pad_id=self.pad_id,
            random_seed=random_seed,
            do_sample=do_sample,
            prompt_table=projected_patch_embeddings,
            prompt_vocab_size=256,
            **kwargs
        )
        torch.cuda.synchronize()
        return outputs
    
    def get_action(self, inputs, unnorm_key=None, do_sample=False, random_seed=0, **kwargs):
        outputs = self.generate(inputs,
                                max_new_tokens=512,
                                return_dict=True,
                                do_sample=do_sample,
                                random_seed=random_seed,
                                **kwargs)
        
        generated_ids = outputs['output_ids']
        
        # TRT-LLM Llama generates EOS tokens until max_new_tokens is saturated, so need to remove all EOS tokens
        not_eos_mask = generated_ids != self.end_id
        generated_ids = generated_ids[not_eos_mask]

        #  Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[-self.get_action_dim(unnorm_key):].cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]

        # Unnormalize actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions, generated_ids

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None and len(norm_stats) != 1:
            raise ValueError(
                f"Your model was trained on more than one dataset. "
                f"Please pass a `unnorm_key` from the following options to choose the statistics used for "
                f"de-normalizing actions: {norm_stats.keys()}"
            )

        # If None, grab the (singular) dataset in `norm_stats` to use as `unnorm_key`
        unnorm_key = unnorm_key if unnorm_key is not None else next(iter(norm_stats.keys()))
        if unnorm_key not in norm_stats:
            raise ValueError(
                f"The `unnorm_key` you chose ({unnorm_key = }) is not in the available statistics. "
                f"Please choose from: {norm_stats.keys()}"
            )

        return unnorm_key

    def get_action_dim(self, unnorm_key=None):
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key=None) :
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

        