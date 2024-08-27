import torch
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer, AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--save-dir', default="./save_dir", type=str)
parser.add_argument('--test-load', action='store_true')
parser.add_argument('--hf-name', 
                    default="Embodied-CoT/ecot-openvla-7b-bridge", 
                    type=str)

args = parser.parse_args()

save_dir = args.save_dir

def main():
    # Load OpenVLA
    print("Saving modules separately")
    print("Loading VLA")
    model_name = args.hf_name
    vla = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    llm = vla.language_model
    vision_backbone = vla.vision_backbone
    projector = vla.projector

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = processor.tokenizer
    print("Done!")

    # Save tokenizer / LLM
    print("Saving LLM and tokenizer...")
    llm.save_pretrained(os.path.join(save_dir, "LLM_backbone")) 
    tokenizer.save_pretrained(os.path.join(save_dir, "LLM_backbone"))

    # Save vision backbone and projector
    print("Saving vision backbone and projector...")
    torch.save(vision_backbone.state_dict(), os.path.join(save_dir, "vision_backbone.pth"))
    torch.save(projector.state_dict(), os.path.join(save_dir, "projector.pth"))

    if args.test_load:
        print("Testing load from new save files")

        # Load LLM via HF AutoModel
        print("Loading LLM...")
        llm = AutoModelForCausalLM.from_pretrained(os.path.join(save_dir, "LLM_backbone"))
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(save_dir, "LLM_backbone"))
        print("Successful!")

        # Get class for vision backbone using HF utils, then load
        print("Loading vision backbone...")
        config = AutoConfig.from_pretrained(
                        model_name,
                        trust_remote_code=True
                    )

        vision_backbone_class_ref = config.auto_map[AutoModelForVision2Seq.__name__].replace("OpenVLAForActionPrediction", "PrismaticVisionBackbone")
        vision_backbone_class = get_class_from_dynamic_module(
            vision_backbone_class_ref, model_name
        )
        vision_backbone = vision_backbone_class(
                    config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
                )
        vision_backbone.load_state_dict(torch.load(os.path.join(save_dir, "vision_backbone.pth")))
        print("Successful!")

        # Get class for projector using HF utils, then load
        print("Loading projector...")
        proj_class_ref = config.auto_map[AutoModelForVision2Seq.__name__].replace("OpenVLAForActionPrediction", "PrismaticProjector")
        proj_class = get_class_from_dynamic_module(
            proj_class_ref, model_name
        )
        projector = proj_class(
            config.use_fused_vision_backbone,
            vision_dim=vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        ).load_state_dict(torch.load(os.path.join(save_dir, "projector.pth")))
        print("Successful!")

if __name__ == "__main__":
    main()