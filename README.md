# TensorRT-OpenVLA

## Installation


## Converting and Compiling with TRT-LLM

TRT-LLM works in two stages: converting from Huggingface to the TRT-LLM checkpoint format, then compiling that checkpoint into a TRT-LLM engine. We currently only support conversion and compilation of the OpenVLA LLM backbone, as that dominates inference times and compute usage. However, TRT-LLM provides examples for how to compile the vision encoder for multimodal models [here](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal).

### Conversion

First, save the LLM backbone, tokenizer, vision backbone, and projector from an OpenVLA-type policy loaded from Huggingface by running the following:
```bash
# Outside Docker shell
cd <PATH TO THIS REPO>/tensorrt-openvla

# Save OpenVLA modules to specified directory
python ./conversion_scripts/save_backbone.py --save-dir <PATH TO SAVE DIR>
# Attempts to load VLA modules separately from specified directory, after being saved with above
python ./conversion_scripts/save_backbone.py --save-dir <PATH TO SAVE DIR> --test-load
```
This saves the Llama 2 backbone from the OpenVLA policy to `<PATH TO SAVE DIR>/LLM_backbone` in the Huggingface format, meaning it can now be converted to the TRT-LLM checkpoint format. If using Docker, we suggest setting `<PATH TO SAVE DIR>` to be within the `TensorRT-LLM` repo, since the bash shell of the Docker automatically mounts that repo to `/code/tensorrt_llm` (and so the Docker container gets access to the save directory). The following will assume that `<PATH TO SAVE DIR>` is `TensorRT-LLM/save_dir`

Now, enter the Docker shell and run:
```bash
# Enter Docker shell
cd ../TensorRT-LLM
sudo make -C docker release_run

cd /code/tensorrt_llm
mkdir ckpts # Used for storing the converted checkpoint and engine
```

To compile with FP8 quantization (faster, but Hopper GPU only, recommended), follow the instructions [here](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/quantization) or run the following commands:
```bash
python ./examples/quantization/quantize.py \
        --model_dir /code/tensorrt_llm/save_dir/LLM_backbone \
        --dtype bfloat16 \
        --qformat fp8 \
        --kv_cache_dtype fp8 \
        --output_dir /code/tensorrt_llm/ckpts/openvla
```
If not using FP8 quantization (slower, but available on non-Hopper), please instead see the instructions for converting [Llama checkpoints](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama) or run the following:
```bash
python ./examples/llama/convert_checkpoint.py \
        --model_dir /code/tensorrt_llm/save_dir/LLM_backbone \
        --dtype bfloat16 \
        --output_dir /code/tensorrt_llm/ckpts/openvla
```

### Compilation
Now that we have the converted checkpoint at `/code/tensorrt_llm/ckpts/openvla`, we can run the TRT-LLM build process. For FP8 builds on Hopper GPUs, run:
```bash
trtllm-build --checkpoint_dir /code/tensorrt_llm/ckpts/openvla \
    --output_dir /code/tensorrt_llm/ckpts/openvla_engine --gemm_plugin fp8 \
    --max_input_len 512 --max_batch_size 4 --max_num_tokens 1024 \
    --use_fp8_context_fmha enable --max_prompt_embedding_table_size 256
```
Otherwise, run:
```bash
trtllm-build --checkpoint_dir /code/tensorrt_llm/ckpts/openvla \
    --output_dir /code/tensorrt_llm/ckpts/openvla_engine --gemm_plugin auto \
    --max_input_len 512 --max_batch_size 4 --max_num_tokens 1024 \
    --max_prompt_embedding_table_size 256
```
This compiles the model into a TRT-LLM engine, located at `/code/tensorrt-llm/ckpts/openvla_engine`. While most arguments above are optional, we note that `--max_prompt_embedding_table_size 256` is not, as this allows the compiled LLM backbone to accept the image embeddings from the vision encoder backbone module.

### Testing Compiled Model

Finally, to test this engine, run the following:
```bash
python ./examples/openvla_run.py --save-dir /code/tensorrt_llm/save_dir
```