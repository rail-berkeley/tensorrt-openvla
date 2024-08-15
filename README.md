# TensorRT-OpenVLA

## Installation

This repo requires TensorRT-LLM and Huggingface Transformers (to load OpenVLA models). We will assume that this repo and TensorRT-LLM share the same parent directory.

The instructions for installing TensorRT-LLM can be found [here](https://nvidia.github.io/TensorRT-LLM/installation/linux.html). We recommend using the Docker container, which may require following the [build from source instructions](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html).

```bash
sudo apt-get update && apt-get -y install git git-lfs
git lfs install

git clone https://github.com/rail-berkeley/tensorrt-openvla.git
git clone https://github.com/NVIDIA/TensorRT-LLM.git

cd TensorRT-LLM
git submodule update --init --recursive
git lfs pull

# Build Docker container, may take a while
make -C docker release_build
```
To enter the Docker's bash shell, run:
```bash
sudo make -C docker release_run
```
You will also need to install some packages within the Docker container to run our example code:
```bash
pip install timm==0.9.10 imageio opencv-python
```
We recommend editing `TensorRT-LLM/docker/Makefile` to also mount this repo when `release_run` is called by adding:
```bash
--volume <PATH TO THIS REPO>/tensorrt-openvla:/code/tensorrt-openvla \
```
between lines 122 and 123.

In a different terminal, set up a Conda environment for our repo, then install the desired OpenVLA model. We will be using the [Embodied Chain-of-Thought VLA](https://github.com/MichalZawalski/embodied-CoT/) as an example, as it was built atop OpenVLA.
```bash
conda create -n tensorrt-openvla python=3.10 -y
conda activate tensorrt-openvla
pip install git+https://github.com/MichalZawalski/embodied-CoT/
```
By default, the Docker container will mount `TensorRT-LLM` to `/code/tensorrt_llm`.

## Converting and Compiling with TRT-LLM

TRT-LLM works in two stages: converting from Huggingface to the TRT-LLM checkpoint format, then compiling that checkpoint into a TRT-LLM engine. We currently only support conversion and compilation of the OpenVLA LLM backbone, as that dominates inference times and compute usage. However, TRT-LLM provides examples for how to compile the vision encoder for multimodal models [here](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal).

### Conversion

First, save the LLM backbone, tokenizer, vision backbone, and projector from an OpenVLA-type policy loaded from Huggingface by running the following:
```bash
# Outside Docker shell
cd <PATH TO THIS REPO>/tensorrt-openvla

# Save OpenVLA modules to specified directory
python ./scripts/save_backbone.py --save-dir <PATH TO SAVE DIR>
# Attempts to load VLA modules separately from specified directory, after being saved with above
python ./scripts/save_backbone.py --save-dir <PATH TO SAVE DIR> --test-load
```
This saves the Llama 2 backbone from the OpenVLA policy to `<PATH TO SAVE DIR>/LLM_backbone` in the Huggingface format, meaning it can now be converted to the TRT-LLM checkpoint format. If using Docker, we suggest setting `<PATH TO SAVE DIR>` to be within this repo, if you modified the `Makefile` as described above to mount this repo within the container. The following will assume that `<PATH TO SAVE DIR>` is `tensorrt-openvla/save_dir` and that this repo is mounted to `/code/tensorrt-openvla`.

Now, enter the Docker shell and run:
```bash
# Enter Docker shell
cd ../TensorRT-LLM
sudo make -C docker release_run
cd /code/tensorrt_llm
```

To compile with FP8 quantization (faster, but Hopper GPU only, recommended), follow the instructions [here](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/quantization) or run the following commands:
```bash
python ./examples/quantization/quantize.py \
        --model_dir /code/tensorrt-openvla/save_dir/LLM_backbone \
        --dtype bfloat16 --qformat fp8 --kv_cache_dtype fp8 \
        --output_dir /code/tensorrt-openvla/ckpts/openvla
```
If not using FP8 quantization (slower, but available on non-Hopper), please instead see the instructions for converting [Llama checkpoints](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama) or run the following:
```bash
python ./examples/llama/convert_checkpoint.py \
        --model_dir /code/tensorrt-openvla/save_dir/LLM_backbone \
        --dtype bfloat16 --output_dir /code/tensorrt-openvla/ckpts/openvla
```

### Compilation
Now that we have the converted checkpoint at `/code/tensorrt_llm/ckpts/openvla`, we can run the TRT-LLM build process. For FP8 builds on Hopper GPUs, run:
```bash
trtllm-build --checkpoint_dir /code/tensorrt-openvla/ckpts/openvla \
    --output_dir /code/tensorrt-openvla/ckpts/openvla_engine --gemm_plugin fp8 \
    --max_input_len 512 --max_batch_size 4 --max_num_tokens 1024 \
    --use_fp8_context_fmha enable --max_prompt_embedding_table_size 256
```
Otherwise, run:
```bash
trtllm-build --checkpoint_dir /code/tensorrt-openvla/ckpts/openvla \
    --output_dir /code/tensorrt-openvla/ckpts/openvla_engine --gemm_plugin auto \
    --max_input_len 512 --max_batch_size 4 --max_num_tokens 1024 \
    --max_prompt_embedding_table_size 256
```
This compiles the model into a TRT-LLM engine, located at `/code/tensorrt-openvla/ckpts/openvla_engine`. While most arguments above are optional, we note that `--max_prompt_embedding_table_size 256` is not, as this allows the compiled LLM backbone to accept the image embeddings from the vision encoder backbone module.

### Testing Compiled Model

Finally, to test this engine, run the following:
```bash
python /code/tensorrt-openvla/scripts/openvla_run.py \
    --save-dir /code/tensorrt-openvla/save-dir --ckpts-dir /code/tensorrt-openvla/ckpts
```
This should run the forward pass on the example image from the [ECoT Colab example](https://colab.research.google.com/drive/1CzRKin3T9dl-4HYBVtuULrIskpVNHoAH?usp=sharing), and the compiled model should output a similarly reasonable reasoning chain as in that notebook.

## TRT-OpenVLA Deployment
As the compiled OpenVLA policy is most easily run inside the Docker container, we provide code to run the model as an inference server (run inside the Docker), which an external client can query for actions.

First, edit `TensorRT-LLM/docker/Makefile` and add `-p 8000:8000 \` to the arguments of the command on line 120 (e.g., after the `--volume` argument added during the Installation section above). This mounts port `8000` on the host machine to `8000` within the container as well. Then, enter the Docker and install some dependencies for running the server:
```bash
pip install timm==0.9.10 fastapi uvicorn json-numpy draccus
```
Finally, run the following within the Docker to start up the server:
```bash
cd /code/tensorrt-openvla
python scripts/deploy_server.py --save-dir /code/tensorrt-openvla/save_dir/ --engine-dir /code/tensorrt-openvla/ckpts/openvla_engine/
```
This will start the server at `0.0.0.0:8000/act` by default. Use `--host` and `--port` to specify the host IP and port respectively.

Now, outside the Docker, you can run:
```bash
python scripts/query_server.py
```
This will send a request to `0.0.0.0:8000/act` containing the image (as an `np.ndarray`) and instruction string. After running inference, it should return with a dictionary containing the action (as a 7 element `np.ndarray`) and the generated reasoning as a string. If the generated IDs are desired instead, change `return_ids` in the sent request to `True`.
