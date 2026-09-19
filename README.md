# MoE-Toolkit
A simple toolkit for experimenting with AI models and making your own Mixture of Experts models.

This is the complete, battle-tested pipeline for splicing two (or more) HuggingFace models into a working MoE, converting it to GGUF, quantizing it, and serving it locally.


## What is a MoE?
Classic dense models use their entire 'brain' to compute and process a response for your input. This causes them to be really slow, especially when the models are a large size. Taking this into account, people invented Mixture-of-Expert (MoE) models.

These models are made up of different 'experts' that are each individually good at different tasks. For example, a small MoE may contain 3 experts: one good at Math and Science, another good at English, and the last being good at image analysis. Once you give the model an input, it first goes through a router which determines what 'expert' should be utilised to best respond to your input.

The BEST part about MoE models, is that they only utilise a small portion of their brain causing them to respond in blistering speeds. These are how 'flash' models of flagship AI brands are made. For example, lets say a 20b model has 4 experts with even amounts of brain, so 5b each (most of the time this is unusual). Lets say the user asks about something related to science; the model will only use 5 billion parameters of its 20 billion parameter brain to respond with the science expert.


## Recommended Requirements
Windows 10/11 x64bit, Python 3.10–3.12, git

GPU: NVIDIA (MINIMUM of 8GB VRAM, 16GB recommended)

RAM: 64GB (32GB works, but you may run into a few issues)

Disk Space: ~120GB+ for casual projects | ~150-200GB+ for larger projects


## CONTENTS
llama.cpp folder - llama.cpp source including needed scripts (mainly convert_hf_to_gguf.py)

llama-tools folder - Windows x64 CUDA 12.4 binaries (including cudart DLLs)

tools folders - Full of custom-made python scripts by me to help with your errors (gate_check.py, strip_vision.py, etc.)

setup.bat - Running this will initialize a virtual environment with all the required libraries installed


## AMD GPU / CPU ONLY?
This is less than ideal, but still doable:
- Replace llama.cpp folder with your compatible binary from https://github.com/ggml-org/llama.cpp/releases/
- Idk about llama-tools but I'm sure you'll figure it out :)...


## EXAMPLE OF HOW TO USE THIS TOOLKIT
- RUN SETUP.BAT
- OPEN A NEW COMMAND PROMPT WINDOW AND RUN `moe-env\Scripts\activate`
- DOWNLOAD 2 MODELS OF YOUR CHOICE VIA `hf download MODEL-OF-YOUR-CHOICE --local-dir ./model-ahf download --local-dir ./model-b`
- RUN A COMPARISON / GATE CHECK VIA `python gate_check.py ./model-a ./model-b`
- CREATE A BRAIN YAML FILE AND CONFIG IT TO YOUR WANTED SETTINGS (i'll add a template in later versions)
- RUN `mergekit-moe brain.yml ./my-brain --copy-tokenizer --allow-crimes --lazy-unpickle --out-shard-size 2B` AND IF YOU'RE ON A NVIDIA GPU INCLUDE `--device cuda`
- CONVERT THE MODEL TO GGUF VIA `python llama.cpp\convert_hf_to_gguf.py .\my-brain --outfile brain-f16.gguf --outtype bf16`
- QUANTIZE THE MODEL VIA `llama-tools\llama-quantize.exe brain-f16.gguf brain-Q4_K_M.gguf Q4_K_M` THEN CLEAN-UP WITH `del brain-f16.gguf`
- FINALLY TEST YOUR MODEL VIA `llama-tools\llama-server.exe -m brain-Q4_K_M.gguf -c 16384 -ngl 99 --jinja --port 8080` AND GO TO `http://localhost:8080` (OR THE PORT YOU CHOSE)
