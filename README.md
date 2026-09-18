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
setup.bat - Running this will initialize a virtual environment with all the required libraries installed

## AMD GPU / CPU ONLY?
This is less than ideal, but still doable:
- Replace llama.cpp folder with your compatible binary from https://github.com/ggml-org/llama.cpp/releases/
- Idk about llama-tools but I'm sure you'll figure it out :)...

## EXAMPLE OF HOW TO USE THIS TOOLKIT
too lazy, will write later
