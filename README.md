# Finetuning Qwen2.5-1.5B model to be a math reasoning model using GRPO
Arabic Math Problem Solver using Qwen2.5-1.5B-GRPO
Project Overview
This project involves fine-tuning the Qwen2.5-1.5B-Instruct language model to solve mathematical problems in Arabic using reinforcement learning, specifically the GRPO (Generative Reinforcement with Preference Optimization) approach. The model is trained to provide structured responses with reasoning steps and numerical answers in Arabic.
Key Features

Fine-tuned Qwen2.5-1.5B model for Arabic mathematical reasoning
GRPO-based training pipeline with specialized Arabic reward functions
Support for structured output with Arabic reasoning tags
Evaluation system for response quality and correctness

Technical Infrastructure
Development Environment

Platform: Windows 11 with WSL2 (Ubuntu 20.04.6 LTS)
GPU: NVIDIA GeForce RTX 4060 Ti (8GB VRAM)
ML Framework: PyTorch with CUDA support

Key Libraries

unsloth: Model optimization
transformers: Base transformer functionality
trl: GRPO implementation
torch: Deep learning framework
vllm: Inference optimization

Project Structure
math-reasoning-arabic-grpo/
├── src/
│   ├── core/              # Core business logic and model interfaces
│   ├── data/              # Dataset handling and preprocessing
│   ├── infrastructure/    # Training and model infrastructure
│   └── utils/             # Utility functions
├── tests/                 # Test suites
├── configs/               # Configuration files
├── notebooks/            # Jupyter notebooks for experiments
└── scripts/              # Utility scripts

