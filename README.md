# Phi-2 Fine-tuning with GRPO and QLoRA

A lightweight and efficient approach to fine-tuning Microsoft's Phi-2 model using Generative Reinforcement Policy Optimization (GRPO) and Quantized Low-Rank Adaptation (QLoRA).

## Overview

This project explores fine-tuning the Phi-2 language model (1.3B parameters) using a combination of advanced techniques:

- **GRPO**: Generative Reinforcement Policy Optimization for reward-based learning
- **QLoRA**: Quantized Low-Rank Adaptation for parameter-efficient fine-tuning
- **Unsloth**: Optimization library for faster training and reduced memory usage

The implementation focuses on optimizing language generation with custom reward functions, specifically using BLEU scores to guide the model's learning process on the TLDR dataset.

## Features

- 4-bit quantization for reduced memory footprint
- LoRA rank 8 for parameter-efficient training
- Only 0.26% (3.9M/1.5B) of parameters trained
- vLLM-style speed optimizations
- Custom BLEU-based reward function
- Optimized hyperparameters for GRPO training

## Training Details

- **Training Steps**: 300
- **Batch Size**: 6 per device × 4 gradient accumulation × 1 GPU = 24 total batch size
- **Dataset Size**: 129,722 examples (TLDR dataset)
- **Trainable Parameters**: 3,932,160 out of 1,525,324,800 (0.26% trained)
- **Hardware**: Single GPU configuration
- **Training Framework**: Unsloth with GRPO and QLoRA

## Results

### Training Loss

| Step | Training Loss |
|------|--------------|
| 5    | 0.003900     |
| 10   | 0.004800     |
| 15   | 0.003500     |
| 20   | 0.004200     |
| 25   | 0.004000     |
| 30   | 0.003600     |
| 35   | 0.003400     |
| 40   | 0.003700     |
| 45   | 0.003500     |
| 50   | 0.003900     |
| 55   | 0.003600     |
| 60   | 0.003400     |
| 65   | 0.003600     |
| 70   | 0.003800     |

Our fine-tuning approach showed stable training loss values around 0.003-0.004 throughout the 300 training steps. While the primary objective was to improve summarization capabilities, the most noticeable change was in the model's response style and tone compared to the base Phi-2 model, rather than significant improvements in summarization quality.

## Future Directions

1. **Improved Reward Functions**: Replace BLEU with more sophisticated metrics like ROUGE, BERTScore, or custom reward models tailored for summarization tasks.

2. **Hybrid Reward Approaches**: Combine multiple reward signals (e.g., relevance, coherence, factuality) to provide more nuanced guidance.

3. **Extended Training**: Increase training steps beyond 300 to potentially achieve more significant improvements in summarization quality.

4. **Alternative Datasets**: Explore different datasets beyond TLDR for fine-tuning to better target specific capabilities.

5. **Hyperparameter Optimization**: Further tune GRPO and QLoRA parameters to enhance training efficiency and outcomes.

6. **Human Feedback Integration**: Incorporate human preferences directly into the reward loop to better align with expectations.

7. **Evaluation Framework**: Develop comprehensive benchmarks to assess improvements in summarization quality beyond tone changes.

8. **Multi-task Learning**: Train on multiple related tasks simultaneously with shared parameters for more versatile models.

9. **Distillation Approaches**: Explore knowledge distillation from larger summarization models to transfer capabilities to Phi-2.

10. **Reward Shaping**: Introduce intermediate rewards during training to guide the model more effectively toward desired summarization behavior.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Unslot
