---
layout: page
title: Projects
subtitle: What's been eating away my free time
---

This is a collection of my personal projects that I work on in my free time. Hope you like them.

---

## Telugu Morph Tokenizer

<a href="https://github.com/UXK005/indic-morph-tokenizer" target="_blank"><img src="https://img.shields.io/badge/GitHub-blue"></a>
<a href="https://huggingface.co" target="_blank"><img src="https://img.shields.io/badge/Hugging%20Face-yellow"></a>

A Telugu-specific tokenizer built with linguistic awareness, designed to perform morphology-aware and sandhi-informed segmentation instead of generic byte-level tokenization.

The tokenizer supports the following features:

**1. Telugu-aware Tokenization**
Processes Telugu text using Unicode-normalized grapheme clusters to preserve valid character groupings and avoid incorrect splits.

**2. Morphology & Sandhi-aware Segmentation**
Splits Telugu words into roots and suffixes (plural, case markers) following grammatical ordering rules, with optional sandhi-aware processing.

**3. Subword Tokenizer Training Support**
Generates linguistically meaningful token streams suitable for training BPE-based subword tokenizers for Telugu language models.

**Tools Used**: Python, Unicode (UTF-8), Regular Expressions, Rule-based NLP, Hugging Face

<br/>

## ResNet ImageNet Training

<a href="https://github.com/adensur/blog/blob/main/computer_vision_zero_to_hero/08_resnet_imagenet/Readme.md" target="_blank"><img src="https://img.shields.io/badge/GitHub-blue"></a>
<a href="https://youtu.be/uztrVK1BhGw" target="_blank"><img src="https://img.shields.io/badge/Video-YouTube-red"></a>

Implementation and training of ResNet models from scratch on the ImageNet-1k dataset, following the original [ResNet paper](https://arxiv.org/pdf/1512.03385.pdf). This project demonstrates training deep convolutional neural networks on large-scale image classification tasks.

**Key Features:**

- **Full ResNet Implementation**: Trained ResNet-18 and ResNet-152 models from scratch using PyTorch
- **ImageNet-1k Dataset**: Worked with 1.2 million images across 1000 classes (~160GB dataset)
- **Performance Optimization**: Achieved top-5 error rate of 11% on ResNet-18 after 90 epochs
- **Training Infrastructure**: Optimized data loading pipeline, GPU memory management, and learning rate scheduling

**Technical Highlights:**
- Implemented proper data preprocessing and augmentation pipelines
- Optimized training loop with batch normalization and residual connections
- Debugged and optimized data loading bottlenecks and GPU memory usage
- Achieved results comparable to original paper implementations

**Tools Used**: Python, PyTorch, TorchVision, CUDA, ImageNet Dataset, TensorBoard

<br/>

## Topic Based Image Captioning

<a href="https://github.com/shan18/Topic-Based-Image-Captioning" target="_blank"><img src="https://img.shields.io/badge/GitHub-blue"></a>

An automatic image caption generation system built using Deep Learning.

- Devised a novel image captioning model using CNNs and LSTMs trained on MSCOCO-2017 Dataset.
- Created a system where LSTMs were given additional information (topics) extracted from image captions using Latent Dirichlet Allocation (LDA).

**Tools Used**: Python, Tensoflow, NLTK, OpenCV

---
