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

## TensorNet

<a href="https://pypi.org/project/torch-tensornet/" target="_blank"><img src="https://img.shields.io/badge/PyPI-9cf"></a>
<a href="https://github.com/shan18/TensorNet" target="_blank"><img src="https://img.shields.io/badge/GitHub-blue"></a>

A high-level deep learning library built on top of PyTorch to solve computer vision problems.

**Tools Used**: Python, PyTorch

<br/>

## Topic Based Image Captioning

<a href="https://github.com/shan18/Topic-Based-Image-Captioning" target="_blank"><img src="https://img.shields.io/badge/GitHub-blue"></a>

An automatic image caption generation system built using Deep Learning.

- Devised a novel image captioning model using CNNs and LSTMs trained on MSCOCO-2017 Dataset.
- Created a system where LSTMs were given additional information (topics) extracted from image captions using Latent Dirichlet Allocation (LDA).

**Tools Used**: Python, Tensoflow, NLTK, OpenCV

---
