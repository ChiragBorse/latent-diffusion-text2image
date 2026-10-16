# 🌌 Conditioned Text-to-Image Generation using Latent Diffusion Models

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Diffusers](https://img.shields.io/badge/%F0%9F%A4%97-Diffusers-orange)](https://huggingface.co/docs/diffusers)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An implementation of text-conditioned image generation using **Latent Diffusion Models (LDMs)**. This project demonstrates how text embeddings guide reverse Markovian noise sampling to synthesize photorealistic images.

---

## 📌 Problem Statement
**Question:** Implement a text-conditioned generative vision pipeline using Latent Diffusion Models (Stable Diffusion) to synthesize high-resolution images from natural language descriptions.

---

## ❓ Project Overview (The 5 Ws)

* **What:** A multimodal Generative AI pipeline that accepts text descriptions (prompts) and outputs high-fidelity images via reverse diffusion denoising in a compressed latent space.
* **Why:** Unlike GANs, which suffer from mode collapse and training instability, diffusion models offer stable likelihood-based training objectives and higher sample diversity. Operating in latent space (via a VAE encoder/decoder) makes high-resolution generation computationally feasible.
* **Who:** Students, researchers, and developers exploring multimodal AI, cross-attention mechanisms, and generative vision architectures.
* **Where:** Designed for execution on Google Colab (with a free T4 GPU) or local CUDA-enabled machines.
* **How:** Text is converted to embeddings using a frozen CLIP text encoder, which directs a U-Net denoiser via cross-attention layers across $T$ discrete timesteps.

---

## ⚙️ Architecture & Technical Workflow

```text
 Text Prompt ("Cyberpunk City")
             │
             ▼
 ┌──────────────────────┐
 │  CLIP Text Encoder   │ ──► Context Embeddings (Key/Value vectors)
 └──────────┬───────────┘
            │
            │ (Cross-Attention)
            ▼
 Random Latent Noise z_T ~ N(0, I)
            │
            ▼
 ┌─────────────────────────────────────────────────────────┐
 │               Time-Conditioned U-Net                    │
 │  Performs iterative denoising: z_t-1 = Denoise(z_t, t)  │ ◄── Guidance Scale (CFG)
 └──────────────────────────┬──────────────────────────────┘
                            │ (Iterate N steps)
                            ▼
                  Cleaned Latent Representation z_0
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────┐
 │                  VAE Decoder (spatial)                  │
 └──────────────────────────┬──────────────────────────────┘
                            │
                            ▼
                 Synthesized RGB Image (512x512)
