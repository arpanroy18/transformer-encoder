# From-Scratch Transformer Decoder with KV Cache

**Author:** Arpan Roy  
**Date:** May 10, 2025

## Overview

The **From-Scratch Transformer Decoder** project implements a decoder-only transformer model using NumPy, focusing on educational purposes and real-time text generation. This project showcases the mechanics of transformer architectures, particularly the integration of a Key-Value (KV) caching mechanism to optimize sequential text generation. The implementation is designed to be simple yet robust, allowing for a deep understanding of the underlying principles without relying on high-level machine learning frameworks.

## Table of Contents

- [Project Goals](#project-goals)
- [Key Features](#key-features)
- [Technical Stack](#technical-stack)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)

## Project Goals

- Implement a functional transformer decoder block entirely from scratch using Python and NumPy.
- Integrate a manual Key-Value (KV) caching mechanism to optimize sequential text generation.
- Demonstrate the performance benefits of using a KV cache.
- Provide a clear, interactive demonstration of token-by-token text generation.
- Serve as an educational tool for understanding the internal workings of transformer-based language models.

## Key Features

1.  **Transformer Decoder from First Principles:**
    -   Implementation of decoder blocks with linear layers for Query (Q), Key (K), Value (V) projections.
    -   Scaled dot-product attention mechanism with causal masking for autoregressive generation.
    -   Layer normalization and residual connections for stability.
2.  **Manual KV Cache System:**
    -   Efficient storage and retrieval of K and V tensors from previous generation steps.
    -   Logic to reuse cached K/V tensors in subsequent attention calculations.
3.  **Interactive CLI/Notebook Demo:**
    -   User interface for inputting text prompts and generating subsequent characters.
    -   Display of generation time for each token, highlighting the KV cache's impact.
4.  **Visualization Module (Optional):**
    -   Attention heatmaps to show which previous tokens are being attended to.
    -   Graphs illustrating the growth of the KV cache over generation steps.

## Technical Stack

-   **Core Language:** Python 3.x
-   **Numerical Computation:** NumPy (latest stable version)
-   **Development Environment:** Jupyter Notebook / JupyterLab
-   **Visualization:** Matplotlib (for optional visualizations)

## Architecture

The project is structured as follows:

-   **Tokenizer:** A character-level tokenizer that maps characters to unique integer IDs and vice versa.
-   **Embedding Layer:** Converts token IDs into dense vector representations.
-   **Decoder Block Module:** Implements multi-head self-attention, feed-forward networks, and layer normalization.
-   **Output Projection Layer:** Projects the decoder output to vocabulary-sized logits.
-   **Generation Logic:** Handles the generation of text token-by-token, utilizing the KV cache for efficiency.

## Installation

To set up the project locally, follow these steps:

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd transformer_from_scratch
    ```
2.  Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the interactive demo, open the Jupyter Notebook:

```bash
jupyter notebook notebooks/demo.ipynb
```

You can input a text prompt, and the model will generate text token-by-token, showcasing the KV cache's efficiency.
