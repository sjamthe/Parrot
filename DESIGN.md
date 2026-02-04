# ParrotMoshi Design Document

## 1. Overview
ParrotMoshi is a **Speech-to-Speech Voice Conversion** model designed to clone a target voice while preserving the semantic content of a source audio.

It mimics the architectural principles of **Moshi** (Kyutai) and **PersonaPlex** (NVIDIA), specifically the handling of **Neural Audio Codecs (Mimi)**. The key difference is that ParrotMoshi removes the Text/LLM component, focusing purely on Audio-to-Audio mapping.

## 2. The Core Problem
Neural Audio Codecs like **Mimi** output multiple "Codebooks" (Layers) of tokens for every time step (e.g., 32 codebooks at 12.5Hz).
- **Codebook 0 (C0):** Coarse, semantic information.
- **Codebook 31 (C31):** Fine, high-frequency acoustic details.

**The Challenge:**
- Predicting all 32 codebooks independently (NAR) leads to inconsistent, noisy audio ("garbled").
- Predicting only C0 (AR) leads to robotic, low-fidelity audio ("machine gun").
- Predicting all 32 sequentially (flattened) makes the sequence too long to train efficiently.

## 3. The Solution: RQ-Transformer (Residual Quantization)
We employ a **Joint Autoregressive** strategy that decomposes the probability of the audio sequence $A$ into **Time** and **Depth** steps.

$P(A) = \prod_{t=1}^{T} \prod_{k=0}^{K} P(Code_{t,k} | History_{<t}, Codes_{t, <k})$

This means:
1.  **Temporal Step:** To predict the audio at time $t$, we look at all previous audio frames.
2.  **Depth Step:** To predict Codebook $k$ at time $t$, we look at the coarser codebooks ($0$ to $k-1$) *at the same time step*.

### 4. Architecture: The "Chain-Rule" Transformer

ParrotMoshi consists of a **single unified model** with internal staging.

#### 4.1. Inputs
1.  **Source Audio Tokens:** $[B, T, 32]$ (Encoded by Mimi). Used as the "Prompt" or "Context".
2.  **Target Speaker Embedding:** $[B, 192]$ (From SpeechBrain). Conditions the style.
3.  **Target Audio History:** $[B, T_{past}, 32]$ (Previous steps for AR generation).

#### 4.2. Components

**A. Source Encoder (Context)**
- A standard **Transformer Encoder**.
- Processes the Source Audio (sum of 32 codebook embeddings).
- Output: `Memory` $[B, T, D]$ used for Cross-Attention.

**B. Temporal Decoder (The "Flow")**
- A **Transformer Decoder** (Causal).
- **Input:** A fused representation of the *previous* time step $t-1$.
    - $Input_t = Linear(\sum_{k=0}^{31} Embed(C_{t-1, k})) + SpeakerEmbedding$
- **Function:** Models the progression of time.
- **Output:** A latent vector $H_t$ representing the "state" of audio at time $t$.

**C. Depth Modules (The "Texture")**
- A **Transformer Decoder** (Depformer).
- **Function:** Models the dependencies between codebooks $C_0 \dots C_{31}$ at a single time step.
- **Input:** 
    - Context: The Temporal Latent $H_t$ (from the Temporal Decoder).
    - Sequence: $[SOS, C_{t,0}, C_{t,1}, \dots, C_{t,30}]$
- **Output:** The next codebook in the sequence $[C_{t,0}, C_{t,1}, \dots, C_{t,31}]$.
- This allows for complex, non-linear dependencies between layers (e.g., layer 10 depending on layer 2 in complex ways), matching the architecture of the original Moshi Depformer.

## 5. Comparison with Moshi

| Feature | Moshi (Original) | ParrotMoshi (Ours) |
| :--- | :--- | :--- |
| **Goal** | Full Duplex Dialogue (Text+Audio) | Voice Conversion (Audio-to-Audio) |
| **Inputs** | Text Stream + User Audio Stream | Source Audio (Content) + Target Speaker |
| **Backbone** | Helium LLM (7B parameters) | Transformer Decoder (~100M parameters) |
| **Codec** | Mimi (32 Codebooks) | Mimi (32 Codebooks) |
| **Temporal Modeling** | Autoregressive (Main Transformer) | Autoregressive (Main Transformer) |
| **Depth Modeling** | Inner "Depformer" (Transformer) | **Inner "Depformer" (Transformer)** |
| **Cross-Attention** | None (Decoder-only) | **Yes** (To align with Source Audio) |

**Key Deviation:**
Moshi creates speech "from scratch" (or text), so it is a pure Decoder.
ParrotMoshi must follow the **Source Audio** content, so we add an **Encoder** and **Cross-Attention** (Seq2Seq). This effectively makes it a "Translator" rather than a "Generator".

## 6. Training Strategy
- **Loss:** Joint Cross-Entropy over all 32 codebooks.
- **Efficiency:** We process the Depth step in parallel during training by flattening dimensions $(B \times T, 32)$.
- **Teacher Forcing:** 
    - **Time:** $C_{t-1}$ predicts $H_t$.
    - **Depth:** $C_{t, <k}$ predicts $C_{t, k}$.
