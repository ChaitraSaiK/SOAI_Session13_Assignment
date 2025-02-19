# SmolLM2 Model

The SmolLM2 model is a lightweight transformer-based architecture designed for efficient language modeling. Below is a breakdown of its main components:

## Configuration (SmolLM2Config)

Defines the hyperparameters for the model:

vocab_size: 49152

hidden_size: 576

intermediate_size: 1536

num_hidden_layers: 30

num_attention_heads: 3

max_position_embeddings: 2048

initializer_range: 0.02

rms_norm_eps: 1e-5

## Key Components

- Rotary Positional Embeddings (LlamaRotaryEmbedding)

Computes and caches sinusoidal embeddings for better contextual understanding in attention mechanisms.

- Attention Mechanism (SmolLM2Attention)

Implements multi-head self-attention using:

Query (q_proj), Key (k_proj), and Value (v_proj) projection layers.

Optional FlashAttention support for efficient computation.

- Feedforward Network (SmolLM2MLP)

Uses:

Gated linear unit (gate_proj).

Upsampling and downsampling layers (up_proj and down_proj).

SiLU activation function.

- Layer Normalization (SmolLM2RMSNorm)

Applies Root Mean Square Layer Normalization (RMSNorm) to stabilize training.

- Decoder Layer (SmolLM2DecoderLayer)

Each decoder layer consists of:

Self-attention block.

Feedforward block.

Layer normalizations before and after attention.

Parameter Calculation

Attention Layer Parameters

Each attention layer contains:

Query, Key, and Value projections:

hidden_size * hidden_size per projection → 576 × 576 = 331,776

3 projections → 995,328 parameters

Output projection:

hidden_size * hidden_size → 331,776 parameters

Total per attention layer: 1,327,104 parameters

MLP Layer Parameters

Gate and up projection:

hidden_size × intermediate_size → 576 × 1536 = 884,736

Down projection:

intermediate_size × hidden_size → 1536 × 576 = 884,736

Total per MLP layer: 1,769,472 parameters

Layer Normalization Parameters

One parameter per hidden dimension (hidden_size)

Two normalizations per layer → 576 × 2 = 1,152 parameters

Total Parameters per Decoder Layer

Attention Layer: 1,327,104

MLP Layer: 1,769,472

Layer Normalization: 1,152

Total: 3,097,728 parameters per layer

Total Model Parameters

With 30 decoder layers:

30 × 3,097,728 = 92,931,840 parameters

Additional Components

Word Embeddings:

vocab_size × hidden_size → 49152 × 576 = 28,311,552

Final Layer Norm:

576 parameters

Final Parameter Count

Decoder Layers: 92,931,840

Word Embeddings: 28,311,552

Final Norm: 576

Total SmolLM2 Model Parameters: 121,243,968 (~121M)
