import torch
import torch.nn as nn
import math

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class ParrotMoshi(nn.Module):
    """
    ParrotMoshi: A Full-Duplex style Voice Conversion model.
    Based on Moshi/PersonaPlex architecture (No Text).
    
    Components:
    1. Source Encoder: Processes source audio context.
    2. Temporal Transformer (Main): Autoregressive over Time (T).
    3. Depth Transformer (Depformer): Autoregressive over Codebooks (K=32).
    """
    def __init__(self, vocab_size=2048, hidden_dim=512, speaker_dim=192, 
                 num_codebooks=32, nhead=8, num_layers_temp=6, num_layers_depth=4):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # --- Embeddings ---
        # Shared embeddings for codebooks (or separate? Moshi uses separate or projected)
        # We'll use separate for simplicity/capacity
        self.codebook_embs = nn.ModuleList([
            nn.Embedding(vocab_size, hidden_dim) for _ in range(num_codebooks)
        ])
        
        self.pos_emb = SinusoidalPositionalEmbedding(hidden_dim)
        self.spk_proj = nn.Linear(speaker_dim, hidden_dim)

        # --- 1. Source Encoder ---
        # Encodes the Source Audio (Content Provider)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True, dim_feedforward=2048),
            num_layers=num_layers_temp
        )

        # --- 2. Temporal Transformer (Main) ---
        # Predicts "Frame Latent" H_t given History H_<t
        # Input: Sum of codebook embeddings at step T-1
        self.temporal_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True, dim_feedforward=2048),
            num_layers=num_layers_temp
        )
        
        # --- 3. Depth Transformer (Depformer) ---
        # Predicts C_k given H_t and C_<k
        # We treat H_t as a "context" or "start token" for this small sequence
        self.depth_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True, dim_feedforward=1024),
            num_layers=num_layers_depth
        )
        
        # Output Head
        self.head = nn.Linear(hidden_dim, vocab_size)

    def _fuse_codebooks(self, tokens):
        # tokens: [B, T, 32]
        # Sum embeddings of all codebooks to represent the frame
        x = 0
        for i in range(self.num_codebooks):
            x = x + self.codebook_embs[i](tokens[:, :, i])
        return x

    def forward(self, src_tokens, tgt_tokens, spk_emb):
        """
        Training Forward Pass.
        
        src_tokens: [B, T_s, 32] (Full source)
        tgt_tokens: [B, T_t, 32] (Full target Ground Truth)
        spk_emb: [B, 192]
        """
        B, T_t, K = tgt_tokens.shape
        
        # --- 1. Encode Source ---
        src_emb = self._fuse_codebooks(src_tokens) # [B, T_s, D]
        src_emb = self.pos_emb(src_emb)
        memory = self.encoder(src_emb) # [B, T_s, D]
        
        # --- 2. Temporal Step (Time AR) ---
        # Input at step t is the fused frame at t-1.
        # Shift target right by 1 for teacher forcing.
        # <SOS> is needed. We'll use 0-vector or specific token.
        tgt_fused = self._fuse_codebooks(tgt_tokens) # [B, T_t, D]
        
        # Create SOS token (Learnable or Zero)
        sos_emb = torch.zeros(B, 1, self.hidden_dim, device=tgt_tokens.device)
        # Shift: [SOS, Fused_0, Fused_1, ..., Fused_T-2] -> Predicts H_0 ... H_T-1
        temporal_in = torch.cat([sos_emb, tgt_fused[:, :-1, :]], dim=1)
        temporal_in = self.pos_emb(temporal_in)
        
        # Add Speaker Style
        spk_vec = self.spk_proj(spk_emb).unsqueeze(1)
        temporal_in = temporal_in + spk_vec
        
        # Causal Mask for Time
        time_mask = torch.triu(torch.ones(T_t, T_t, device=tgt_tokens.device) * float('-inf'), diagonal=1)
        
        # Run Temporal Decoder
        # Cross-Attend to Source Memory
        temporal_out = self.temporal_decoder(temporal_in, memory, tgt_mask=time_mask) # [B, T_t, D]
        
        # --- 3. Depth Step (Codebook AR) ---
        # We process ALL time steps in parallel by flattening B*T
        # Input to Depth: Sequence of 32 codebooks.
        # At each 'k', we want to predict C_k given H_t (temporal context) and C_<k.
        
        # Flatten: [B, T, K] -> [B*T, K]
        flat_tgt = tgt_tokens.reshape(B * T_t, K) 
        
        # Embeddings for depth sequence
        # We need to embed each codebook index specific embedding
        # [B*T, K, D]
        depth_embs = []
        for k in range(K):
             depth_embs.append(self.codebook_embs[k](flat_tgt[:, k]))
        depth_seq = torch.stack(depth_embs, dim=1) # [B*T, K, D]
        
        # Shift Right for Depth AR: [SOS, C0, C1, ..., C30] -> Predict [C0, ..., C31]
        # SOS for depth is the Temporal Latent H_t !
        # H_t: [B, T, D] -> [B*T, 1, D]
        temporal_context = temporal_out.reshape(B * T_t, 1, self.hidden_dim)
        
        # Input sequence: [H_t, C0, C1, ... C30] (Length 32)
        # We want output to align with [C0, C1, ... C31].
        # So we feed [H_t, C0...C30] and expect [C0...C31].
        
        depth_in = torch.cat([temporal_context, depth_seq[:, :-1, :]], dim=1) # [B*T, 32, D]
        
        # Positional Embedding for Depth (Since K=32 is a sequence)
        depth_in = self.pos_emb(depth_in) 
        
        # Causal Mask for Depth
        depth_mask = torch.triu(torch.ones(K, K, device=tgt_tokens.device) * float('-inf'), diagonal=1)
        
        # Run Depth Decoder
        # Self-Attention only (no memory needed, or memory could be H_t if we didn't use it as SOS)
        # Here we treat it as "Decoder-only" mode where 'memory' is None or self-referential?
        # Standard nn.TransformerDecoder requires 'memory' for cross-attention.
        # If we use it as GPT (Decoder-only), we usually ignore memory or pass a dummy.
        # OR better: Use H_t as 'memory' and [SOS, C0...C30] as input?
        # Moshi usually conditions Depformer on H_t via Cross-Attention.
        # Let's DO THAT. It's cleaner.
        
        # Revised Depth Strategy:
        # Input: [SOS, C0, ... C30]. 
        # Memory (Cross-Attn): H_t (Expanded to [B*T, 1, D])
        
        depth_seq_shifted = torch.cat([
            torch.zeros_like(depth_seq[:, 0:1, :]), # SOS (Zero)
            depth_seq[:, :-1, :]
        ], dim=1)
        depth_seq_shifted = self.pos_emb(depth_seq_shifted)
        
        depth_out = self.depth_decoder(depth_seq_shifted, temporal_context, tgt_mask=depth_mask) # [B*T, 32, D]
        
        # Predict Logits
        logits = self.head(depth_out) # [B*T, 32, Vocab]
        
        return logits.reshape(B, T_t, K, self.vocab_size)

    @torch.no_grad()
    def generate(self, src_tokens, spk_emb, max_len=500):
        # src_tokens: [1, T_s, 32]
        # spk_emb: [1, 192]
        
        # 1. Encode Source
        src_emb = self._fuse_codebooks(src_tokens)
        src_emb = self.pos_emb(src_emb)
        memory = self.encoder(src_emb)
        
        # 2. Loop
        curr_tgt_tokens = torch.zeros(1, 0, 32, dtype=torch.long, device=src_tokens.device)
        
        # Start with just SOS for Temporal (handled implicitly by empty history loop logic or explicit start)
        # We need an explicit loop because at each T we run the Depth loop.
        
        for t in range(max_len):
            # --- Temporal Step ---
            # Prepare Temporal Input from History
            if t == 0:
                temporal_in = torch.zeros(1, 1, self.hidden_dim, device=src_tokens.device) # SOS
            else:
                # Fused history
                # We only need the last step if using KV cache, but here we re-run full seq (slow but safe)
                fused_hist = self._fuse_codebooks(curr_tgt_tokens)
                sos = torch.zeros(1, 1, self.hidden_dim, device=src_tokens.device)
                temporal_in = torch.cat([sos, fused_hist], dim=1) # [1, T+1, D]
                temporal_in = self.pos_emb(temporal_in)

            # Add Speaker
            temporal_in = temporal_in + self.spk_proj(spk_emb).unsqueeze(1)
            
            # Run Temporal
            temp_out = self.temporal_decoder(temporal_in, memory) # [1, T+1, D]
            current_latent = temp_out[:, -1, :].unsqueeze(1) # [1, 1, D] (H_t)
            
            # --- Depth Step (Loop K=32) ---
            # Generate C0 -> C1 ... -> C31
            curr_codes = [] # List of indices
            
            # Depth Input starts as SOS (Zero)
            depth_input_seq = torch.zeros(1, 1, self.hidden_dim, device=src_tokens.device)
            
            for k in range(self.num_codebooks):
                # Pos Emb
                # We can't easily use the full pos_emb class for step-by-step unless we index it
                # For simplicity in this non-optimized gen, we re-run depth seq
                
                # Input to depth decoder: [1, k+1, D]
                # Cross-Attend to current_latent (H_t)
                
                # Add Pos Emb
                depth_in_pos = self.pos_emb(depth_input_seq)
                
                d_out = self.depth_decoder(depth_in_pos, current_latent)
                logit_k = self.head(d_out[:, -1, :]) # [1, Vocab]
                
                code_k = torch.argmax(logit_k, dim=-1) # Greedy
                curr_codes.append(code_k)
                
                # Append predicted code embedding to input for next k
                if k < self.num_codebooks - 1:
                    next_emb = self.codebook_embs[k+1](code_k).unsqueeze(1) # [1, 1, D]
                    # Note: We should probably embed code_k with embs[k] not k+1?
                    # The model learned: Input C_k predicts C_k+1?
                    # No, Training: Input [SOS, C0] -> Predicts [C0, C1].
                    # So to predict C1, we input C0.
                    # So we embed code_k using embs[k].
                    next_emb = self.codebook_embs[k](code_k).unsqueeze(1)
                    depth_input_seq = torch.cat([depth_input_seq, next_emb], dim=1)
            
            # Stack codes
            stack_codes = torch.stack(curr_codes, dim=1).unsqueeze(0) # [1, 1, 32]
            curr_tgt_tokens = torch.cat([curr_tgt_tokens, stack_codes], dim=1)
            
            # Stop condition? Length check.
            if curr_tgt_tokens.shape[1] >= src_tokens.shape[1] + 10:
                break
                
        return curr_tgt_tokens