import torch
import torch.nn as nn

class ParrotBrain(nn.Module):
    def __init__(self, vocab_size=2048, hidden_dim=512, speaker_dim=192, num_codebooks=32):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size, hidden_dim) for _ in range(num_codebooks)])
        
        # This layer projects the 192 speaker vector to match your 512 transformer dimension
        self.spk_projection = nn.Linear(speaker_dim, hidden_dim)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=6
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, vocab_size) for _ in range(num_codebooks)])

    def forward(self, source_tokens, speaker_emb):
        # source_tokens: [Batch, Time, 32]
        
        x = sum(self.embeddings[i](source_tokens[:, :, i]) for i in range(self.num_codebooks))
        
        # Inject Speaker Identity
        style_vector = self.spk_projection(speaker_emb).unsqueeze(1) # [Batch, 1, 512]
        x = x + style_vector  # Broad-casts the style across all time steps
            
        features = self.transformer(x)
        logits = [head(features) for head in self.heads]
        return torch.stack(logits, dim=-1)