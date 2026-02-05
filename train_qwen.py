import torch
import torch.nn as nn
import torchaudio
import wandb
import soundfile as sf
import random
import time
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from model import ParrotMoshi
from qwen_wrapper import QwenWrapper
from pathlib import Path

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
SAMPLE_RATE = 24000 
BATCH_SIZE = 64
if DEVICE == "mps": BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHTS_PATH = "parrot_qwen_weights.pt"
VOCAB_SIZE = 4096 

def load_audio(path):
    wav, sr = sf.read(path)
    if wav.ndim == 1: wav = wav[None, :] 
    else: wav = wav.T
    return torch.from_numpy(wav).float(), sr

class ParrotDataset(Dataset):
    def __init__(self, root_dir, split="train", val_ratio=0.1, seed=42):
        self.root = Path(root_dir).resolve()
        if not self.root.exists(): raise FileNotFoundError(f"{self.root}")
        all_voices = sorted([d.name.strip() for d in self.root.iterdir() if d.is_dir()])
        rng = random.Random(seed)
        rng.shuffle(all_voices)
        n_val = max(1, int(len(all_voices) * val_ratio))
        val_voices = all_voices[:n_val]
        train_voices = all_voices[n_val:]
        self.active_voices = train_voices if split == "train" else val_voices
        
        self.pairs = []
        for src_v in self.active_voices:
            src_dir = self.root / src_v
            try:
                sentences = sorted([f for f in src_dir.iterdir() if f.suffix.lower() in [".wav", ".mp3"]])
            except: continue
            if not sentences: continue

            for tgt_v in self.active_voices:
                if src_v == tgt_v: continue
                tgt_dir = self.root / tgt_v
                for s_path in sentences:
                    s_name = s_path.name
                    target_path = tgt_dir / s_name
                    ext = s_path.suffix
                    ref_name = f"sentence_001{ext}" if s_name.lower() != f"sentence_001{ext}" else f"sentence_000{ext}"
                    ref_path = tgt_dir / ref_name
                    if target_path.exists() and ref_path.exists():
                        self.pairs.append((str(s_path), str(target_path), str(ref_path)))

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        src_p, tgt_p, ref_p = self.pairs[idx]
        src_wav, _ = load_audio(src_p)
        tgt_wav, _ = load_audio(tgt_p)
        ref_wav, _ = load_audio(ref_p)
        return src_wav, tgt_wav, ref_wav

def collate_fn(batch):
    src_wavs, tgt_wavs, ref_wavs = zip(*batch)
    def pad(wavs):
        wavs_t = [w.t() for w in wavs]
        padded = torch.nn.utils.rnn.pad_sequence(wavs_t, batch_first=True, padding_value=0)
        return padded.transpose(1, 2)
    return pad(src_wavs), pad(tgt_wavs), pad(ref_wavs)

def train():
    wandb.init(project="parrot-qwen", config={"lr": LEARNING_RATE, "batch": BATCH_SIZE})
    
    print("Loading Qwen Models...")
    qwen = QwenWrapper(device=DEVICE)
    
    train_ds = ParrotDataset("parrot_dataset", split="train")
    val_ds = ParrotDataset("parrot_dataset", split="val")
    
    num_workers = 0 if DEVICE == "mps" else 4
    pin_memory = DEVICE == "cuda"

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=num_workers // 2 if num_workers > 0 else 0, pin_memory=pin_memory)
    
    # Auto-detect number of codebooks
    dummy_wav = torch.zeros(1, 1, 24000).to(DEVICE)
    with torch.no_grad():
        dummy_codes = qwen.encode(dummy_wav).audio_codes
        num_codebooks = dummy_codes.shape[1]
        print(f"Detected {num_codebooks} codebooks from Qwen.")

    # Init ParrotMoshi (BF16 for A40)
    model = ParrotMoshi(vocab_size=VOCAB_SIZE, num_codebooks=num_codebooks, speaker_dim=2048).to(DEVICE, dtype=torch.bfloat16)
    
    if Path(WEIGHTS_PATH).exists():
        print("Resuming Qwen model...")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    resampler_16k = torchaudio.transforms.Resample(SAMPLE_RATE, 16000).to(DEVICE) # Keeps float32 audio usually?
    # Audio processing usually stays FP32 until encode/embed. Qwen encode handles conversion.
    
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for src_wav, tgt_wav, ref_wav in train_dl:
            src_wav, tgt_wav, ref_wav = src_wav.to(DEVICE), tgt_wav.to(DEVICE), ref_wav.to(DEVICE)
            
            with torch.no_grad():
                # Resample to 16k on GPU first to avoid CPU bottleneck in tokenizer
                src_wav_16k = resampler_16k(src_wav)
                tgt_wav_16k = resampler_16k(tgt_wav)
                ref_wav_16k = resampler_16k(ref_wav) # Already used for spk_emb
                
                # Qwen Encode (Input 16k)
                src_tokens = qwen.encode(src_wav_16k, sr=16000).audio_codes.transpose(1, 2)
                tgt_tokens = qwen.encode(tgt_wav_16k, sr=16000).audio_codes.transpose(1, 2)
                
                # Speaker Embedding
                spk_emb = qwen.get_speaker_embedding(ref_wav_16k).to(dtype=torch.bfloat16)

            # Forward pass (BF16 model handles BF16 inputs)
            logits = model(src_tokens, tgt_tokens, spk_emb)
            
            # Loss (Logits are BF16, CrossEntropy handles it or auto-casts)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_tokens.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            wandb.log({"qwen_batch_loss": loss.item()})

        avg_train_loss = total_loss / len(train_dl)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src_wav, tgt_wav, ref_wav in val_dl:
                src_wav, tgt_wav, ref_wav = src_wav.to(DEVICE), tgt_wav.to(DEVICE), ref_wav.to(DEVICE)
                
                src_wav_16k = resampler_16k(src_wav)
                tgt_wav_16k = resampler_16k(tgt_wav)
                ref_wav_16k = resampler_16k(ref_wav)

                src_tokens = qwen.encode(src_wav_16k, sr=16000).audio_codes.transpose(1, 2)
                tgt_tokens = qwen.encode(tgt_wav_16k, sr=16000).audio_codes.transpose(1, 2)
                spk_emb = qwen.get_speaker_embedding(ref_wav_16k).to(dtype=torch.bfloat16)

                logits = model(src_tokens, tgt_tokens, spk_emb)
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_tokens.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dl)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
        wandb.log({"epoch": epoch+1, "qwen_loss": avg_train_loss, "qwen_val_loss": avg_val_loss})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"Saved best model.")
            
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"parrot_qwen_epoch_{epoch+1}.pt")

    wandb.finish()

if __name__ == "__main__":
    train()