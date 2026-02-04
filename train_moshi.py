import torch
import torch.nn as nn
import torchaudio
import wandb
import soundfile as sf
import random
import time
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

# Monkeypatch for SpeechBrain compatibility - MUST be before speechbrain import
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from transformers import MimiModel
from speechbrain.inference.speaker import EncoderClassifier
from model import ParrotMoshi
from pathlib import Path

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")
SAMPLE_RATE = 24000
BATCH_SIZE = 24 # Increased for A40 GPU
if DEVICE == "mps": BATCH_SIZE = 4
print(f"Batch Size: {BATCH_SIZE}")
EPOCHS = 100   # Increased for convergence
LEARNING_RATE = 1e-4
WEIGHTS_PATH = "parrot_moshi_weights.pt"

def load_audio(path):
    # Support mp3 and wav
    wav, sr = sf.read(path)
    if wav.ndim == 1: wav = wav[None, :] 
    else: wav = wav.T
    return torch.from_numpy(wav).float(), sr

class ParrotDataset(Dataset):
    def __init__(self, root_dir, split="train", val_ratio=0.1, seed=42):
        self.root = Path(root_dir).resolve() # Force absolute path
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        all_voices = sorted([d.name.strip() for d in self.root.iterdir() if d.is_dir()])
        rng = random.Random(seed)
        rng.shuffle(all_voices)
        n_val = max(1, int(len(all_voices) * val_ratio))
        val_voices = all_voices[:n_val]
        train_voices = all_voices[n_val:]
        self.active_voices = train_voices if split == "train" else val_voices
        
        print(f"[{split.upper()}] Scanning {len(self.active_voices)} voices in {self.root}...")
        
        self.pairs = []
        for src_v in self.active_voices:
            src_dir = self.root / src_v
            try:
                all_files = list(src_dir.iterdir())
                # Look for wav or mp3
                sentences = sorted([f for f in all_files if f.suffix.lower() in [".wav", ".mp3"]])
            except Exception as e:
                print(f"Error accessing {src_dir}: {e}")
                continue

            if not sentences:
                print(f"Warning: No audio files found in {src_dir} (Files present: {len(all_files)})")
                continue

            for tgt_v in self.active_voices:
                if src_v == tgt_v: continue
                tgt_dir = self.root / tgt_v
                
                for s_path in sentences:
                    s_name = s_path.name
                    target_path = tgt_dir / s_name
                    
                    # Ref logic
                    # Try to find ref with same extension
                    ext = s_path.suffix
                    ref_name = f"sentence_001{ext}" if s_name.lower() != f"sentence_001{ext}" else f"sentence_000{ext}"
                    ref_path = tgt_dir / ref_name
                    
                    if target_path.exists() and ref_path.exists():
                        self.pairs.append((str(s_path), str(target_path), str(ref_path)))
                        
        print(f"[{split.upper()}] Found {len(self.pairs)} pairs.")

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
    wandb.init(project="parrot-moshi", config={"lr": LEARNING_RATE, "batch": BATCH_SIZE})
    
    print("Loading Mimi/SpeakerEncoder...")
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(DEVICE).eval()
    speaker_encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": DEVICE})
    for p in mimi.parameters(): p.requires_grad = False
    
    train_ds = ParrotDataset("parrot_dataset", split="train")
    val_ds = ParrotDataset("parrot_dataset", split="val")
    
    # Optimized DataLoader
    num_workers = 0 if DEVICE == "mps" else 4
    pin_memory = DEVICE == "cuda"

    train_dl = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers // 2 if num_workers > 0 else 0,
        pin_memory=True
    )
    
    print(f"Moshi Training on {len(train_ds)} samples. Validation on {len(val_ds)} samples.")
    
    model = ParrotMoshi().to(DEVICE)
    if Path(WEIGHTS_PATH).exists():
        print("Model weights file found. Resuming Moshi model training ...")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    
    # Initialize Resampler once on GPU
    resampler_16k = torchaudio.transforms.Resample(SAMPLE_RATE, 16000).to(DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for src_wav, tgt_wav, ref_wav in train_dl:
            # Move to GPU immediately
            src_wav, tgt_wav, ref_wav = src_wav.to(DEVICE), tgt_wav.to(DEVICE), ref_wav.to(DEVICE)
            
            with torch.no_grad():
                # Mimi Encode happens on GPU because inputs are on GPU
                src_tokens = mimi.encode(src_wav).audio_codes.transpose(1, 2) # [B, T, 32]
                tgt_tokens = mimi.encode(tgt_wav).audio_codes.transpose(1, 2) # [B, T, 32]
                
                ref_wav_16k = resampler_16k(ref_wav)
                spk_emb = speaker_encoder.encode_batch(ref_wav_16k.squeeze(1)).squeeze(1)

            logits = model(src_tokens, tgt_tokens, spk_emb) # [B, T, 32, Vocab]
            
            # Loss: Flatten everything
            loss = criterion(logits.reshape(-1, 2048), tgt_tokens.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            wandb.log({"moshi_batch_loss": loss.item()})

        avg_train_loss = total_loss / len(train_dl)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src_wav, tgt_wav, ref_wav in val_dl:
                src_wav, tgt_wav, ref_wav = src_wav.to(DEVICE), tgt_wav.to(DEVICE), ref_wav.to(DEVICE)
                
                src_tokens = mimi.encode(src_wav).audio_codes.transpose(1, 2)
                tgt_tokens = mimi.encode(tgt_wav).audio_codes.transpose(1, 2)
                
                ref_wav_16k = resampler_16k(ref_wav)
                spk_emb = speaker_encoder.encode_batch(ref_wav_16k.squeeze(1)).squeeze(1)

                logits = model(src_tokens, tgt_tokens, spk_emb)
                loss = criterion(logits.reshape(-1, 2048), tgt_tokens.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dl)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"[{timestamp}] Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        wandb.log({"epoch": epoch+1, "moshi_loss": avg_train_loss, "moshi_val_loss": avg_val_loss})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        else:
            print(f"Validation loss did not improve (Best: {best_val_loss:.4f})")

        torch.save(model.state_dict(), WEIGHTS_PATH)
        print(f"Saved best model (Val Loss: {best_val_loss:.4f})")

        # Periodic Save
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"parrot_moshi_epoch_{epoch+1}.pt")
            print(f"Saved checkpoint: parrot_moshi_epoch_{epoch+1}.pt")

    wandb.finish()

if __name__ == "__main__":
    train()