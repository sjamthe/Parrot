import torch
import torch.nn as nn
import torchaudio
import wandb
import soundfile as sf
import random
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, random_split

# Monkeypatch for SpeechBrain compatibility with newer Torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from transformers import MimiModel, AutoFeatureExtractor
from speechbrain.inference.speaker import EncoderClassifier
from model import ParrotBrain
from pathlib import Path

# --- CONFIGURATION ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SAMPLE_RATE = 24000
BATCH_SIZE = 8
EPOCHS = 20  # Extended to 20
LEARNING_RATE = 1e-4
WEIGHTS_PATH = "parrot_weights.pt"

# Helper to load audio with soundfile and convert to torch tensor
def load_audio(path):
    wav, sr = sf.read(path)
    if wav.ndim == 1:
        wav = wav[None, :] 
    else:
        wav = wav.T
    return torch.from_numpy(wav).float(), sr

# 1. Load Pre-trained "Translators" (Frozen)
print("Loading Mimi and Speaker Encoder...")
mimi = MimiModel.from_pretrained("kyutai/mimi").to(DEVICE).eval()
feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
speaker_encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": DEVICE})

for param in mimi.parameters(): param.requires_grad = False

# 2. Custom Dataset
class ParrotDataset(Dataset):
    def __init__(self, root_dir, split="train", val_ratio=0.1, seed=42):
        self.root = Path(root_dir)
        all_voices = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        
        # Deterministic shuffle using the provided seed
        rng = random.Random(seed)
        rng.shuffle(all_voices)
        
        # Split voices
        n_val = max(1, int(len(all_voices) * val_ratio))
        val_voices = all_voices[:n_val]
        train_voices = all_voices[n_val:]
        
        if split == "train":
            self.active_voices = train_voices
        else:
            self.active_voices = val_voices
            
        print(f"[{split.upper()}] Loading {len(self.active_voices)} voices...")
        
        # Create parallel pairs: (source_wav, target_wav, ref_wav_for_target_style)
        # We only create pairs WHERE BOTH SOURCE AND TARGET ARE IN THE ACTIVE SET
        # This ensures complete isolation of speakers.
        self.pairs = []
        for src_v in self.active_voices:
            for tgt_v in self.active_voices:
                if src_v == tgt_v: continue # Different voices only
                
                # Find all common sentences
                sentences = sorted(list((self.root / src_v).glob("*.wav")))
                for s_path in sentences:
                    s_name = s_path.name
                    target_path = self.root / tgt_v / s_name
                    # Reference is just a different sentence from the target speaker
                    ref_name = "sentence_001.wav" if s_name != "sentence_001.wav" else "sentence_000.wav"
                    ref_path = self.root / tgt_v / ref_name
                    if target_path.exists() and ref_path.exists():
                        self.pairs.append((s_path, target_path, ref_path))

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

# 3. Training Loop
def train():
    wandb.init(project="parrot-training", config={
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "device": DEVICE,
        "sample_rate": SAMPLE_RATE
    })

    # Instantiate datasets with strict speaker separation
    train_ds = ParrotDataset("parrot_dataset", split="train")
    val_ds = ParrotDataset("parrot_dataset", split="val")
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    print(f"Dataset Split: {len(train_ds)} Training Samples | {len(val_ds)} Validation Samples")
    
    model = ParrotBrain().to(DEVICE)

    # Resume logic
    start_epoch = 0
    if Path(WEIGHTS_PATH).exists():
        print(f"Found existing weights at {WEIGHTS_PATH}. Loading and resuming...")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        print(f"Resuming from initial training.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    prev_avg_val_loss = 0
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        
        # --- Training Loop ---
        for src_wav, tgt_wav, ref_wav in train_dl:
            src_wav, tgt_wav, ref_wav = src_wav.to(DEVICE), tgt_wav.to(DEVICE), ref_wav.to(DEVICE)

            with torch.no_grad():
                src_codes = mimi.encode(src_wav).audio_codes
                tgt_codes = mimi.encode(tgt_wav).audio_codes
                src_tokens = src_codes.transpose(1, 2)
                tgt_tokens = tgt_codes.transpose(1, 2)
                
                if SAMPLE_RATE != 16000:
                    resampler = torchaudio.transforms.Resample(SAMPLE_RATE, 16000).to(DEVICE)
                    ref_wav_16k = resampler(ref_wav)
                else:
                    ref_wav_16k = ref_wav

                spk_emb = speaker_encoder.encode_batch(ref_wav_16k.squeeze(1)).squeeze(1)

            predictions = model(src_tokens, spk_emb)
            
            min_len = min(predictions.shape[1], tgt_tokens.shape[1])
            predictions = predictions[:, :min_len, :, :]
            tgt_tokens = tgt_tokens[:, :min_len, :]
            
            loss = criterion(predictions.reshape(-1, 2048), tgt_tokens.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            wandb.log({"batch_loss": loss.item()})

        avg_train_loss = total_loss / len(train_dl)
        
        # --- Validation Loop ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src_wav, tgt_wav, ref_wav in val_dl:
                src_wav, tgt_wav, ref_wav = src_wav.to(DEVICE), tgt_wav.to(DEVICE), ref_wav.to(DEVICE)
                
                src_codes = mimi.encode(src_wav).audio_codes
                tgt_codes = mimi.encode(tgt_wav).audio_codes
                src_tokens = src_codes.transpose(1, 2)
                tgt_tokens = tgt_codes.transpose(1, 2)

                resampler = torchaudio.transforms.Resample(SAMPLE_RATE, 16000).to(DEVICE)
                ref_wav_16k = resampler(ref_wav)
                spk_emb = speaker_encoder.encode_batch(ref_wav_16k.squeeze(1)).squeeze(1)

                predictions = model(src_tokens, spk_emb)
                min_len = min(predictions.shape[1], tgt_tokens.shape[1])
                predictions = predictions[:, :min_len, :, :]
                tgt_tokens = tgt_tokens[:, :min_len, :]
                
                loss = criterion(predictions.reshape(-1, 2048), tgt_tokens.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dl)
        if avg_val_loss > prev_avg_val_loss and epoch > 0:
            print(f"Early stopping triggered at epoch {epoch+1} as avg_val_loss {avg_val_loss:.4f} is greater than previous avg_val_loss {prev_avg_val_loss:.4f}")
            break
        prev_avg_val_loss = avg_val_loss

        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        wandb.log({"epoch": epoch+1, "loss": avg_train_loss, "val_loss": avg_val_loss})
        
        # Save weights
        torch.save(model.state_dict(), WEIGHTS_PATH)
    
    wandb.finish()

if __name__ == "__main__":
    train()
