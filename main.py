import torch
import torchaudio
import soundfile as sf
from transformers import MimiModel, AutoFeatureExtractor

# Monkeypatch for SpeechBrain compatibility - MUST be before speechbrain import
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from speechbrain.inference.speaker import EncoderClassifier
from model import ParrotBrain
from pathlib import Path
import random

# --- CONFIGURATION ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SAMPLE_RATE = 24000
WEIGHTS_PATH = "parrot_weights.pt"

def load_audio(path, target_sr=24000):
    """Load audio using soundfile and convert to tensor [1, T]."""
    wav, sr = sf.read(path)
    if wav.ndim == 1:
        wav = wav[None, :] # (1, T)
    else:
        wav = wav.T # (C, T)
    
    wav_tensor = torch.from_numpy(wav).float()
    
    # Resample if needed (Mimi needs 24k, SpeakerEncoder needs 16k usually, but we handle that separately)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        wav_tensor = resampler(wav_tensor)
        
    return wav_tensor

def main():
    print(f"Running inference on {DEVICE}...")

    # 1. Load Models
    print("Loading models...")
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(DEVICE).eval()
    speaker_encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        run_opts={"device": DEVICE}
    )
    
    model = ParrotBrain().to(DEVICE)
    if Path(WEIGHTS_PATH).exists():
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        print("Loaded trained weights.")
    else:
        print("Warning: No weights found! Running with random weights.")
    model.eval()

    # 2. Select Test Data (Randomly from dataset)
    # We want: Source (Voice A) -> Target (Voice B)
    dataset_path = Path("parrot_dataset")
    voices = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if len(voices) < 2:
        print("Not enough voices to test conversion.")
        return

    src_voice = voices[0]
    tgt_voice = voices[1]
    
    # Use Sentence 001 for content, Sentence 002 (from target voice) for style
    src_audio_path = src_voice / "sentence_001.wav"
    ref_audio_path = tgt_voice / "sentence_002.wav" # Different sentence for style
    output_path = "output_generated.wav"

    print(f"Source Content: {src_audio_path}")
    print(f"Target Style:   {ref_audio_path}")

    # 3. Process Inputs
    # Load and encode Source Content -> Tokens
    src_wav = load_audio(src_audio_path).to(DEVICE)     # [1, T]
    with torch.no_grad():
        src_encoded = mimi.encode(src_wav.unsqueeze(1)) # Mimi expects [B, 1, T]
        # src_encoded.audio_codes is [B, 8, T_tokens] or similar?
        # Check Mimi output structure: usually result.audio_codes is [Batch, NumQ, Len]
        src_tokens = src_encoded.audio_codes
        
        # We use all codebooks
        src_tokens = src_tokens.transpose(1, 2) # [B, T, 32] for ParrotBrain

    # Load and encode Reference Style -> Speaker Embedding
    ref_wav_24k = load_audio(ref_audio_path).to(DEVICE)
    # Resample to 16k for Speaker Encoder
    resampler_16k = torchaudio.transforms.Resample(24000, 16000).to(DEVICE)
    ref_wav_16k = resampler_16k(ref_wav_24k) # [1, T]
    
    with torch.no_grad():
        # EncoderClassifier expects [Batch, Time] (no channel dim if mono)? 
        # But we saw in training it worked with squeeze(1).
        # Training used: ref_wav_16k.squeeze(1) where ref_wav_16k was [B, 1, T]
        # So we pass [1, T] -> squeeze(1) -> [T]? No, batch size is 1.
        # EncoderClassifier expects [B, T].
        spk_emb = speaker_encoder.encode_batch(ref_wav_16k).squeeze(1) # [1, 192]

    # 4. Run Parrot Inference
    print("Generating...")
    with torch.no_grad():
        # src_tokens: [1, T, 8], spk_emb: [1, 192]
        logits = model(src_tokens, spk_emb) # [1, T, 2048, 8]
        
        # Greedy Decoding: Take max prob
        pred_tokens = torch.argmax(logits, dim=2) # [1, T, 32]
        
        # Prepare for Mimi Decoding
        pred_tokens = pred_tokens.transpose(1, 2) # [1, 32, T]
        
        # Decode
        decoded_audio = mimi.decode(pred_tokens).audio_values # [B, 1, T_out]

    # 5. Save Output
    print(f"Saving to {output_path}...")
    audio_cpu = decoded_audio.squeeze().cpu().numpy()
    sf.write(output_path, audio_cpu, SAMPLE_RATE)
    print("Done!")

if __name__ == "__main__":
    main()