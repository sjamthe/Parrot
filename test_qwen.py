import torch
import torchaudio
import soundfile as sf
from transformers import AutoModel
from model import ParrotMoshi
from qwen_wrapper import QwenWrapper
from pathlib import Path
import random

# Monkeypatch
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
SAMPLE_RATE = 24000
WEIGHTS_PATH = "parrot_qwen_weights.pt"
VOCAB_SIZE = 4096

def load_audio(path):
    wav, sr = sf.read(path)
    if wav.ndim == 1: wav = wav[None, :] 
    else: wav = wav.T
    return torch.from_numpy(wav).float(), sr

def main():
    print(f"Testing ParrotQwen on {DEVICE}...")

    # 1. Load Models
    qwen = QwenWrapper(device=DEVICE)
    
    # Auto-detect number of codebooks
    dummy_wav = torch.zeros(1, 1, 24000).to(DEVICE) 
    with torch.no_grad():
        num_codebooks = qwen.encode(dummy_wav).audio_codes.shape[1]
        print(f"Detected {num_codebooks} codebooks.")

    # Model in BF16
    model = ParrotMoshi(vocab_size=VOCAB_SIZE, num_codebooks=num_codebooks, speaker_dim=2048).to(DEVICE, dtype=torch.bfloat16)
    
    if Path(WEIGHTS_PATH).exists():
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        print("Loaded weights.")
    else:
        print("No weights found!")
        return
    model.eval()

    # 2. Setup Test Pair
    dataset_path = Path("parrot_dataset")
    voices = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    v1 = voices[0] 
    v2 = voices[1] 
    
    s_id = "sentence_005" 
    #ext = ".mp3" # Or .wav
    for ext in [".mp3", ".wav"]:
        if (v1 / f"{s_id}{ext}").exists(): break
    
    src_path = v1 / f"{s_id}{ext}"
    ref_path = v2 / f"sentence_001{ext}" # Any sentence for style
    gt_path = v2 / f"{s_id}{ext}"        # SAME sentence ID as source, but in target voice
    
    if not gt_path.exists():
        print(f"Warning: {gt_path} not found, using random.")
        # Fallback to whatever is available
        gt_path = src_path

    print(f"Source Content: {src_path}")
    print(f"Target Style:   {ref_path}")
    print(f"Comparing vs:   {gt_path} (Ground Truth)")
    
    # 3. Process
    src_wav, _ = load_audio(src_path)
    src_wav = src_wav.to(DEVICE)
    
    ref_wav, _ = load_audio(ref_path)
    ref_wav = ref_wav.to(DEVICE)
    
    gt_wav, _ = load_audio(gt_path)
    gt_wav = gt_wav.to(DEVICE)
    
    with torch.no_grad():
        src_tokens = qwen.encode(src_wav.unsqueeze(1)).audio_codes.transpose(1, 2)
        tgt_tokens_gt = qwen.encode(gt_wav.unsqueeze(1)).audio_codes.transpose(1, 2)
        
        # New Speaker Embedding
        spk_emb = qwen.get_speaker_embedding(ref_wav.unsqueeze(1))
        
        print("\nGenerating...")
        max_len = tgt_tokens_gt.shape[1]
        
        # Adaptive Temp
        temps = [0.2] * 4 + [0.8] * (num_codebooks - 4)
        generated_tokens = model.generate(src_tokens, spk_emb, max_len=max_len, temperature=temps, top_k=50)
        
        print("\n--- Token Match Diagnosis (First 4 Codebooks) ---")
        limit = min(generated_tokens.shape[1], tgt_tokens_gt.shape[1], 15)
        matches = 0
        for t in range(limit):
            gen = generated_tokens[0, t, :4].cpu().numpy()
            gt = tgt_tokens_gt[0, t, :4].cpu().numpy()
            match = (gen == gt).all()
            if match: matches += 1
            print(f"Step {t:02d}: GT {gt} | GEN {gen} | Match: {match}")
        
        print(f"\nSummary: {matches}/{limit} matched.")

        # Decode
        decoded = qwen.decode(generated_tokens.transpose(1, 2)).audio_values
        
    out_path = "test_qwen_output.wav"
    sf.write(out_path, decoded.squeeze().float().cpu().numpy(), 24000) # .float() before numpy
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
