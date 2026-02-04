import torch
import torchaudio
import soundfile as sf
from transformers import MimiModel
from pathlib import Path
import random

# Monkeypatch
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from speechbrain.inference.speaker import EncoderClassifier
from model import ParrotMoshi

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
SAMPLE_RATE = 24000
WEIGHTS_PATH = "parrot_moshi_weights.pt"

def load_audio(path):
    wav, sr = sf.read(path)
    if wav.ndim == 1: wav = wav[None, :] 
    else: wav = wav.T
    return torch.from_numpy(wav).float(), sr

def main():
    print(f"Testing ParrotMoshi on {DEVICE}...")

    # 1. Load Models
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(DEVICE).eval()
    speaker_encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        run_opts={"device": DEVICE}
    )
    
    model = ParrotMoshi().to(DEVICE)
    if Path(WEIGHTS_PATH).exists():
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        print(f"Loaded weights from {WEIGHTS_PATH}")
    else:
        print("No weights found!")
        return
    model.eval()

    # 2. Setup Test Pair (Content: Voice A S1 -> Style: Voice B S2 -> GT: Voice B S1)
    dataset_path = Path("parrot_dataset")
    voices = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    # Let's pick two voices from the training set (to check if it learned training data)
    # We'll use Natasha and William (if they exist)
    v1 = voices[0] # Usually Natasha
    v2 = voices[1] # Usually William
    
    s_id = "sentence_005" # Let's use a sentence that isn't the first one
    ext = ".mp3" # Or .wav
    
    src_path = v1 / f"{s_id}{ext}"
    ref_path = v2 / f"sentence_001{ext}" # Any sentence for style
    gt_path = v2 / f"{s_id}{ext}"        # SAME sentence ID as source, but in target voice
    
    if not gt_path.exists():
        # Fallback to whatever is available
        print(f"Warning: {gt_path} not found, using random.")
        return

    print(f"Source Content: {src_path}")
    print(f"Target Style:   {ref_path}")
    print(f"Comparing vs:   {gt_path} (Ground Truth)")
    
    # 3. Process
    src_wav, _ = load_audio(src_path)
    src_tokens = mimi.encode(src_wav.unsqueeze(1).to(DEVICE)).audio_codes.transpose(1, 2)
    
    ref_wav, _ = load_audio(ref_path)
    resampler = torchaudio.transforms.Resample(SAMPLE_RATE, 16000).to(DEVICE)
    spk_emb = speaker_encoder.encode_batch(resampler(ref_wav.to(DEVICE)).squeeze(1)).squeeze(1)
    
    gt_wav, _ = load_audio(gt_path)
    tgt_tokens_gt = mimi.encode(gt_wav.unsqueeze(1).to(DEVICE)).audio_codes.transpose(1, 2)
    
    # 4. Generate with Adaptive Sampling
    print("\nGenerating with Adaptive Temperature (C0-C3: 0.2, C4-C31: 0.8)...")
    max_len = tgt_tokens_gt.shape[1]
    
    # Create Schedule
    temps = [0.2] * 4 + [0.8] * (32 - 4)
    generated_tokens = model.generate(src_tokens, spk_emb, max_len=max_len, temperature=temps, top_k=50)
    
    # 5. DIAGNOSIS
    print("\n--- Token Match Diagnosis (First 4 Codebooks) ---")
    limit = min(generated_tokens.shape[1], tgt_tokens_gt.shape[1], 15)
    matches = 0
    for t in range(limit):
        gen = generated_tokens[0, t, :4].cpu().numpy()
        gt = tgt_tokens_gt[0, t, :4].cpu().numpy()
        match = (gen == gt).all()
        if match: matches += 1
        print(f"Step {t:02d}: GT {gt} | GEN {gen} | Match: {match}")
    
    print(f"\nSummary: {matches}/{limit} frames matched perfectly in first 4 codebooks.")

    # 6. Save
    decoded = mimi.decode(generated_tokens.transpose(1, 2)).audio_values
    out_path = "test_moshi_output.wav"
    sf.write(out_path, decoded.squeeze().detach().cpu().numpy(), 24000)
    print(f"Saved generated audio to {out_path}")

if __name__ == "__main__":
    main()