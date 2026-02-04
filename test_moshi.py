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

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
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
        print("Loaded Moshi weights.")
    else:
        print("No weights found! Please train first.")
        return
    model.eval()

    # 2. Pick Test Data
    dataset_path = Path("parrot_dataset")
    voices = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    if len(voices) < 2: return
    
    # Let's try to find a training voice (to see if it learned ANYTHING)
    # and a validation voice (to check generalization).
    # Since we shuffled, we don't know exactly which is which without seed match,
    # but let's just pick index 0 and 1.
    
    src_voice = voices[0]
    tgt_voice = voices[1]
    
    src_path = src_voice / "sentence_001.wav"
    ref_path = tgt_voice / "sentence_002.wav"
    
    print(f"Source Content: {src_path}")
    print(f"Target Style:   {ref_path}")
    
    # 3. Process
    src_wav, _ = load_audio(src_path)
    src_wav = src_wav.to(DEVICE)
    
    ref_wav, _ = load_audio(ref_path)
    ref_wav = ref_wav.to(DEVICE)
    
    with torch.no_grad():
        # Encode Source -> [1, T, 32]
        src_tokens = mimi.encode(src_wav.unsqueeze(1)).audio_codes.transpose(1, 2)
        
        # Encode Ref -> Speaker
        resampler = torchaudio.transforms.Resample(SAMPLE_RATE, 16000).to(DEVICE)
        ref_wav_16k = resampler(ref_wav)
        spk_emb = speaker_encoder.encode_batch(ref_wav_16k.squeeze(1)).squeeze(1)

        # Generate
        print("Generating audio (Double Loop AR)... this will take time...")
        # We generate roughly the same length as source
        max_len = src_tokens.shape[1] + 10
        generated_tokens = model.generate(src_tokens, spk_emb, max_len=max_len) # [1, T_gen, 32]
        
        print(f"Generated {generated_tokens.shape[1]} frames.")

        # Decode
        # Mimi expects [B, 32, T]
        codes_to_decode = generated_tokens.transpose(1, 2)
        decoded = mimi.decode(codes_to_decode).audio_values

    # Save
    out_path = "test_moshi_output.wav"
    sf.write(out_path, decoded.squeeze().cpu().numpy(), 24000)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
