import torch
import torchaudio
import soundfile as sf
from transformers import MimiModel
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def load_audio(path, target_sr=24000):
    wav, sr = sf.read(path)
    if wav.ndim == 1:
        wav = wav[None, :] 
    else:
        wav = wav.T 
    wav_tensor = torch.from_numpy(wav).float()
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        wav_tensor = resampler(wav_tensor)
    return wav_tensor

def check_reconstruction():
    print(f"Testing Mimi reconstruction (4/32 codebooks) on {DEVICE}...")
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(DEVICE).eval()
    
    # Pick a random sample from your dataset
    dataset_path = Path("parrot_dataset")
    samples = list(dataset_path.glob("**/*.wav"))
    if not samples:
        print("No samples found in parrot_dataset!")
        return
    
    test_file = samples[0]
    print(f"Using sample: {test_file}")
    
    wav = load_audio(test_file).to(DEVICE).unsqueeze(1) # [1, 1, T]
    
    with torch.no_grad():
        # 1. Encode
        encoded = mimi.encode(wav)
        codes = encoded.audio_codes # [1, 32, T_tokens]
        
        # 2. Keep only 1, Zero the rest
        codes_4 = codes.clone()
        codes_4[:, 1:, :] = 0 
        
        # 3. Decode
        reconstructed_4 = mimi.decode(codes_4).audio_values

    # Save output
    sf.write("mimi_test_1_codebooks.wav", reconstructed_4.squeeze().cpu().numpy(), 24000)
    
    print("File saved:")
    print(" - mimi_test_4_codebooks.wav (Minimal content/intelligibility test)")

if __name__ == "__main__":
    check_reconstruction()
