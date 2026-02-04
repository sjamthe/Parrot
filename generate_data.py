import asyncio
import edge_tts
import os
import torch
import torchaudio
import soundfile as sf
import json
from pathlib import Path

# Config
OUTPUT_DIR = "parrot_dataset"
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.jsonl")
SAMPLE_RATE = 24000  # Mimi expects 24kHz
CONCURRENCY_LIMIT = 5 # Avoid rate limiting
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def load_lines(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# Load data
try:
    VOICES = load_lines("voices.txt")
    SENTENCES = load_lines("sentences.txt")
except FileNotFoundError as e:
    print(f"Error: Could not find input file - {e}")
    exit(1)

async def generate_single(sem, text, index, voice):
    async with sem:
        folder = Path(OUTPUT_DIR) / voice
        folder.mkdir(parents=True, exist_ok=True)
        
        filename = f"sentence_{index:03d}.wav"
        output_path = folder / f"sentence_{index:03d}.mp3"
        final_path = folder / filename
        
        relative_path = f"{voice}/{filename}"

        try:
            # 1. Generate via Edge TTS
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
            
            # 2. Convert to 24kHz Mono WAV (Mimi Standard)
            # Use soundfile directly to avoid torchaudio backend issues
            data, sr = sf.read(output_path)
            waveform = torch.from_numpy(data).float()
            
            # [T, C] -> [C, T]
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.transpose(0, 1)
            
            waveform = waveform.to(DEVICE)

            if sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE).to(DEVICE)
                waveform = resampler(waveform)
            
            # Ensure Mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                
            # Save from CPU
            torchaudio.save(final_path, waveform.cpu(), SAMPLE_RATE)
            os.remove(output_path) # Clean up mp3
            
            print(f"Generated {voice} | Sentence {index} on {DEVICE}")
            
            return {
                "id": f"{voice}_{index:03d}",
                "sentence_id": index,
                "text": text,
                "voice": voice,
                "file_path": relative_path
            }
        except Exception as e:
            print(f"Failed to generate {voice} | Sentence {index}: {e}")
            return None

async def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = []
    
    # Generate tasks for all combinations
    for i, text in enumerate(SENTENCES):
        for voice in VOICES:
            tasks.append(generate_single(sem, text, i, voice))
            
    results = await asyncio.gather(*tasks)
    
    # Filter None results
    valid_results = [r for r in results if r is not None]
    
    # Write metadata
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        for entry in valid_results:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Finished. Generated {len(valid_results)} files. Metadata saved to {METADATA_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
