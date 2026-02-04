import random
from pathlib import Path

def check_split(root_dir="parrot_dataset", seed=42, val_ratio=0.1):
    root = Path(root_dir)
    if not root.exists():
        print(f"Error: {root} not found.")
        return

    # Get all voices exactly as the training script does
    all_voices = sorted([d.name.strip() for d in root.iterdir() if d.is_dir()])
    
    # Shuffle exactly as training script does
    rng = random.Random(seed)
    rng.shuffle(all_voices)
    
    # Split
    n_val = max(1, int(len(all_voices) * val_ratio))
    val_voices = all_voices[:n_val]
    train_voices = all_voices[n_val:]
    
    print(f"--- Data Split (Seed {seed}) ---")
    print(f"Total Voices: {len(all_voices)}")
    print(f"Train Voices ({len(train_voices)}):")
    for v in sorted(train_voices):
        print(f"  - {v}")
        
    print(f"\nValidation Voices ({len(val_voices)}):")
    for v in sorted(val_voices):
        print(f"  - {v}")

if __name__ == "__main__":
    check_split()

