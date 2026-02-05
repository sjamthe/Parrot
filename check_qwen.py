
from qwen_tts import Qwen3TTSModel
import torch

try:
    print("Inspecting Qwen3TTSModel...")
    # model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base") # heavy to load
    # Just checking class attributes if possible or small load
    print(dir(Qwen3TTSModel))
except Exception as e:
    print(e)
