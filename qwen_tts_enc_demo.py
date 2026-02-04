import torch
import soundfile as sf
from qwen_tts import Qwen3TTSTokenizer, Qwen3TTSModel

if torch.cuda.is_available():
    device = "cuda:0"
    attn_implementation="flash_attention_2"
elif torch.backends.mps.is_available():
    device = "mps"
    attn_implementation="sdpa"
else:
    device = "cpu"
    attn_implementation="flash_attention_2"

ref_audio = "parrot_dataset/en-GB-ThomasNeural/sentence_000.wav"
ref_text = "The quick brown fox jumps over the lazy dog."

# test tokenizer/encoder, decoder
tokenizer = Qwen3TTSTokenizer.from_pretrained(
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation=attn_implementation,
)

# Encode audio from a URL (or local path)
if 0:
    enc = tokenizer.encode(ref_audio)
    # print size: enc['audio_codes'][0].shape torch.Size([45, 16])

    # Decode codes back into waveforms
    wavs, sr = tokenizer.decode(enc)
    sf.write("decode_output.wav", wavs[0], sr)

#example or voice prompt
MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

model = Qwen3TTSModel.from_pretrained(
    MODEL_PATH,
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation=attn_implementation,
)

# Create a prompt
prompt_items = model.create_voice_clone_prompt(
    ref_audio=ref_audio,
    ref_text=ref_text,
    x_vector_only_mode=False,
)
# shate prompt_items[0].ref_code.shape torch.Size([45, 16])
# prompt_items[0]. ref_spk_embedding.shape torch.Size([2048])
print(prompt_items)