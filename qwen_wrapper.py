import torch
import torch.nn as nn
from qwen_tts import Qwen3TTSTokenizer
import numpy as np

class QwenOutput:
    def __init__(self, codes):
        self.audio_codes = codes

class QwenWrapper(nn.Module):
    def __init__(self, model_id="Qwen/Qwen3-TTS-Tokenizer-12Hz", device=None):
        super().__init__()
        
        # Determine device and attn_implementation (matching user's demo)
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda:0"
                attn_implementation="flash_attention_2"
            elif torch.backends.mps.is_available():
                self.device = "mps"
                attn_implementation="sdpa"
            else:
                self.device = "cpu"
                attn_implementation="flash_attention_2" # Default to FA2 if CPU supports or fallback? 
                # Actually usually 'eager' or 'sdpa' for CPU. 
                # But sticking to user's logic:
                attn_implementation="sdpa" # Safer default
        else:
            self.device = device
            if "cuda" in device:
                attn_implementation="flash_attention_2"
            else:
                attn_implementation="sdpa"

        print(f"Loading Qwen3-TTS Tokenizer from {model_id} on {self.device} with {attn_implementation}...")
        
        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
            model_id,
            device_map=self.device,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )
        
    def encode(self, wav_tensor):
        """
        wav_tensor: [B, C, T] or list of waveforms
        Returns: QwenOutput with .audio_codes of shape [B, 32, T]
        """
        if torch.is_tensor(wav_tensor):
            # [B, 1, T] -> list of [T]
            audios = [w.squeeze().cpu().numpy() for w in wav_tensor]
        else:
            audios = wav_tensor

        with torch.no_grad():
            outputs = self.tokenizer.encode(audios, sr=24000, return_dict=True)
            codes = outputs.audio_codes 
            
            # If list of tensors, pad sequence and stack
            if isinstance(codes, list):
                from torch.nn.utils.rnn import pad_sequence
                # codes is list of [T, C] tensors
                # Pad to [B, T, C]
                codes = pad_sequence(codes, batch_first=True, padding_value=0)
            
            # Qwen returns [B, T, C]. Mimi expects [B, C, T].
            # Always transpose last two dims.
            codes = codes.transpose(1, 2)
                
            return QwenOutput(codes)

    def decode(self, codes):
        # [B, 32, T] -> [B, T, 32]
        if codes.shape[1] == 32:
            codes = codes.transpose(1, 2)
            
        with torch.no_grad():
            wavs, sr = self.tokenizer.decode(codes)
            wav_tensors = []
            for w in wavs:
                t = torch.from_numpy(w).float()
                if t.ndim == 1: t = t.unsqueeze(0)
                wav_tensors.append(t)
            
            wav_tensor = torch.stack(wav_tensors).to(self.device)
            
            class AudioOutput:
                def __init__(self, v): self.audio_values = v
            return AudioOutput(wav_tensor)
