import torch
import torch.nn as nn
from qwen_tts import Qwen3TTSTokenizer, Qwen3TTSModel
import numpy as np

class QwenOutput:
    def __init__(self, codes):
        self.audio_codes = codes

class QwenWrapper(nn.Module):
    def __init__(self, model_id="Qwen/Qwen3-TTS-Tokenizer-12Hz", base_model_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base", device=None):
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
                attn_implementation="sdpa" # Safer default
        else:
            self.device = device
            if "cuda" in device:
                attn_implementation="flash_attention_2"
            else:
                attn_implementation="sdpa"

        print(f"Loading Qwen3-TTS Tokenizer from {model_id} on {self.device}...")
        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
            model_id,
            device_map=self.device,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )
        
        print(f"Loading Qwen3-TTS Base Model (for embeddings) from {base_model_id}...")
        self.model = Qwen3TTSModel.from_pretrained(
            base_model_id,
            device_map=self.device,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )
        
    def get_speaker_embedding(self, wav_tensor, text="dummy text"):
        """
        wav_tensor: [B, 1, T] or list of [T]. 
        Returns: [B, 2048]
        """
        # Note: create_voice_clone_prompt takes path or tensor? 
        # User demo used path: ref_audio="path.wav".
        # Doc string says ref_audio: Union[str, np.ndarray, List].
        # We need to handle batching? 
        # create_voice_clone_prompt seems to handle single input or list.
        
        if torch.is_tensor(wav_tensor):
            audios = [w.squeeze().cpu().numpy() for w in wav_tensor]
        else:
            audios = wav_tensor # List of numpy or paths
            
        # We process one by one or batch if supported?
        # The demo output: prompt_items[0].ref_spk_embedding -> [2048]
        # Let's try batch call.
        
        embeddings = []
        for audio in audios:
            # For numpy input, Qwen expects tuple (audio, sr)
            # We assume 24000 as per our pipeline
            audio_input = (audio, 24000)
            
            # We provide a dummy text because it's required but we only want embedding
            prompt = self.model.create_voice_clone_prompt(
                ref_audio=audio_input,
                ref_text=text,
                x_vector_only_mode=False 
            )
            # prompt is List[PromptItem]
            emb = prompt[0].ref_spk_embedding # [2048] Tensor on device?
            embeddings.append(emb)
            
        return torch.stack(embeddings).to(self.device)

    def encode(self, wav_tensor, sr=24000):
        """
        wav_tensor: [B, C, T] or list of waveforms
        sr: Sampling rate of input audio (Default 24000)
        Returns: QwenOutput with .audio_codes of shape [B, 32, T]
        """
        if torch.is_tensor(wav_tensor):
            # [B, 1, T] -> list of [T]
            audios = [w.squeeze().cpu().numpy() for w in wav_tensor]
        else:
            audios = wav_tensor

        with torch.no_grad():
            outputs = self.tokenizer.encode(audios, sr=sr, return_dict=True)
            codes = outputs.audio_codes 
            
            # If list of tensors, pad sequence and stack
            if isinstance(codes, list):
                from torch.nn.utils.rnn import pad_sequence
                # codes is list of [T, C] tensors
                # Pad to [B, T, C]
                codes = pad_sequence(codes, batch_first=True, padding_value=0)
            
            # Qwen returns [B, T, C]. 
            # We want to mimic Mimi's [B, C, T] format so the training script 
            # (which does .transpose(1,2)) works consistently.
            codes = codes.transpose(1, 2) # [B, C, T]
                
            return QwenOutput(codes)

    def decode(self, codes):
        # Input is [B, K, T] (Mimi-like)
        # Qwen expects [B, T, K]
        codes = codes.transpose(1, 2)
            
        with torch.no_grad():
            # Pass as dict to satisfy Qwen3TTSTokenizer requirements
            # audio_codes expected shape: [B, T, K]
            wavs, sr = self.tokenizer.decode({'audio_codes': codes})
            
            wav_tensors = []
            for w in wavs:
                t = torch.from_numpy(w).float()
                if t.ndim == 1: t = t.unsqueeze(0)
                wav_tensors.append(t)
            
            wav_tensor = torch.stack(wav_tensors).to(self.device)
            
            class AudioOutput:
                def __init__(self, v): self.audio_values = v
            return AudioOutput(wav_tensor)
