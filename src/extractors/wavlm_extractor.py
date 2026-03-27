import librosa
import torch
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from .base_extractor import BaseExtractor

class WavLMExtractor(BaseExtractor):
    """
    Implementation of the WavLM feature extractor.
    Reference: notebooks/wavlm_Embeddings_extraction.ipynb
    """
    def load_model(self):
        """Initializes the processor and model from HuggingFace."""
        processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name).to(self.device)
        model.eval()
        return {"processor": processor, "model": model}

    def extract_one(self, audio_path):
        """
        Processes a single audio clip and returns a mean-pooled embedding.
        Uses librosa for sample rate transformation and WavLM for feature extraction.
        """
        try:
            # 1. Load and resample audio
            audio, _ = librosa.load(audio_path, sr=self.TARGET_SR)

            # 2. Process with WavLM processor and move to device
            inputs = self.model_data["processor"](
                audio, sampling_rate=self.TARGET_SR, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 3. Model Inference (No Gradient tracking needed)
            with torch.no_grad():
                outputs = self.model_data["model"](**inputs)

            # 4. Mean Pooling: (batch_size, sequence_length, hidden_size) -> (hidden_size)
            # Notebook uses dim=1 to average across the time dimension
            embedding = outputs.last_hidden_state.mean(dim=1)
            
            # 5. Bring back to CPU and convert to NumPy
            return embedding.squeeze().cpu().numpy()

        except Exception as e:
            # We print the error but return None so the batch doesn't crash
            print(f"[WavLMExtractor] Error processing {audio_path}: {e}")
            return None
