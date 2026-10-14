from functools import lru_cache
import os


MODEL_SOURCE = os.environ.get(
    "SPEECHBRAIN_MODEL_SOURCE",
    "speechbrain/spkrec-ecapa-voxceleb",
)
MODEL_DIRECTORY = os.environ.get("SPEECHBRAIN_MODEL_DIR", "ml_models/ecapa")


@lru_cache(maxsize=1)
def get_verification_model():
    from speechbrain.inference.speaker import SpeakerRecognition
    from speechbrain.utils.fetching import LocalStrategy

    return SpeakerRecognition.from_hparams(
        source=MODEL_SOURCE,
        savedir=MODEL_DIRECTORY,
        local_strategy=LocalStrategy.COPY,
    )


class EmbeddingService:

    @staticmethod
    def generate(wav_path):
        verification = get_verification_model()
        signal = verification.load_audio(wav_path)
        signal = signal.unsqueeze(0)
        embedding = verification.encode_batch(signal).squeeze().detach().cpu().numpy().tolist()
        return {
            'embedding': embedding,
            'dimension': len(embedding),
            'model': MODEL_SOURCE,
        }
    
