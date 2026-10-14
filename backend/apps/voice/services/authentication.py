from apps.voice.services.audio import AudioService
from apps.voice.services.embedding import EmbeddingService
from apps.voice.services.similarity import SimilarityService
from apps.voice.models import EnrollmentSample, VoiceEmbedding, AuthenticationLog
from django.db import transaction
from time import perf_counter


class VoiceAuthenticationService:

    THRESHOLD = 0.75

    @staticmethod
    def authenticate(user, audio_file):
        started_at = perf_counter()
        input_path = AudioService.save_audio(user, audio_file)
        wav_path = None
        try:
            wav_path = AudioService.convert_to_wav(input_path['absolute_path'])
            embedding_result = EmbeddingService.generate(wav_path)
            query_embedding = embedding_result['embedding']

            # New enrollment records embeddings on EnrollmentSample. Keep the
            # VoiceEmbedding fallback so profiles created by the older flow
            # remain usable.
            enrolled_embeddings = list(
                EnrollmentSample.objects.filter(
                    voice_profile=user.voice_profile
                ).values_list('embedding', flat=True)
            )
            if not enrolled_embeddings:
                enrolled_embeddings = list(
                    VoiceEmbedding.objects.filter(
                        voice_profile=user.voice_profile
                    ).values_list('embedding', flat=True)
                )
            if not enrolled_embeddings:
                raise ValueError(
                    "No enrolled voice samples found. Complete enrollment before authenticating."
                )

            similarity_scores = [
                SimilarityService.consine_similarity(query_embedding, enrolled_embedding)
                for enrolled_embedding in enrolled_embeddings
            ]
            best_score = max(similarity_scores)
            is_authenticated = best_score >= VoiceAuthenticationService.THRESHOLD
            latency_ms = (perf_counter() - started_at) * 1000

            with transaction.atomic():
                AuthenticationLog.objects.create(
                    user=user,
                    similarity_score=best_score,
                    threshold=VoiceAuthenticationService.THRESHOLD,
                    is_authenticated=is_authenticated,
                    latency_ms=latency_ms,
                )

            return {
                'is_authenticated': is_authenticated,
                'similarity_score': round(best_score, 4),
                'threshold': VoiceAuthenticationService.THRESHOLD,
            }
        finally:
            AudioService.cleanup(input_path['absolute_path'], wav_path)
