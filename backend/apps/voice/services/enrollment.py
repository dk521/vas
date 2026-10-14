from apps.voice.services.audio import AudioService
from apps.voice.services.embedding import EmbeddingService
from apps.voice.services.enrollment_session import EnrollmentSessionService
from apps.voice.models import EnrollmentSample, Pangram
from django.db import transaction
from django.utils import timezone

class VoiceEnrollmentService:

    @staticmethod
    def enroll(
        user,
        audio_file,
        prompt_type,
        pangram_id=None,
        device=""

    ):
        input_path = AudioService.save_temp_audio(audio_file)
        wav_path = None
        try:
            wav_path = AudioService.convert_to_wav(input_path)
            embedding_result = EmbeddingService.generate(wav_path)

            voice_profile = user.voice_profile
            enrollment_session = EnrollmentSessionService.get_or_create_active_session(user)

            pangram = None

            if prompt_type == 'pangram':
                pangram = Pangram.objects.get(id=pangram_id, is_active=True)

            meta_data = AudioService.get_metadata(wav_path)

            audio_file.seek(0)
            s3_audio_name = AudioService.save_enrollment_audio(
                user=user,
                audio_file=audio_file,
                session_id=enrollment_session.session_id
            )

            with transaction.atomic():
                sample = EnrollmentSample.objects.create(
                    voice_profile=voice_profile,
                    enrollment_session=enrollment_session,
                    prompt_type=prompt_type,
                    pangram=pangram,
                    audio_file=s3_audio_name,
                    embedding=embedding_result['embedding'],
                    duration_seconds=meta_data['duration_seconds'],
                    sample_rate=meta_data['sample_rate'],
                    model_version=embedding_result['model'],
                    device=device
                )

            free_speech_count = enrollment_session.sample.filter(
                prompt_type='free_speech'
            ).count()
            pangram_count = enrollment_session.sample.filter(
                prompt_type='pangram'
            ).count()

            is_complete = (
                free_speech_count >= enrollment_session.free_speech_target
                and pangram_count >= enrollment_session.pangram_speech_target
            )
            if is_complete:
                enrollment_session.is_complete = True
                enrollment_session.completed_at = timezone.now()
                enrollment_session.save(update_fields=['is_complete', 'completed_at'])

            return {
                'sample_id': sample.id,
                'prompt_type': prompt_type,
                'free_speech_collected': free_speech_count,
                'free_speech_target': enrollment_session.free_speech_target,
                'pangram_collected': pangram_count,
                'pangram_target': enrollment_session.pangram_speech_target,
                'is_complete': is_complete,
            }
        finally:
            AudioService.cleanup(input_path, wav_path)
