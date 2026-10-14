import uuid
from django.db import models
from apps.voice.models import VoiceProfile


class EnrollmentSession(models.Model):
    voice_profile = models.ForeignKey(
        VoiceProfile,
        on_delete=models.CASCADE,
        related_name='enrollment_session'
    )

    session_id = models.UUIDField(
        default=uuid.uuid4,
        unique=True,
        editable=False
    )

    free_speech_target = models.PositiveBigIntegerField(default=10)
    pangram_speech_target = models.PositiveBigIntegerField(default=10)

    is_complete = models.BooleanField(
        default=False
    )

    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f'{self.voice_profile.user.username}'