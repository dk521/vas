from django.db import models

from apps.voice.models.enrollment_session import EnrollmentSession
from apps.voice.models.pangrams import Pangram
from apps.voice.models.voice_profile import VoiceProfile


class EnrollmentSample(models.Model):

    class PromptType(models.TextChoices):
        FREE_SPEECH = 'free_speech', 'Free Speech'
        PANGRAM = 'pangram', 'Pangram'

    voice_profile = models.ForeignKey(
        VoiceProfile,
        on_delete=models.CASCADE,
        related_name='enrollment_sample'
    )

    enrollment_session = models.ForeignKey(
        EnrollmentSession,
        on_delete=models.CASCADE,
        related_name='sample'
    )

    prompt_type = models.CharField(
        max_length = 20,
        choices=PromptType.choices
    )

    pangram = models.ForeignKey(
        Pangram,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='enrollment_sample'
    )

    audio_file = models.FileField(
        upload_to='enrollment_samples'
    )

    embedding = models.JSONField()
    duration_seconds = models.FloatField()
    sample_rate = models.PositiveIntegerField()

    model_version = models.CharField(max_length=100)
    device = models.CharField(max_length=200, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.voice_profile.user.username} - {self.prompt_type}'