from django.db import models
from .voice_profile import VoiceProfile

class VoiceEmbedding(models.Model):

    voice_profile = models.ForeignKey(
        VoiceProfile,
        on_delete=models.CASCADE,
        related_name='embeddings'
    )

    embedding = models.JSONField()

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f'{self.voice_profile.user.username} - Sample {self.sample_number}'
    