from django.db import models
from django.contrib.auth.models import User

class VoiceProfile(models.Model):
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='voice_profile'
    )

    is_enrolled = models.BooleanField(default=False)

    model_version = models.CharField(max_length=50, default='ecapa-v1')

    num_samples = models.PositiveBigIntegerField(default=0)

    enrolled_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f'{self.user.username} - Voice Profile'

    

