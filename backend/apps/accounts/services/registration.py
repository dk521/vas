from django.contrib.auth.models import User
from apps.voice.models.voice_profile import VoiceProfile


class UserService:

    @staticmethod
    def register(data):

        user = User.objects.create_user(
            username=data["username"],
            email=data["email"],
            password=data["password"],
        )

        VoiceProfile.objects.create(user=user)

        return user