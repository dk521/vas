from apps.voice.models import EnrollmentSession


class EnrollmentSessionService:

    @staticmethod
    def get_or_create_active_session(user):

        session = EnrollmentSession.objects.filter(voice_profile=user.voice_profile, is_complete=False).order_by('-started_at').first()

        if session:
            return session 
        
        return EnrollmentSession.objects.create(voice_profile=user.voice_profile)
    