import random
from apps.voice.models import Pangram, EnrollmentSample
from apps.voice.services.enrollment_session import EnrollmentSessionService


class PangramService:

    @staticmethod
    def get_random_pangrams(user, count=3):

        session = EnrollmentSessionService.get_or_create_active_session(user)

        user_pangram_ids = EnrollmentSample.objects.filter(
            enrollment_session=session, 
            prompt_type=EnrollmentSample.PromptType.PANGRAM,
            pangram__isnull=False,
        ).values_list('pangram_id', flat=True)

        available_pangrams = list(
            Pangram.objects.filter(
                is_active=True,
            ).exclude(id__in=user_pangram_ids)
        )

        count = min(count, len(available_pangrams))

        return random.sample(
            available_pangrams, count
        )