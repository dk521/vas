from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .serializers import VoiceEnrollmentSerializer, VoiceAuthenticationSerializer, PangramSerializer
from .services import VoiceEnrollmentService
from .services import VoiceAuthenticationService
from .services import PangramService
from .models import EnrollmentSession

class VoiceEnrollmentView(APIView):
    permission_classes = [IsAuthenticated] 

    def post(self, request):
        serializer = VoiceEnrollmentSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        enrollment = VoiceEnrollmentService.enroll(
            request.user,
            serializer.validated_data['audio_file'],
            serializer.validated_data['prompt_type'],
            serializer.validated_data.get('pangram_id'),
            serializer.validated_data.get('device', ''),
        )

        return Response({"message": "Audio file saved successfully.", **enrollment}, status=201)


class VoiceEnrollmentStatusView(APIView):
    """Return the signed-in volunteer's progress through voice enrollment."""

    permission_classes = [IsAuthenticated]

    def get(self, request):
        session = EnrollmentSession.objects.filter(
            voice_profile=request.user.voice_profile
        ).order_by('-started_at').first()
        if not session:
            session = EnrollmentSession.objects.create(voice_profile=request.user.voice_profile)

        free_speech_collected = session.sample.filter(prompt_type='free_speech').count()
        pangram_collected = session.sample.filter(prompt_type='pangram').count()
        is_complete = (
            free_speech_collected >= session.free_speech_target
            and pangram_collected >= session.pangram_speech_target
        )
        if is_complete and not session.is_complete:
            session.is_complete = True
            session.completed_at = timezone.now()
            session.save(update_fields=['is_complete', 'completed_at'])

        return Response({
            "free_speech_collected": free_speech_collected,
            "free_speech_target": session.free_speech_target,
            "pangram_collected": pangram_collected,
            "pangram_target": session.pangram_speech_target,
            "is_complete": is_complete,
        })


class VoiceAuthenticationView(APIView):

    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = VoiceAuthenticationSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            result = VoiceAuthenticationService.authenticate(
                request.user,
                serializer.validated_data['audio_file']
            )
        except ValueError as error:
            return Response({'detail': str(error)}, status=400)

        return Response(result, status=201)


class RandomPangramView(APIView):

    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            count = int(request.query_params.get('count',3))
        except ValueError:
            count = 3

        count = max(1, min(count, 10))

        pangrams =  PangramService.get_random_pangrams(
            user=request.user,
            count=count
        )

        serializer = PangramSerializer(
            pangrams,
            many=True
        )

        return Response({
            'count': len(pangrams),
            'pangrams' : serializer.data
        })
