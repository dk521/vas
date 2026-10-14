from django.urls import path
from .views import VoiceEnrollmentView, VoiceEnrollmentStatusView
from .views import VoiceAuthenticationView, RandomPangramView

urlpatterns = [
    path("enrollment-status/", VoiceEnrollmentStatusView.as_view()),
    path(
        "enroll/",
        VoiceEnrollmentView.as_view(),
    ),
    path(
        "authenticate/",
        VoiceAuthenticationView.as_view(),
    ),
    path(
        'pangrams/',
        RandomPangramView.as_view(),
    ),
]
