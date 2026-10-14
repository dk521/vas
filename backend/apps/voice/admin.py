from django.contrib import admin
from .models import VoiceProfile, AuthenticationLog, EnrollmentSample


@admin.register(VoiceProfile)
class VoiceProfileAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "is_enrolled",
        "num_samples",
        "model_version",
        "created_at",
    )

    search_fields = (
        "user__username",
        "user__email",
    )

    list_filter = (
        "is_enrolled",
        "model_version",
    )

    ordering = (
        "-created_at",
    )

@admin.register(AuthenticationLog)
class AuthenticationLogAdmin(admin.ModelAdmin):

    list_display = (
        "user",
        "is_authenticated",
        "similarity_score",
        "latency_ms",
        "failure_reason",
        "created_at",

    )

    list_filter = (
        "is_authenticated",
    )

    search_fields = (
        "user__username",
    )

    ordering = (
        "-created_at",
    )


class PangramAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "text",
        "is_active",
        "created_at",
    )

    list_filter = (
        "is_active",
    )

    search_fields = (
        "text",
    )

    ordering = (
        "-created_at",
    )


class EnrollmentSessionAdmin(admin.ModelAdmin):
    list_display = (
        'session_id',
        'voice_profile__user',
        'free_speech_target',
        'pangram_speech_target',
        'is_complete',
        'started_at',
        'completed_at'
    )

    list_filter = (
        'is_complete'
    )

    search_fields = (
        'voice_profile__user__username',
        'voice_profile__user__email',
        'session_id',
    )

    ordering = (
        '-started_at'
    )





@admin.register(EnrollmentSample)
class EnrollmentSampleAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "voice_profile",
        "prompt_type",
        "pangram",
        "duration_seconds",
        "sample_rate",
        "model_version",
        "created_at",
    )

    list_filter = (
        "prompt_type",
        "model_version",
    )

    search_fields = (
        "voice_profile__user__username",
        "voice_profile__user__email",
    )

    ordering = (
        "-created_at",
    )