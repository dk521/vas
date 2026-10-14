from rest_framework import serializers


class VoiceEnrollmentSerializer(serializers.Serializer):
    audio_file = serializers.FileField()
    prompt_type = serializers.ChoiceField(
        choices=['free_speech', 'pangram']
    )

    pangram_id = serializers.IntegerField(
        required=False,
        allow_null=True
    )
    device = serializers.CharField(
        required=False,
        allow_blank=True
    )

    def validate(self, attrs):
        prompt_type = attrs['prompt_type']
        pangram_id = attrs.get('pangram_id')

        if prompt_type == 'pangram' and not pangram_id:
            raise serializers.ValidationError(
                {'pangram_id': 'required'}
            )
        elif prompt_type == 'free_speech' and pangram_id:
            raise serializers.ValidationError(
                {'pangram_id': 'NA'}
            )

        return attrs

    def validate_audio_file(self, audio_file):
        allowed_content_types = [
            '.wav',
            '.m4a',
            '.mp3',
            '.webm',
        ]
        filename = audio_file.name.lower()
        if not any(filename.endswith(ext) for ext in allowed_content_types):
            raise serializers.ValidationError(
                "Invalid file type. Only .wav, .m4a, .mp3, and .webm files are allowed."
            )
        return audio_file

class VoiceAuthenticationSerializer(serializers.Serializer):
    audio_file = serializers.FileField()


class PangramSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    text = serializers.CharField()
    