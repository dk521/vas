from types import SimpleNamespace
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase

from apps.voice.services.audio import AudioService


class AudioServiceTests(SimpleTestCase):
    @patch("apps.voice.services.audio.default_storage.save", side_effect=lambda name, _: name)
    def test_enrollment_audio_uses_a_unique_storage_name_per_sample(self, save):
        user = SimpleNamespace(id=42)
        audio = SimpleUploadedFile("voice-recording.webm", b"audio", content_type="audio/webm")

        first_name = AudioService.save_enrollment_audio(user, audio, "session-1")
        second_name = AudioService.save_enrollment_audio(user, audio, "session-1")

        self.assertNotEqual(first_name, second_name)
        self.assertTrue(first_name.startswith("enrollment/42/session-1/"))
        self.assertTrue(first_name.endswith(".webm"))
        self.assertEqual(save.call_count, 2)
