import os
import uuid
import ffmpeg
from django.core.files.storage import default_storage
import soundfile as sf
import tempfile
from django.core.files.base import File



class AudioService:
    @staticmethod
    def save_audio(user, audio_file):
        temp_path = AudioService.save_temp_audio(audio_file)
        return {
            'filename': os.path.basename(temp_path),
            'absolute_path': temp_path,
        }


    @staticmethod
    def get_metadata(wav_path):
        info = sf.info(wav_path)

        return {
            'duration_seconds' : info.duration,
            'sample_rate': info.samplerate
        }


    @staticmethod
    def save_temp_audio(audio_file):
        suffix = os.path.splitext(audio_file.name)[1]

        temp_file = tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        )

        for chunk in audio_file.chunks():
            temp_file.write(chunk)
        temp_file.close()

        return temp_file.name

    @staticmethod
    def convert_to_wav(input_file_path):

        output_path = os.path.splitext(input_file_path)[0] + '.wav'
        try:
            ffmpeg.input(input_file_path).output(output_path, ac=1, ar=16000, format='wav').overwrite_output().run(capture_stdout=True, capture_stderr=True)
            return output_path
        except ffmpeg.Error as e:
            stderr = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f'Failed to convert audio: {stderr}') 


    @staticmethod
    def save_enrollment_audio(
        user,
        audio_file,
        session_id
    ):
        # Browser recordings use a repeated client-side name (for example,
        # ``voice-recording.webm``). Do not use it as an object key: samples in
        # one enrollment session would otherwise overwrite each other in S3.
        extension = os.path.splitext(os.path.basename(audio_file.name))[1].lower()
        unique_filename = f"{uuid.uuid4().hex}{extension}"
        storage_name = os.path.join(
            'enrollment', str(user.id), str(session_id), unique_filename
        )

        return default_storage.save(
            storage_name,
            audio_file
        )


    @staticmethod
    def cleanup(*paths):
        for path in paths:
            if path and os.path.exists(path):
                os.remove(path)

                
    
