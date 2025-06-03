import librosa
import whisper


model = whisper.load_model("tiny")

def transcribe_audio_file(file_path):
    audio, _ = librosa.load(file_path, sr=16000)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    return result.text



