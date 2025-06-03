
import sounddevice as sd

from gtts import gTTS
import sounddevice as sd
import soundfile as sf
import os

def TTS2(text, lang="en", output_path="output.mp3"):
    # Générer la parole avec gTTS
    tts = gTTS(text=text, lang=lang, slow=False)
    
    # Sauvegarder l'audio dans un fichier
    tts.save(output_path)
    
    # Lire l'audio en temps réel (avec sounddevice et soundfile)
    # Ouvre le fichier audio et le lit
    data, samplerate = sf.read(output_path)
    sd.play(data, samplerate)
    sd.wait()  # Attend que la lecture soit terminée
    
    return output_path  # Retourne le chemin du fichier audio
