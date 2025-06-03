from fastapi import APIRouter, File, UploadFile, Form
from backend.models.stt_model import  transcribe_audio_file
from backend.models.process_text import process_text_input
from backend.models.Tts_model import TTS2
import sys
import os

# Ajouter le répertoire racine du projet au sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
chatbot_router = APIRouter() 

@chatbot_router.post ("/text")
def process_text_route(input_text: str = Form(...), inscrit: str = Form("non_inscrit")):
    return process_text_input(input_text, "Texte", inscrit)

@chatbot_router.post("/audio")
async def process_audio_route(file: UploadFile = File(...), inscrit: str = Form("non_inscrit")):
    file_path = f"{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    transcription = transcribe_audio_file(file_path)
    response = process_text_input(transcription, "Audio", inscrit)

    # Génère l'audio TTS
    output_path=TTS2(response["en"], lang="en", output_path="output_audio.wav")

    return {
        "transcription": transcription,
        "response": response["en"],
        #"translated_response": response ["translated_response"] ,
        "audio":output_path,
        "audio_url": "/chatbot/audio_file"
    }


from fastapi.responses import FileResponse
@chatbot_router.get("/audio_file")
def get_audio_file():
    return FileResponse("output_audio.wav", media_type="audio/wav", filename="response.wav")