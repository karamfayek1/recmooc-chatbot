
import sys
import os

# Ajouter le répertoire racine du projet au sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)


from backend.models.classification import classify_input
from backend.models.feedback import treat_feedback
from backend.models.assistance import generate_answer
from backend.models.language_detection import detect_language
from backend.models.translation import translate_text_to_en,translate_text_to_src
from backend.models.pydantic_rag import rag_pipeline

def process_text_input(text, input_type="Texte", inscrit="non_inscrit"):
    detected_lang = detect_language(text)
    translated_text = translate_text_to_en(text, detected_lang) if detected_lang != "en" else text
    classe = classify_input(translated_text)
    
    if classe == "recommendation":
        response = "📝 Recommandation basée sur le profil utilisateur." if inscrit == "inscrit" else rag_pipeline(translated_text)
    elif classe == "assistance":
        response = generate_answer(translated_text)
    elif classe == "feedback":
        response = treat_feedback(translated_text)
    else:
        response = "Je n'ai pas compris votre demande."

    return {
        "en": response,  # pour TTS
        "translated_response": translate_text_to_src(response, detected_lang) if detected_lang != "en" else response
    }
