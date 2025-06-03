# Détection de la langue
import langdetect , langid 
from collections import Counter
def detect_language(text):
    detected_languages = []
    if not text.strip():
        return "unknown"
    
    try:
        detected_languages.append(langdetect.detect(text))
    except:
        pass  
    try:
        detected_languages.append(langid.classify(text)[0])
    except:
        pass 
    """ 
    try:
        detector = Detector(text)
        language_pyglot = detector.language.code
        detect_language.append(language_pyglot)
    except:
        pass
        """
    if not detected_languages:
        return "unknown"
    return Counter(detected_languages).most_common(1)[0][0]