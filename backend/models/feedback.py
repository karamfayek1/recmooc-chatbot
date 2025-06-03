from transformers import pipeline

# Créer le pipeline de zero-shot classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Labels cibles
candidate_labels = ["positive_feedback", "negative_feedback"]

def treat_feedback(text):
    """Classifie le texte de l'utilisateur avec zero-shot classification."""
    result = classifier(text, candidate_labels)
    top_label = result["labels"][0]

    if top_label == "positive_feedback":
        output = "We are happy that you like the course."
    elif top_label == "negative_feedback":
        output = "We will ameliorate our recommendation."
    else:
        output = "Thanks for your feedback!"

    return output

