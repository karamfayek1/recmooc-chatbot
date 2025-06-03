from typing import Literal
from enum import Enum
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable
from langchain_community.llms.ollama import Ollama


classification_few_shot_prompt_examples = """\
Classify the user message into one of the following categories:
- 'recommendation': user asks for a course recommendation
- 'feedback': user clarifies or corrects a previous recommendation
- 'assistance': any other help unrelated to course recommendations

Use the prior conversation context when available to distinguish between feedback and new requests. 
Feedback is only possible if the message is modifying, clarifying, or correcting something previously discussed.

Examples:
User: Can you recommend a good online Python course?
Classification: recommendation

User: i want to learn data science 
Classification: recommendation

User: find me java courses
Classification: recommendation

User: i want intermediate python course 
Classification: recommendation

User: Actually, I'm more interested in beginner-level material.
Classification: feedback

User: I want to start learning data science, what should I take?
Classification: recommendation

User: Can you look for beginner courses?
Classification: feedback

User: I said I was looking for backend, not frontend.
Classification: feedback

User: i didn't like this course
Classification: feedback

User: it's a very good course 
Classification: feedback

User: What's the difference between supervised and unsupervised learning?
Classification: assistance

User: how to create an account here 
Classification: assistance

User: how can i change my passeword
Classification: assistance

User: which are the feature of accesibility 
Classification: assistance

User: What is 1+1?
Classification: assistance

User: I'd like to learn Java
Classification: recommendation

User: I prefer courses with hands-on projects.
Classification: feedback

User: Is there a difference between pandas and NumPy?
Classification: assistance

User: Any courses for web development?
Classification: recommendation

Now classify the next user message. Respond only with a JSON object in the following format:
{{"classification": "recommendation"}}  # or "feedback" or "assistance"
"""



# Définir l'enum et le modèle Pydantic
class Classification(str, Enum):
    assistance = "assistance"
    recommendation = "recommendation"
    feedback = "feedback"

class UserIntent(BaseModel):
    classification: Classification = Field(
        description=(
            "The type of user request:\n"
            "- 'recommendation': user asks for course recommendation\n"
            "- 'feedback': user clarifies or corrects a previous recommendation\n"
            "- 'assistance': any other help unrelated to course recommendations"
        )
    )

# Initialiser le LLM (modifie si besoin selon ton modèle local)
local_llm = "llama3.2" 
llm = Ollama(model=local_llm, temperature=0)


# Création du parseur et du prompt
user_intent_parser = PydanticOutputParser(pydantic_object=UserIntent)

classification_prompt = ChatPromptTemplate.from_messages([
    ("system", classification_few_shot_prompt_examples),
    ("user", "{input}")
])

# Construire la chaîne
classification_chain: Runnable = classification_prompt | llm | user_intent_parser

# Fonction finale à utiliser
def classify_input(text: str) -> str:
    """Utilise un LLM pour classer une intention utilisateur."""
    try:
        result: UserIntent = classification_chain.invoke({"input": text})
        return result.classification.value
    except Exception as e:
        print(f"Erreur lors de la classification : {e}")
        return "unknown"

# Exemple d'utilisation
if __name__ == "__main__":
    user_input = input("Entrez votre message : ")
    prediction = classify_input(user_input)
    print(f"Classe prédite : {prediction}")
