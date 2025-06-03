import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Charger le modèle GPT-2 et le tokenizer
model_path = r"C:\Users\DELL\Desktop\chatbot\backend\models\gptmodel"


tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# GPT-2 n'a pas de token de padding, donc on utilise <|endoftext|>
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

def generate_answer(question):
    prompt = f" Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Générer la réponse
    output = model.generate(**inputs, max_length=256, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    
    # Décoder la sortie
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extraire uniquement la réponse après "Answer:"
    answer = generated_text.split("Answer:")[1].strip() if "Answer:" in generated_text else generated_text.strip()
    
    return answer

# Exemple d'utilisation
if __name__ == "__main__":
    user_input = input("Entrez votre texte : ")
    answer = generate_answer(user_input)
    print(answer)
