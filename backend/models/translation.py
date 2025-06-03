from transformers import MarianMTModel, MarianTokenizer,M2M100Tokenizer,M2M100ForConditionalGeneration

def translate_text_to_en(text, src_lang):
    """Traduit un texte donné en anglais en utilisant le modèle MarianMT."""
    model_name = "Helsinki-NLP/opus-mt-mul-en"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    # Traduction
    translated = tokenizer(text, return_tensors="pt", padding=True)
    translated_text = model.generate(**translated)
    output = tokenizer.decode(translated_text[0], skip_special_tokens=True)
    return output
def translate_text_to_src(text, target_lang):
   model_name = "facebook/m2m100_418M"
     
   tokenizer = M2M100Tokenizer.from_pretrained(model_name)
   model = M2M100ForConditionalGeneration.from_pretrained(model_name)

   text = text
   target_lang = target_lang # change to "de", "es", "ar", etc.
   tokenizer.src_lang = "en"
   encoded = tokenizer(text, return_tensors="pt")
   generated = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
   print(tokenizer.decode(generated[0], skip_special_tokens=True))
   return tokenizer.decode(generated[0], skip_special_tokens=True)

# Exemple d'utilisation
if __name__ == "__main__":
    test_text = "Bonjour tout le monde!"
    src_lang='fr'
    print(f"Texte traduit: {translate_text_to_en(test_text, src_lang)}")
