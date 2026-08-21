import warnings
warnings.filterwarnings('ignore')
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/nllb-200-distilled-600M"
print(f"Chargement de {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Le code NLLB pour le Fulfulde (Nigerian Fulfulde) est fuv_Latn
# Pulaar (Sénégal) est fuc_Latn
target_lang = "fuv_Latn"

texts = [
    "The community needs clean water for health and agriculture.",
    "We are going to visit the capital city next week.",
    "Bonjour, comment allez-vous aujourd'hui ?"
]

print("\n--- Traduction avec NLLB-200 (Open Source) ---")
for text in texts:
    inputs = tokenizer(text, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs, 
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang], 
        max_length=50
    )
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    print(f"Original : {text}")
    print(f"Fulfulde : {translated_text}\n")
