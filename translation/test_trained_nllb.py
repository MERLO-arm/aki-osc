import os
import json
import inspect
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, LoraConfig

base_model_id = "facebook/nllb-200-distilled-600M"
lora_model_dir = "outputs/nllb_fulfulde_lora_final"
config_path = os.path.join(lora_model_dir, "adapter_config.json")

# Correction de la compatibilité des versions de PEFT (Kaggle vs local Mac)
if os.path.exists(config_path):
    print("Nettoyage de adapter_config.json pour la compatibilité PEFT locale...")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    # Récupérer les arguments acceptés par la classe LoraConfig locale
    sig = inspect.signature(LoraConfig.__init__)
    supported_args = set(sig.parameters.keys())
    
    # Garder uniquement les clés compatibles
    clean_config = {k: v for k, v in config_dict.items() if k in supported_args}
    
    with open(config_path, "w") as f:
        json.dump(clean_config, f, indent=2)

print("Chargement du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

print("Chargement du modèle de base NLLB-200...")
model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id)

print("Application des poids LoRA entraînés...")
if os.path.exists(lora_model_dir):
    model = PeftModel.from_pretrained(model, lora_model_dir)
else:
    print("Dossier LoRA introuvable !")
    exit()

def translate(text, src_lang="fra_Latn", tgt_lang="fuv_Latn"):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_length=128
        )
    
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

# Tester quelques phrases médicales ou agricoles présentes dans notre dataset
test_phrases = [
    # Français -> Fulfuldé
    ("Une injection provoquant une anesthésie générale", "fra_Latn", "fuv_Latn"),
    ("Le médecin lui fait une perfusion.", "fra_Latn", "fuv_Latn"),
    ("argent", "fra_Latn", "fuv_Latn"),
    # Fulfuldé -> Français
    ("baatal oanninanngal", "fuv_Latn", "fra_Latn"),
    ("ceede siga", "fuv_Latn", "fra_Latn"),
    ("babal seekgo", "fuv_Latn", "fra_Latn"),
]

print("\n=== Résultats des tests de traduction ===")
for text, src, tgt in test_phrases:
    result = translate(text, src, tgt)
    print(f"\nSource ({src}) : {text}")
    print(f"Traduction ({tgt}) : {result}")
