import os
import json
import inspect
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, LoraConfig

base_model_id = "facebook/nllb-200-distilled-600M"
lora_model_dir = "outputs/nllb_fulfulde_lora_final"
config_path = os.path.join(lora_model_dir, "adapter_config.json")

# Correction de la compatibilité PEFT
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    sig = inspect.signature(LoraConfig.__init__)
    supported_args = set(sig.parameters.keys())
    clean_config = {k: v for k, v in config_dict.items() if k in supported_args}
    with open(config_path, "w") as f:
        json.dump(clean_config, f, indent=2)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id)
if os.path.exists(lora_model_dir):
    model = PeftModel.from_pretrained(model, lora_model_dir)

def translate(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_length=128,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            num_beams=4
        )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

# Liste de phrases à tester fournie par l'utilisateur
eval_data = [
    {"fr": "écorces de Sterculia setigera", "ff": "Laalaaje boɓori"},
    {"fr": "écorces de Ficus platyphylla", "ff": "Laaalaaje dundeehi"},
    {"fr": "gingembre sec", "ff": "Citta-afo joorɗe"},
    {"fr": "feuilles d'Abrus precatorius", "ff": "Haako belɗamhi"},
    {"fr": "graines de Vigna unguiculata", "ff": "Nyebbe"},
    {"fr": "peau de céphalophe de Grimm", "ff": "Laral hamfurde"},
    {"fr": "Le matin de saison froide, on doit bien se réchauffer avant de sortir.", "ff": "Fajira dabbunde, sey goɗɗoo aamta booɗɗum lutta wurtaago."},
    {"fr": "Pourquoi es-tu aussi idiot qu'un premier-né?", "ff": "Ko waɗ maa a paataaɗo bana afo?"},
    {"fr": "Ils les prennent, mais ils ne sont pas efficaces.", "ff": "Ɓe ɗon njara ɗe, ɗe nafaay."},
    {"fr": "Les amibes, c'est ce qu'il y a en grand nombre dans le ventre.", "ff": "Amiiɓ, kanjum ɓuri ɗuɗogo nder reedu."}
]

print("\n=== EVALUATION DES TRADUCTIONS DU MODELE ===")
for item in eval_data:
    pred_ff = translate(item["fr"], "fra_Latn", "fuv_Latn")
    print(f"\n[FR] : {item['fr']}")
    print(f"[Attendu FF] : {item['ff']}")
    print(f"[Modèle FF]  : {pred_ff}")
