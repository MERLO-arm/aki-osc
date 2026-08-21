import pandas as pd
import torch
import warnings
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

warnings.filterwarnings('ignore')

tsv_path = "data/fulfulde_run/raw/Adamawa-Fulfulde-TTS-Dataset/Mapping_MP3.tsv"
output_path = "data/fulfulde_run/raw/Adamawa-Fulfulde-TTS-Dataset/Mapping_MP3_translated.tsv"

print("Lecture du fichier TSV...")
df = pd.read_csv(tsv_path, sep='\t')

# 2. Configuration PyTorch (8 threads CPU)
torch.set_num_threads(8)
device = "cpu"
print(f"Utilisation de l'appareil : {device} (threads={torch.get_num_threads()})")

# 3. Chargement du modèle NLLB-200
model_name = "facebook/nllb-200-distilled-600M"
print(f"Chargement du modèle {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.src_lang = "fuv_Latn"  # TRÈS IMPORTANT : spécifier que la source est en Fulfulde
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

src_lang = "fuv_Latn"  # Fulfulde
tgt_lang = "fra_Latn"  # Français

# Chargement des traductions déjà faites si interruption
if os.path.exists(output_path):
    print("Reprise de la traduction existante...")
    df_existing = pd.read_csv(output_path, sep='\t')
    translations = df_existing['translation_fr'].tolist()
    start_idx = len(translations)
    print(f"Déjà traduit : {start_idx}/{len(df)} phrases.")
else:
    translations = []
    start_idx = 0

# 4. Traduction par lots
batch_size = 8  # Réduit pour une meilleure utilisation du cache CPU
sentences = df['sentence'].tolist()

print(f"Traduction de {len(sentences) - start_idx} phrases restantes...")
for i in range(start_idx, len(sentences), batch_size):
    batch = sentences[i:i+batch_size]
    batch = [str(s) if pd.notna(s) else "" for s in batch]
    
    # Tokenisation
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Génération
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_length=100,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3
        )
    
    # Décodage
    decoded = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    translations.extend(decoded)
    
    # Affichage des 3 premières traductions de chaque lot pour donner un retour à l'utilisateur
    for idx_in_batch, (orig, trans) in enumerate(zip(batch[:2], decoded[:2])):
        print(f"[{i + idx_in_batch + 1}/{len(sentences)}] FF: {orig[:60]}... ➔ FR: {trans}")
    
    # Sauvegarde incrémentale
    df_temp = df.iloc[:len(translations)].copy()
    df_temp['translation_fr'] = translations
    df_temp.to_csv(output_path, sep='\t', index=False)

print(f"\nTraduction terminée ! Résultats sauvegardés dans {output_path}")
