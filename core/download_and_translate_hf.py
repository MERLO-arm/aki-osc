import os
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def main():
    print("=== Chargement et Filtrage du Dataset Hugging Face ===")
    
    # 1. Télécharger le dataset Hugging Face en mode streaming pour éviter de saturer la RAM (OOM)
    dataset_name = "abdouaziiz/fulfulde_clean"
    print(f"Téléchargement du dataset {dataset_name} depuis Hugging Face (mode streaming)...")
    try:
        ds = load_dataset(dataset_name, split="train", streaming=True)
    except Exception as e:
        print(f"Erreur lors du chargement du dataset : {e}")
        return

    # 2. Parcourir et filtrer uniquement le dialecte Adamawa sans charger les données audio lourdes en mémoire
    print("Filtrage du dialecte Adamawa...")
    adamawa_rows = []
    for row in ds:
        dialect = row.get("dialect", "")
        if dialect and str(dialect).lower().strip() == "adamawa":
            # Récupérer les infos textuelles uniquement
            audio_info = row.get("audio", {})
            audio_path = audio_info.get("path", "") if isinstance(audio_info, dict) else ""
            adamawa_rows.append({
                "transcription": row.get("transcription", ""),
                "dialect": dialect,
                "audio_path": audio_path
            })
            
    df_adamawa = pd.DataFrame(adamawa_rows)
    print(f"Nombre de lignes filtrées pour le dialecte Adamawa : {len(df_adamawa)}")
    
    if len(df_adamawa) == 0:
        print("Aucune ligne correspondante trouvée. Fin du programme.")
        return

    # 3. Charger le modèle NLLB-200 pour la traduction automatique en Français
    print("\n=== Initialisation de NLLB-200 pour la traduction ===")
    model_name = "facebook/nllb-200-distilled-600M"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Appareil de calcul utilisé : {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.src_lang = "fuv_Latn"  # Source : Fulfuldé
    tgt_lang = "fra_Latn"  # Cible : Français

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    # 4. Traduire les transcriptions Fulfuldé vers le Français
    print("\nTraduction des transcriptions en cours (Français)...")
    translations = []
    
    # On fait la traduction par lots pour aller plus vite si GPU disponible
    batch_size = 16 if device == "cuda" else 2
    transcripts = df_adamawa["transcription"].tolist()

    for i in tqdm(range(0, len(transcripts), batch_size)):
        batch = transcripts[i:i+batch_size]
        batch = [str(t).strip() for t in batch]
        
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
                max_length=128
            )
        
        decoded = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        translations.extend(decoded)

    df_adamawa["translation_fr"] = translations

    # 5. Sauvegarder les données de traduction
    output_dir = "data/compiled"
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder la version texte parallèle pour NLLB
    df_parallel = pd.DataFrame({
        "french": df_adamawa["translation_fr"],
        "fulfulde": df_adamawa["transcription"],
        "source_dataset": "hf_abdouaziiz_adamawa_synthetic"
    })
    
    parallel_csv = os.path.join(output_dir, "hf_adamawa_parallel.csv")
    df_parallel.to_csv(parallel_csv, index=False)
    print(f"Jeu de données de traduction sauvegardé dans : {parallel_csv}")

    # Sauvegarder la version audio-texte pour Whisper
    audio_csv = os.path.join(output_dir, "hf_adamawa_audio_metadata.csv")
    df_adamawa.to_csv(audio_csv, index=False)
    print(f"Métadonnées audio-texte sauvegardées dans : {audio_csv}")

if __name__ == "__main__":
    main()
