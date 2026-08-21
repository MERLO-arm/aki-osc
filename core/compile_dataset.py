import os
import re
import json
import pandas as pd

def parse_java_arrays(java_file_path):
    """Extrait les tableaux motsFR et motsFB du fichier Java."""
    if not os.path.exists(java_file_path):
        print(f"Fichier Java introuvable : {java_file_path}")
        return []
    
    with open(java_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex pour capturer les éléments entre accolades du tableau
    fr_match = re.search(r"String\[\]\s+motsFR\s*=\s*\{([^}]+)\};", content)
    fb_match = re.search(r"String\[\]\s+motsFB\s*=\s*\{([^}]+)\};", content)

    if not fr_match or not fb_match:
        print("Impossible de trouver les tableaux motsFR ou motsFB dans le fichier Java.")
        return []

    # Nettoyage et découpage des chaînes Java
    def clean_java_array(raw_str):
        # Séparer par virgules, mais faire attention aux échappements
        # Pour faire simple, on va utiliser une expression régulière pour trouver tous les éléments entre guillemets
        items = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', raw_str)
        # Nettoyer les échappements Java
        cleaned = [item.replace("\\'", "'").replace('\\"', '"').strip() for item in items]
        return cleaned

    mots_fr = clean_java_array(fr_match.group(1))
    mots_fb = clean_java_array(fb_match.group(1))

    pairs = []
    for fr, fb in zip(mots_fr, mots_fb):
        if fr and fb:  # Éviter les éléments vides
            pairs.append({"french": fr, "fulfulde": fb, "source_dataset": "github_samglish"})
            
    print(f"Dictionnaire Java extrait : {len(pairs)} paires de mots.")
    return pairs

def main():
    external_dir = "data/external_repos"
    output_dir = "data/compiled"
    os.makedirs(output_dir, exist_ok=True)

    all_pairs = []

    # 1. Charger les données locales déjà traduites (Mapping_MP3_translated.tsv)
    local_tsv = "data/fulfulde_run/raw/Adamawa-Fulfulde-TTS-Dataset/Mapping_MP3_translated.tsv"
    if os.path.exists(local_tsv):
        df_local = pd.read_csv(local_tsv, sep="\t")
        for _, row in df_local.iterrows():
            sentence = row.get("sentence")
            translation = row.get("translation_fr")
            if pd.notna(sentence) and pd.notna(translation):
                all_pairs.append({
                    "french": str(translation).strip(),
                    "fulfulde": str(sentence).strip(),
                    "source_dataset": "local_tts_dataset"
                })
        print(f"Données locales chargées : {df_local.shape[0]} paires.")

    # 2. Extraire le dictionnaire Java de Samglish
    java_file = os.path.join(external_dir, "traduction_francais_fulfude/src/traduction_foulbe/main.java")
    java_pairs = parse_java_arrays(java_file)
    all_pairs.extend(java_pairs)

    # 3. Charger les fichiers TSV de fulfulde-translator
    translator_dir = os.path.join(external_dir, "fulfulde-translator/data")
    
    tsv_files = [
        "adamawa_english_fulfulde_french_fub.tsv",
        "corps_sante_english_fulfulde_french_fub.tsv"
    ]
    
    for tsv_name in tsv_files:
        path = os.path.join(translator_dir, tsv_name)
        if os.path.exists(path):
            df = pd.read_csv(path, sep="\t")
            count = 0
            for _, row in df.iterrows():
                fr = row.get("French")
                fb = row.get("Fulfulde")
                if pd.notna(fr) and pd.notna(fb):
                    all_pairs.append({
                        "french": str(fr).strip(),
                        "fulfulde": str(fb).strip(),
                        "source_dataset": f"github_translator_{tsv_name}"
                    })
                    count += 1
            print(f"Chargé {count} paires de {tsv_name}")

    # 4. Charger les fichiers JSONL de fulfulde-translator
    jsonl_files = [
        "adamawa_english_fulfulde_french_base.jsonl",
        "adamawa_health_english_fulfulde_french.jsonl"
    ]
    
    for jsonl_name in jsonl_files:
        path = os.path.join(translator_dir, jsonl_name)
        if os.path.exists(path):
            count = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        fr = data.get("french")
                        fb = data.get("fulfulde")
                        if fr and fb:
                            all_pairs.append({
                                "french": str(fr).strip(),
                                "fulfulde": str(fb).strip(),
                                "source_dataset": f"github_translator_{jsonl_name}"
                            })
                            count += 1
                    except Exception as e:
                        pass
            print(f"Chargé {count} paires de {jsonl_name}")

    # 4.5. Charger le dictionnaire Peul-Fulfuldé-Français extrait du ZIP
    zip_csv = "data/compiled/extracted_docs/Document en fulfuldé/dictionnaire_peul_fulfulde_francais.csv"
    if os.path.exists(zip_csv):
        df_zip = pd.read_csv(zip_csv)
        count = 0
        for _, row in df_zip.iterrows():
            fr = row.get("francais")
            fb = row.get("fulfulde")
            if pd.notna(fr) and pd.notna(fb):
                all_pairs.append({
                    "french": str(fr).strip(),
                    "fulfulde": str(fb).strip(),
                    "source_dataset": "extracted_zip_dictionary"
                })
                count += 1
        print(f"Chargé {count} paires de dictionnaire_peul_fulfulde_francais.csv")

    # 4.6. Charger les paires médicales extraites du PDF de la santé
    pdf_sante_csv = "data/compiled/sante_pdf_extracted_pairs.csv"
    if os.path.exists(pdf_sante_csv):
        df_pdf_sante = pd.read_csv(pdf_sante_csv)
        count = 0
        for _, row in df_pdf_sante.iterrows():
            fr = row.get("french")
            fb = row.get("fulfulde")
            if pd.notna(fr) and pd.notna(fb):
                all_pairs.append({
                    "french": str(fr).strip(),
                    "fulfulde": str(fb).strip(),
                    "source_dataset": "extracted_pdf_health"
                })
                count += 1
        print(f"Chargé {count} paires de sante_pdf_extracted_pairs.csv")

    # 4.7. Charger les paires du monde rural extraites du PDF
    pdf_rural_csv = "data/compiled/rural_pdf_extracted_pairs.csv"
    if os.path.exists(pdf_rural_csv):
        df_pdf_rural = pd.read_csv(pdf_rural_csv)
        count = 0
        for _, row in df_pdf_rural.iterrows():
            fr = row.get("french")
            fb = row.get("fulfulde")
            if pd.notna(fr) and pd.notna(fb):
                all_pairs.append({
                    "french": str(fr).strip(),
                    "fulfulde": str(fb).strip(),
                    "source_dataset": "extracted_pdf_rural"
                })
                count += 1
        print(f"Chargé {count} paires de rural_pdf_extracted_pairs.csv")

    # 5. Créer le DataFrame final, dédupliquer et sauvegarder
    df_all = pd.DataFrame(all_pairs)
    
    # Nettoyage
    df_all = df_all.drop_duplicates(subset=["french", "fulfulde"])
    
    output_csv = os.path.join(output_dir, "nllb_parallel_corpus.csv")
    df_all.to_csv(output_csv, index=False)
    
    print("\n=== Synthèse de la Compilation ===")
    print(f"Nombre total de paires uniques : {df_all.shape[0]}")
    print("\nRépartition par source :")
    print(df_all["source_dataset"].value_counts())
    print(f"\nSauvegardé avec succès dans : {output_csv}")

if __name__ == "__main__":
    main()
