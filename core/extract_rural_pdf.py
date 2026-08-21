import os
import re
import pandas as pd
from pypdf import PdfReader

def clean_rural_chars(text):
    """Nettoie les glyphes mal encodés pour le dictionnaire rural."""
    if not isinstance(text, str):
        return text
    
    # Remplacements de glyphes spécifiques à ce PDF
    replacements = [
        (r"", "ɓ"),
        (r"", "ɗ"),
        (r"", "ŋ"),
        (r"", "ŋ"),
        (r"’", "'"),
        (r"‘", "'"),
    ]
    
    cleaned = text
    for old, new in replacements:
        cleaned = re.sub(old, new, cleaned)
        
    return cleaned.strip()

def extract_pairs_from_text(text):
    """Extrait les paires de traduction d'une page du dictionnaire rural."""
    lines = [line.strip() for line in text.split("\n")]
    pairs = []
    
    for line in lines:
        # Ignorer les numéros de page ou en-têtes
        if re.match(r"^\d+$", line) or "VOCABULAIRE" in line or "DICTIONNAIRE" in line:
            continue
            
        # 1. Recherche d'une sous-entrée (commençant par une puce •)
        if line.startswith("•"):
            m = re.match(r"^•\s*([^:]+)\s*:\s*(.+)$", line)
            if m:
                term = clean_rural_chars(m.group(1))
                trans = m.group(2).strip()
                if term and trans:
                    pairs.append((trans, term))
        
        # 2. Recherche d'une entrée principale (contenant un signe ':')
        elif ":" in line:
            # Séparer l'entrée et la traduction
            parts = line.split(":", 1)
            term_part = parts[0].strip()
            trans_part = parts[1].strip()
            
            # Nettoyer le terme (enlever les indications grammaticales comme (ndi) ou [Maroua])
            term = re.sub(r"\[[^\]]+\]", "", term_part)  # Enlever [Maroua], [Garoua]
            term = re.sub(r"\([^)]+\)", "", term)        # Enlever (ndi), (ki/ɗe), etc.
            term = clean_rural_chars(term)
            
            # Ne garder que si le terme nettoyé ressemble à un mot Fulfuldé
            if re.match(r"^[a-zA-Z'ɓɗƴŋ’\s/\\-]+$", term) and len(term) > 1:
                pairs.append((trans_part, term))
                
    return pairs

def main():
    pdf_path = "data/compiled/extracted_docs/Document en fulfuldé/Vocabulaire peul du monde rural.pdf"
    output_csv = "data/compiled/rural_pdf_extracted_pairs.csv"
    
    if not os.path.exists(pdf_path):
        print(f"Fichier PDF introuvable : {pdf_path}")
        return

    print(f"Chargement et extraction de {pdf_path}...")
    reader = PdfReader(pdf_path)
    
    all_pairs = []
    
    # Le dictionnaire commence à la page 19 et va jusqu'à la page 240
    start_page = 18
    end_page = 240
    
    for page_num in range(start_page, end_page):
        try:
            text = reader.pages[page_num].extract_text()
            pairs = extract_pairs_from_text(text)
            all_pairs.extend(pairs)
        except Exception as e:
            print(f"Erreur à la page {page_num + 1} : {e}")

    # Convertir en DataFrame
    df = pd.DataFrame(all_pairs, columns=["french", "fulfulde"])
    
    # Nettoyage
    df = df.dropna().drop_duplicates()
    df = df[df["french"].str.len() > 1]
    df = df[df["fulfulde"].str.len() > 1]
    
    df.to_csv(output_csv, index=False)
    print(f"\nExtraction terminée !")
    print(f"Nombre total de paires de traduction extraites : {df.shape[0]}")
    print(f"Données sauvegardées dans : {output_csv}")

if __name__ == "__main__":
    main()
