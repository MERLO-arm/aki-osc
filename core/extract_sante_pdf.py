import os
import re
import pandas as pd
from pypdf import PdfReader

def clean_fulfulde_chars(text):
    """Nettoie les glyphes mal encodés dans le calque de texte du PDF."""
    if not isinstance(text, str):
        return text
    
    # Remplacements pour les caractères spéciaux Fulfuldé mal encodés
    replacements = [
        (r"I>", "ɓ"),
        (r"l>", "ɗ"),  # Parfois l> représente ɗ (ex: jal>l>aago -> jooɗaago/jallaago)
        (r"d'", "ɗ"),
        (r"b'", "ɓ"),
        (r"y'", "ƴ"),
        (r"66", "ɓɓ"),
        (r"6", "ɓ"),    # Souvent le chiffre 6 représente ɓ dans le texte extrait
        (r"’", "'"),
        (r"\[00\]", "..."),
        (r"\[...\]", "..."),
    ]
    
    cleaned = text
    for old, new in replacements:
        cleaned = re.sub(old, new, cleaned)
        
    return cleaned.strip()

def extract_pairs_from_page(text):
    """Extrait les paires Fulfuldé-Français d'une page."""
    lines = [line.strip() for line in text.split("\n")]
    pairs = []
    current_entry = None
    
    for line in lines:
        # 1. Sous-entrée commençant par un puce
        if line.startswith("•"):
            m = re.match(r"^•\s*([a-zA-Z'ɓɗƴŋ’\s\d,;!?/\\><()&%-]+)", line)
            if m:
                # Nettoyer et garder l'entrée
                current_entry = clean_fulfulde_chars(m.group(1))
        
        # 2. Entrée principale (mot suivi d'une classe grammaticale entre parenthèses)
        elif re.match(r"^[a-zA-Z0-9'ɓɗƴŋ’\s/\\><%-]+\s*\([^)]+\)", line):
            m = re.match(r"^([a-zA-Z0-9'ɓɗƴŋ’\s/\\><%-]+)", line)
            if m:
                current_entry = clean_fulfulde_chars(m.group(1))
                
        # 3. Ligne de traduction commençant par ~
        elif line.startswith("~") and current_entry:
            trans = line[1:].strip()
            # Nettoyer un peu la traduction (enlever les notes littérales ou synonymes complexes si nécessaire)
            # On garde l'essentiel en retirant la note littérale entre parenthèses pour avoir une traduction plus directe
            clean_trans = re.sub(r"\(litt\..*?\)", "", trans)
            clean_trans = re.sub(r";\s*syn\..*?$", "", clean_trans)
            clean_trans = clean_trans.strip()
            
            # Enregistrer la paire
            if current_entry and clean_trans:
                pairs.append((clean_trans, current_entry))
            current_entry = None
            
    return pairs

def main():
    pdf_path = "data/compiled/extracted_docs/Document en fulfuldé/dictionnaire du corps et de la sante peul.pdf"
    output_csv = "data/compiled/sante_pdf_extracted_pairs.csv"
    
    if not os.path.exists(pdf_path):
        print(f"Fichier PDF introuvable : {pdf_path}")
        return

    print(f"Chargement et extraction de {pdf_path}...")
    reader = PdfReader(pdf_path)
    
    all_pairs = []
    
    # Le dictionnaire proprement dit commence vers la page 30 (index 29) et va jusqu'à la page 610
    start_page = 29
    end_page = len(reader.pages) - 5  # Éviter l'index de fin
    
    for page_num in range(start_page, end_page):
        try:
            text = reader.pages[page_num].extract_text()
            pairs = extract_pairs_from_page(text)
            all_pairs.extend(pairs)
        except Exception as e:
            print(f"Erreur à la page {page_num + 1} : {e}")

    # Convertir en DataFrame
    df = pd.DataFrame(all_pairs, columns=["french", "fulfulde"])
    
    # Nettoyage et déduplication
    df = df.dropna().drop_duplicates()
    df = df[df["french"].str.len() > 1]
    df = df[df["fulfulde"].str.len() > 1]
    
    df.to_csv(output_csv, index=False)
    print(f"\nExtraction terminée !")
    print(f"Nombre total de paires de traduction extraites : {df.shape[0]}")
    print(f"Données sauvegardées dans : {output_csv}")

if __name__ == "__main__":
    main()
