import os
import zipfile

def main():
    zip_path = "/Users/ekwali/Downloads/Document en fulfuldé-20260821T092303Z-1-001.zip"
    dest_dir = "data/compiled/extracted_docs"
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"Extraction de {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.infolist():
            # Essayer de décoder le nom de fichier correctement
            try:
                # La plupart des fichiers ZIP utilisent CP437 ou UTF-8
                filename = member.filename.encode('cp437').decode('utf-8')
            except Exception:
                filename = member.filename
            
            # Remplacer les caractères problématiques ou le dossier racine
            filename = filename.replace("Document en fulfulde\xcc\x81", "Document_en_fulfulde")
            filename = filename.replace("Document en fulfulde??", "Document_en_fulfulde")
            
            target_path = os.path.join(dest_dir, filename)
            
            # Créer les répertoires parents
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # Si c'est un fichier, on l'écrit
            if not member.is_dir():
                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                    target.write(source.read())
                print(f"Extrait : {filename}")

    print("\nExtraction terminée avec succès !")

if __name__ == "__main__":
    main()
