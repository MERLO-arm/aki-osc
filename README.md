# Pipeline de Nettoyage de Données ASR Multilingue (WAXAL Dataset)

Ce projet fournit un pipeline MLOps de nettoyage et prétraitement de données audio et textuelles **prêt pour la production**, conçu pour traiter des jeux de données ASR multilingues (notamment le dataset **WAXAL** de Hugging Face : `google/waxal`).

---

## 🌟 Fonctionnalités Principales

- **Support Multilingue** : Conçu pour s'adapter à toutes les langues du dataset WAXAL (Lingala, Wolof, Swahili, Fulfulde, etc.) via la configuration CLI `--config_name`.
- **Chargement en Streaming** : Utilise Hugging Face `datasets` en mode streaming pour traiter des centaines d'heures d'audio sans saturer la RAM.
- **Traitement Audio Avancé** :
  - Standardisation à **16 kHz Mono PCM WAV**.
  - Découpage automatique des silences (`pydub`).
  - Filtrage Voice Activity Detection (**WebRTC VAD**).
  - Calcul du rapport signal/bruit (**SNR**).
  - Détection de musique optionnelle via **YAMNet** (TensorFlow Hub) avec fallback silencieux.
  - Découpage et attribution d'identifiants uniques (**UUID**) pour éviter les collisions de fichiers.
- **Nettoyage Textuel Adapté aux Langues Africaines** :
  - Normalisation Unicode (NFC).
  - Normalisation de la casse.
  - Suppression des balises de métadonnées (`[...]`, `(...)`, etc.).
  - Nettoyage de la ponctuation en **conservant les caractères accentués et caractères spécifiques** (`ɛ`, `ɔ`, `ŋ`, `é`, `è`, `â`, `ê`, `î`, `ô`, `û`, `à`, etc.).
  - Normalisation des espaces et filtrage des phrases trop courtes.
- **Architecture Production & MLOps** :
  - **Parallélisation** multi-processus (`multiprocessing`).
  - **Checkpointing** résilient à l'interruption (reprise automatique via fichier `.pkl`).
  - **Écriture par lots** au format **Parquet** pour l'efficacité E/S.
  - **Génération automatique de splits** (`train`, `validation`, `test`) avec graine aléatoire fixe (`seed=42`).
  - Génération d'un résumé global structuré dans `manifest.json`.

---

## 📁 Structure du Projet

```
multilingual_asr_pipeline/
├── .env                  # Overrides de configuration d'environnement
├── .gitignore            # Exclusions de fichiers temporaires / données
├── README.md             # Documentation globale
├── requirements.txt      # Dépendances Python
├── setup.py              # Configuration d'installation du paquet
├── config/
│   ├── __init__.py
│   └── settings.py       # Configuration centralisée via Pydantic
├── src/
│   ├── __init__.py
│   ├── audio_processor.py# Standardisation audio, SNR, VAD, YAMNet
│   ├── text_cleaner.py   # Normalisation textuelle multilingue
│   ├── pipeline.py       # Orchestration, multiprocessing, Parquet, splits
│   ├── utils.py          # Logging structuré & utilitaires système
│   └── main.py           # CLI principal
├── tests/
│   ├── __init__.py
│   └── test_audio_processor.py # Tests unitaires Pytest
├── notebooks/
│   └── test_pipeline.ipynb # Notebook d'expérimentation & test rapide
├── data/                 # Répertoire de sortie (audio/, splits, checkpoints)
└── logs/                 # Fichiers de journaux d'exécution
```

---

## 🛠️ Installation

### 1. Prérequis Système
Assurez-vous d'avoir `ffmpeg` installé sur votre système :
- **macOS** : `brew install ffmpeg`
- **Linux (Ubuntu/Debian)** : `sudo apt update && sudo apt install -y ffmpeg`

### 2. Environnement Virtuel Python (3.9+)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installation des dépendances
Mode développement basique :
```bash
pip install -e .
```

Pour inclure la détection de musique YAMNet (TensorFlow) et l'environnement de test :
```bash
pip install -e ".[dev,music]"
```

---

## ⚙️ Configuration

La configuration se trouve dans `config/settings.py` (Pydantic v2).
Toutes les valeurs peuvent être surchargées via le fichier `.env` ou des variables d'environnement avec le préfixe `PIPELINE_`.

Exemple d'overrides dans `.env` :
```env
PIPELINE_SAMPLE_RATE=16000
PIPELINE_MIN_SNR_DB=5.0
PIPELINE_VAD_MODE=3
PIPELINE_NUM_WORKERS=8
```

---

## 🚀 Utilisation

### 1. Test Rapide sur 100 Échantillons (Lingala)
Pour valider le pipeline en local sur un petit échantillon :
```bash
python3 src/main.py --output_dir ./data/test_run --hf_dataset google/WaxalNLP --config_name lin_asr --max_samples 100 --num_workers 2
```

### 2. Lancement en Production (Pour n'importe quelle langue WAXAL)
Pour lancer le traitement complet parallélisé sur 48 workers :
```bash
python src/main.py --output_dir /mnt/data/clean_waxal --hf_dataset google/waxal --config_name wol --num_workers 48 --batch_size 2048 --resume
```

---

## 🧪 Tests Unitaires

Exécutez la suite de tests avec Pytest :
```bash
pytest tests/ -v
```
