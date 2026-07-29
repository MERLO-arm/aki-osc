"""Centralized Pydantic configuration settings for the Multilingual ASR Pipeline (WAXAL)."""
import os
from typing import Optional
from pydantic import Field

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    PYDANTIC_V2 = True
except ImportError:
    from pydantic import BaseSettings  # fallback for Pydantic v1
    PYDANTIC_V2 = False


class PipelineSettings(BaseSettings):
    """Pipeline configuration parameters with environment variable overrides (prefix `PIPELINE_`)."""

    # Paramètres Audio
    sample_rate: int = Field(default=16000, description="Taux d'échantillonnage audio cible (Hz)")
    channels: int = Field(default=1, description="Nombre de canaux audio cible (1 = mono)")
    min_silence_len: int = Field(default=500, description="Longueur minimale de silence pour le découpage (ms)")
    silence_threshold_dbfs: int = Field(default=-60, description="Seuil de silence en dBFS")
    use_rms_split: bool = Field(default=True, description="Utiliser une méthode hybride de découpage (librosa + pydub)")
    vad_mode: int = Field(default=3, description="Mode d'agressivité WebRTC VAD (0 à 3)")
    min_snr_db: float = Field(default=5.0, description="Rapport signal/bruit minimal (dB)")
    min_segment_duration: float = Field(default=2.0, description="Durée minimale d'un segment audio (secondes)")
    max_segment_duration: float = Field(default=25.0, description="Durée maximale d'un segment audio (secondes)")
    enable_yamnet: bool = Field(default=True, description="Activer la détection de musique YAMNet si TensorFlow est présent")

    # Paramètres Nettoyage Texte
    min_word_count: int = Field(default=2, description="Nombre minimal de mots par transcription")

    # Paramètres Execution & Parallélisme
    num_workers: int = Field(default=4, description="Nombre de processus de travail parallèles")
    batch_size: int = Field(default=100, description="Taille des lots d'écriture Parquet et checkpoint")

    # Paramètres Splits Dataset
    train_ratio: float = Field(default=0.8, description="Proportion du split Entraînement")
    val_ratio: float = Field(default=0.1, description="Proportion du split Validation")
    test_ratio: float = Field(default=0.1, description="Proportion du split Test")
    seed: int = Field(default=42, description="Graine aléatoire pour la reproductibilité")

    # Paramètres Hugging Face & Entrées/Sorties
    hf_dataset_name: str = Field(default="google/waxal", description="Identifiant du jeu de données Hugging Face")
    hf_config_name: Optional[str] = Field(default="lin", description="Configuration / langue du jeu de données")
    output_dir: str = Field(default="./data", description="Dossier racine pour l'enregistrement des résultats")

    if PYDANTIC_V2:
        model_config = SettingsConfigDict(
            env_prefix="PIPELINE_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )
    else:
        class Config:
            env_prefix = "PIPELINE_"
            env_file = ".env"
            env_file_encoding = "utf-8"
            extra = "ignore"


def get_settings(**kwargs) -> PipelineSettings:
    """Instancie et retourne la configuration globale."""
    return PipelineSettings(**kwargs)
