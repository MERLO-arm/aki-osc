"""Utilitaires système et logging pour le pipeline ASR Lingala."""
import logging
import os
import sys
from pathlib import Path
from typing import Union


def setup_logging(log_dir: Union[str, Path] = "./logs", log_filename: str = "pipeline.log", level: int = logging.INFO) -> logging.Logger:
    """Configure et retourne le logger principal avec écriture console et fichier.

    Args:
        log_dir: Répertoire de stockage des fichiers de logs.
        log_filename: Nom du fichier journal.
        level: Niveau de journalisation (ex. logging.INFO).

    Returns:
        logging.Logger instancié.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_path = log_path / log_filename

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger("waxal_asr_pipeline")
    logger.setLevel(level)

    # Réinitialiser les handlers existants s'il y en a
    if logger.hasHandlers():
        logger.handlers.clear()

    # Handler Fichier
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    # Handler Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    return logger


def ensure_dirs(*dirs: Union[str, Path]) -> None:
    """Crée les répertoires spécifiés s'ils n'existent pas déjà.

    Args:
        *dirs: Liste des chemins de dossiers à créer.
    """
    for d in dirs:
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)
