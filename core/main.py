"""Point d'entrée CLI pour le pipeline de nettoyage ASR Lingala."""
import argparse
import sys
import logging
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH si nécessaire
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from core.utils import setup_logging
from core.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de nettoyage de données ASR multilingue prêt pour la production (WAXAL dataset)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Répertoire racine de sortie pour stocker les audios nettoyés et les fichiers Parquet",
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="google/waxal",
        help="Nom du dataset Hugging Face (ex: google/waxal)",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="lin",
        help="Sous-ensemble ou langue du dataset (ex: lin pour Lingala, wol pour Wolof, etc.)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Nombre de processus de travail parallèles",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Taille des lots de traitement et d'écriture Parquet",
    )
    parser.add_argument(
        "--local_tar",
        type=str,
        default=None,
        help="Chemin vers une archive locale (.tar.gz) à utiliser au lieu de Hugging Face",
    )
    parser.add_argument(
        "--local_tsv",
        type=str,
        default="Mapping_MP3.tsv",
        help="Nom du fichier TSV de mapping dans l'archive",
    )
    parser.add_argument(
        "--local_audio_dir",
        type=str,
        default="audio_files_mp3",
        help="Nom du dossier contenant les audios dans l'archive",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Nombre maximal d'échantillons bruts à traiter (ex: 100 pour les tests)",
    )
    parser.add_argument(
        "--no_resume",
        action="store_false",
        dest="resume",
        help="Ignorer le dernier checkpoint et recommencer le traitement depuis le début",
    )
    parser.set_defaults(resume=True)

    args = parser.parse_args()

    # Instanciation de la configuration avec surcharges CLI
    settings = get_settings(
        output_dir=args.output_dir,
        hf_dataset_name=args.hf_dataset,
        hf_config_name=args.config_name,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        local_tar=args.local_tar,
        local_tsv_file=args.local_tsv,
        local_audio_dir=args.local_audio_dir,
    )

    # Configuration des logs
    log_dir = Path(settings.output_dir) / "logs"
    logger = setup_logging(log_dir=log_dir, level=logging.INFO)
    logger.info("=== Démarrage du Pipeline ASR Multilingue (WAXAL) ===")
    logger.info(f"Dossier de sortie : {settings.output_dir}")
    if settings.local_tar:
        logger.info(f"Dataset Local : {settings.local_tar}")
    else:
        logger.info(f"Dataset HF : {settings.hf_dataset_name} ({settings.hf_config_name})")
    logger.info(f"Nombre de workers : {settings.num_workers}")
    if args.max_samples:
        logger.info(f"Échantillons max : {args.max_samples}")

    try:
        pipeline = Pipeline(settings)
        pipeline.run(max_samples=args.max_samples, resume=args.resume)
        logger.info("=== Pipeline exécuté avec succès ! ===")
    except Exception as e:
        logger.critical(f"Échec critique de l'exécution du pipeline: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
