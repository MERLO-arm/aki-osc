"""Module principal d'orchestration du pipeline ASR Lingala."""
import os
import json
import pickle
import logging
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from multiprocessing import Pool
from datasets import load_dataset
from tqdm import tqdm
import tarfile

from config.settings import PipelineSettings
from core.audio_processor import AudioProcessor
from core.text_cleaner import TextCleaner
from core.utils import ensure_dirs

logger = logging.getLogger("waxal_asr_pipeline")

# Helper global pour l'exécution parallèle dans multiprocessing.Pool
_global_processor: Optional[AudioProcessor] = None
_global_cleaner: Optional[TextCleaner] = None
_global_audio_dir: Optional[Path] = None


def _init_worker(settings_dict: Dict, audio_dir_str: str):
    """Initialise les instances globales pour chaque processus worker."""
    global _global_processor, _global_cleaner, _global_audio_dir
    processor_kwargs = {
        "sample_rate": settings_dict["sample_rate"],
        "channels": settings_dict["channels"],
        "min_silence_len": settings_dict["min_silence_len"],
        "silence_threshold_dbfs": settings_dict["silence_threshold_dbfs"],
        "use_rms_split": settings_dict["use_rms_split"],
        "vad_mode": settings_dict["vad_mode"],
        "min_snr_db": settings_dict["min_snr_db"],
        "min_segment_duration": settings_dict["min_segment_duration"],
        "max_segment_duration": settings_dict["max_segment_duration"],
        "enable_yamnet": settings_dict["enable_yamnet"],
    }
    _global_processor = AudioProcessor(**processor_kwargs)
    _global_cleaner = TextCleaner(min_word_count=settings_dict["min_word_count"])
    _global_audio_dir = Path(audio_dir_str)


def _worker_process_sample(sample_tuple: Tuple[str, Dict, str, str]) -> List[Dict]:
    """Fonction exécutée par chaque worker pour traiter un échantillon audio."""
    global _global_processor, _global_cleaner, _global_audio_dir
    sample_id, audio_data, raw_text, speaker_id = sample_tuple

    if _global_processor is None or _global_cleaner is None or _global_audio_dir is None:
        return []

    return _global_processor.process_audio_file(
        audio_input=audio_data,
        raw_text=raw_text,
        text_cleaner_fn=_global_cleaner.clean,
        output_audio_dir=_global_audio_dir,
        speaker_id=speaker_id,
    )


class Pipeline:
    """Orchestrateur principal du pipeline de nettoyage de données ASR."""

    def __init__(self, settings: PipelineSettings):
        """Initialise le pipeline avec la configuration fournie.

        Args:
            settings: Configuration Pydantic.
        """
        self.settings = settings
        self.output_dir = Path(self.settings.output_dir)
        self.audio_dir = self.output_dir / "audio"
        self.batches_dir = self.output_dir / "batches"
        self.checkpoint_file = self.output_dir / "checkpoint.pkl"
        self.manifest_file = self.output_dir / "manifest.json"

        # Création des dossiers nécessaires
        ensure_dirs(self.output_dir, self.audio_dir, self.batches_dir)

        # Ensemble des identifiants d'échantillons déjà traités
        self.processed_ids: Set[str] = set()

    def load_checkpoint(self) -> None:
        """Charge l'état du checkpoint à partir du fichier pickle s'il existe."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "rb") as f:
                    self.processed_ids = pickle.load(f)
                logger.info(f"Checkpoint chargé: {len(self.processed_ids)} échantillons déjà traités.")
            except Exception as e:
                logger.warning(f"Erreur lors du chargement du checkpoint ({e}). Nouveau départ.")
                self.processed_ids = set()
        else:
            logger.info("Aucun checkpoint existant trouvé. Nouveau traitement.")

    def save_checkpoint(self) -> None:
        """Sauvegarde les IDs traités dans le fichier pickle de checkpoint."""
        try:
            with open(self.checkpoint_file, "wb") as f:
                pickle.dump(self.processed_ids, f)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du checkpoint: {e}")

    def run(self, max_samples: Optional[int] = None, resume: bool = True) -> None:
        """Exécute le pipeline complet de la lecture au découpage final.

        Args:
            max_samples: Limite optionnelle d'échantillons bruts à traiter.
            resume: Indique s'il faut reprendre depuis le dernier checkpoint.
        """
        if resume:
            self.load_checkpoint()

        logger.info(f"Chargement du jeu de données...")

        if self.settings.local_tar:
            # Mode dataset local
            ds = self._load_local_dataset()
        else:
            # Chargement streaming Hugging Face
            try:
                if self.settings.hf_config_name:
                    ds = load_dataset(
                        self.settings.hf_dataset_name,
                        self.settings.hf_config_name,
                        split="train",
                        streaming=True,
                    )
                else:
                    ds = load_dataset(
                        self.settings.hf_dataset_name,
                        split="train",
                        streaming=True,
                    )
            except Exception as e:
                logger.error(f"Impossible de charger le dataset Hugging Face: {e}")
                raise

        settings_dict = self.settings.model_dump() if hasattr(self.settings, "model_dump") else self.settings.dict()

        batch_buffer: List[Dict] = []
        batch_count = len(list(self.batches_dir.glob("batch_*.parquet")))
        sample_counter = 0

        # Pool de processus pour la parallélisation
        num_workers = max(1, self.settings.num_workers)
        logger.info(f"Démarrage du traitement parallélisé avec {num_workers} processus workers...")

        with Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(settings_dict, str(self.audio_dir)),
        ) as pool:

            pending_items: List[Tuple[str, Dict, str, str]] = []

            for idx, item in enumerate(tqdm(ds, desc="Traitement du flux audio")):
                if max_samples and sample_counter >= max_samples:
                    logger.info(f"Limite maximale de {max_samples} échantillons atteinte.")
                    break

                # Extraction robuste de l'identifiant et du texte
                item_id = str(item.get("id") or item.get("path") or f"sample_{idx}")
                
                if item_id in self.processed_ids:
                    continue

                raw_text = item.get("transcription") or item.get("text") or item.get("sentence") or ""
                audio_data = item.get("audio")
                speaker_id = str(item.get("speaker_id") or item.get("client_id") or "unknown")

                if not audio_data or not raw_text:
                    continue

                # Décode l'audio dans le processus principal pour éviter les erreurs de sérialisation multiprocessing
                if not isinstance(audio_data, (dict, str, Path)):
                    try:
                        audio_data = {
                            "array": audio_data["array"],
                            "sampling_rate": audio_data["sampling_rate"]
                        }
                    except Exception as e:
                        logger.error(f"Erreur lors du décodage de l'audio dans le processus principal pour {item_id}: {e}")
                        continue

                pending_items.append((item_id, audio_data, raw_text, speaker_id))
                self.processed_ids.add(item_id)
                sample_counter += 1

                # Traitement par lots (chunking pour le multiprocessing)
                if len(pending_items) >= self.settings.batch_size:
                    results_list = pool.map(_worker_process_sample, pending_items)
                    for res in results_list:
                        batch_buffer.extend(res)
                    pending_items.clear()

                    # Écriture du lot en Parquet si le buffer a du contenu
                    if len(batch_buffer) >= self.settings.batch_size:
                        self._write_batch_parquet(batch_buffer, batch_count)
                        batch_count += 1
                        batch_buffer.clear()
                        self.save_checkpoint()

            # Traitement des derniers éléments restants dans pending_items
            if pending_items:
                results_list = pool.map(_worker_process_sample, pending_items)
                for res in results_list:
                    batch_buffer.extend(res)
                pending_items.clear()

            if batch_buffer:
                self._write_batch_parquet(batch_buffer, batch_count)
                batch_count += 1
                batch_buffer.clear()
                self.save_checkpoint()

        logger.info("Traitement par lots terminé. Génération des splits et du manifest final...")
        self._finalize_dataset(sample_counter)

    def _load_local_dataset(self):
        """Extrait l'archive locale si nécessaire et retourne un générateur d'éléments."""
        raw_dir = self.output_dir / "raw"
        tar_path = Path(self.settings.local_tar)

        if not tar_path.exists():
            raise FileNotFoundError(f"Archive locale introuvable: {tar_path}")

        # On extrait si ce n'est pas déjà fait
        if not raw_dir.exists() or not any(raw_dir.iterdir()):
            logger.info(f"Extraction de l'archive {tar_path} dans {raw_dir}...")
            raw_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=raw_dir)

        # Recherche dynamique du fichier TSV
        tsv_paths = list(raw_dir.rglob(self.settings.local_tsv_file))
        if not tsv_paths:
            raise ValueError(f"Fichier TSV '{self.settings.local_tsv_file}' introuvable dans {raw_dir}")
        tsv_path = tsv_paths[0]
        
        audio_base_dir = tsv_path.parent / self.settings.local_audio_dir
        if not audio_base_dir.exists():
            logger.warning(f"Le dossier audio '{audio_base_dir}' n'existe pas. On va chercher les audios relativement à '{tsv_path.parent}'.")
            audio_base_dir = tsv_path.parent

        logger.info(f"Fichier de mapping trouvé : {tsv_path}")
        df = pd.read_csv(tsv_path, sep="\t")

        def local_generator():
            for _, row in df.iterrows():
                audio_filename = row.get("audio_filename")
                sentence = row.get("sentence")
                
                if not audio_filename or pd.isna(audio_filename) or not sentence or pd.isna(sentence):
                    continue
                
                audio_path = audio_base_dir / str(audio_filename)
                
                yield {
                    "id": Path(audio_filename).stem,
                    "text": str(sentence),
                    "audio": str(audio_path),
                    "speaker_id": "unknown"
                }

        return local_generator()

    def _write_batch_parquet(self, data: List[Dict], batch_idx: int) -> None:
        """Écrit une liste de métadonnées de segments dans un fichier Parquet temporaire."""
        if not data:
            return
        df = pd.DataFrame(data)
        batch_path = self.batches_dir / f"batch_{batch_idx:05d}.parquet"
        df.to_parquet(batch_path, index=False)
        logger.info(f"Lot {batch_idx} sauvegardé: {len(df)} segments dans {batch_path.name}")

    def _finalize_dataset(self, total_processed: int) -> None:
        """Combine tous les lots Parquet, effectue le split Train/Val/Test et génère le manifest."""
        batch_files = list(self.batches_dir.glob("batch_*.parquet"))
        if not batch_files:
            logger.critical("CRITICAL: Aucun lot Parquet trouvé dans le dossier temporaire. Aucun segment valide n'a été produit !")
            raise ValueError("Aucun segment valide produit par le pipeline.")

        logger.info(f"Fusion de {len(batch_files)} fichiers de lots Parquet...")
        dfs = [pd.read_parquet(f) for f in batch_files]
        full_df = pd.concat(dfs, ignore_index=True)

        if full_df.empty:
            logger.critical("CRITICAL: Le DataFrame global fusionné est vide. Aucun segment valide n'a été produit !")
            raise ValueError("Aucun segment valide produit par le pipeline.")

        # Vérification critique du rendement (yield)
        n_total = len(full_df)
        if total_processed > 0:
            yield_ratio = n_total / total_processed
            if yield_ratio < 0.05:
                logger.critical(f"CRITICAL: Rendement de segments anormalement bas : {n_total} segments pour {total_processed} échantillons traités ({yield_ratio:.2%}).")
                raise ValueError(f"Rendement de segments trop faible ({yield_ratio:.2%}). Pipeline interrompu pour inspection.")

        n_total = len(full_df)

        # Partitionnement étanche par locuteur (Speaker-Disjoint Split) si disponible
        has_speaker_info = (
            "speaker_id" in full_df
            and full_df["speaker_id"].nunique() > 1
            and not (full_df["speaker_id"] == "unknown").all()
        )

        if has_speaker_info:
            logger.info("Application d'un partitionnement étanche par locuteur (Speaker-Disjoint Split)...")
            speakers = full_df["speaker_id"].unique()
            np.random.seed(self.settings.seed)
            np.random.shuffle(speakers)

            n_speakers = len(speakers)
            n_train_spk = max(1, int(n_speakers * self.settings.train_ratio))
            n_val_spk = max(1, int(n_speakers * self.settings.val_ratio)) if n_speakers > 2 else 0

            train_spks = set(speakers[:n_train_spk])
            val_spks = set(speakers[n_train_spk : n_train_spk + n_val_spk])
            test_spks = set(speakers[n_train_spk + n_val_spk :])

            train_df = full_df[full_df["speaker_id"].isin(train_spks)].reset_index(drop=True)
            val_df = full_df[full_df["speaker_id"].isin(val_spks)].reset_index(drop=True)
            test_df = full_df[full_df["speaker_id"].isin(test_spks)].reset_index(drop=True)
        else:
            # Mélange déterministe simple (fallback si aucun locuteur identifié)
            np.random.seed(self.settings.seed)
            shuffled_indices = np.random.permutation(len(full_df))
            full_df = full_df.iloc[shuffled_indices].reset_index(drop=True)

            n_train = int(n_total * self.settings.train_ratio)
            n_val = int(n_total * self.settings.val_ratio)

            train_df = full_df.iloc[:n_train].reset_index(drop=True)
            val_df = full_df.iloc[n_train : n_train + n_val].reset_index(drop=True)
            test_df = full_df.iloc[n_train + n_val :].reset_index(drop=True)

        # Sauvegarde des splits Parquet
        train_path = self.output_dir / "train.parquet"
        val_path = self.output_dir / "validation.parquet"
        test_path = self.output_dir / "test.parquet"

        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)

        logger.info(f"Splits créés: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        # Calcul des statistiques globales
        total_duration_sec = full_df["duration"].sum()
        total_hours = round(total_duration_sec / 3600.0, 3)
        mean_snr = round(float(full_df["snr_db"].mean()), 2) if "snr_db" in full_df else 0.0

        manifest = {
            "dataset_name": self.settings.hf_dataset_name,
            "config_name": self.settings.hf_config_name,
            "total_raw_samples_processed": len(self.processed_ids),
            "total_valid_segments": n_total,
            "total_duration_hours": total_hours,
            "mean_snr_db": mean_snr,
            "splits": {
                "train": {
                    "count": len(train_df),
                    "hours": round(train_df["duration"].sum() / 3600.0, 3),
                    "file": str(train_path),
                },
                "validation": {
                    "count": len(val_df),
                    "hours": round(val_df["duration"].sum() / 3600.0, 3),
                    "file": str(val_path),
                },
                "test": {
                    "count": len(test_df),
                    "hours": round(test_df["duration"].sum() / 3600.0, 3),
                    "file": str(test_path),
                },
            },
        }

        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        logger.info(f"Manifest résume sauvegardé avec succès dans {self.manifest_file}")

        # Nettoyage des fichiers temporaires Parquet de lots
        shutil.rmtree(self.batches_dir, ignore_errors=True)
        logger.info("Dossier temporaire de lots nettoyé.")
