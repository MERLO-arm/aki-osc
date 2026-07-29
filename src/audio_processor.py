"""Module de traitement et d'analyse audio pour le pipeline ASR Lingala. """
import io
import os
import uuid
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from pydub import AudioSegment
from pydub.silence import split_on_silence
import webrtcvad

logger = logging.getLogger("waxal_asr_pipeline")

# Import optionnel de TensorFlow & YAMNet pour la détection de musique
TF_HUB_AVAILABLE = False
try:
    import tensorflow as pd_tf  # type: ignore
    import tensorflow_hub as hub  # type: ignore
    TF_HUB_AVAILABLE = True
except ImportError:
    TF_HUB_AVAILABLE = False


class AudioProcessor:
    """Gestionnaire complet du prétraitement audio (formatage, VAD, SNR, détection musique, découpage)."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        min_silence_len: int = 500,
        silence_threshold_dbfs: int = -60,
        use_rms_split: bool = True,
        vad_mode: int = 3,
        min_snr_db: float = 5.0,
        min_segment_duration: float = 2.0,
        max_segment_duration: float = 25.0,
        enable_yamnet: bool = True,
    ):
        """Initialise le processeur audio avec les paramètres configurés.

        Args:
            sample_rate: Taux d'échantillonnage cible (Hz).
            channels: Canaux cibles (1 = mono).
            min_silence_len: Durée de silence minimale pour la segmentation (ms).
            silence_threshold_dbfs: Seuil de silence en dBFS.
            use_rms_split: Activer le découpage hybride RMS.
            vad_mode: Mode WebRTC VAD (0 agressivité faible à 3 très agressif).
            min_snr_db: Rapport signal/bruit minimal (dB).
            min_segment_duration: Durée minimale des segments autorisés (s).
            max_segment_duration: Durée maximale des segments autorisés (s).
            enable_yamnet: Activer la détection de musique YAMNet si disponible.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.min_silence_len = min_silence_len
        self.silence_threshold_dbfs = silence_threshold_dbfs
        self.use_rms_split = use_rms_split
        self.vad_mode = vad_mode
        self.min_snr_db = min_snr_db
        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max_segment_duration
        self.enable_yamnet = enable_yamnet and TF_HUB_AVAILABLE
        self.enable_yamnet = enable_yamnet and TF_HUB_AVAILABLE

        # VAD Initialisation
        self.vad = webrtcvad.Vad(self.vad_mode)

        # YAMNet Initialisation si disponible
        self.yamnet_model = None
        if self.enable_yamnet:
            try:
                logger.info("Chargement du modèle YAMNet via TensorFlow Hub...")
                self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
            except Exception as e:
                logger.warning(f"Impossible de charger YAMNet ({e}). La détection de musique sera désactivée.")
                self.enable_yamnet = False

    def convert_audio(self, audio_input: Union[bytes, str, Path, Dict]) -> AudioSegment:
        """Convertit une entrée audio quelconque vers un AudioSegment pydub Mono 16kHz PCM.

        Args:
            audio_input: Chemin de fichier, octets bruts, ou dictionnaire Hugging Face audio {"array", "sampling_rate"}.

        Returns:
            pydub.AudioSegment au format cible (16kHz Mono).
        """
        # Vérification si l'entrée est un dictionnaire ou un objet décodeur audio (ex: AudioDecoder avec torchcodec)
        is_audio_dict = isinstance(audio_input, dict) and "array" in audio_input
        is_audio_decoder = False
        if not is_audio_dict and hasattr(audio_input, "__getitem__") and not isinstance(audio_input, (str, bytes, list, tuple)):
            try:
                _ = audio_input["array"]
                is_audio_decoder = True
            except (KeyError, TypeError, AttributeError):
                pass

        if is_audio_dict or is_audio_decoder:
            arr = audio_input["array"]
            if is_audio_dict:
                sr = audio_input.get("sampling_rate", self.sample_rate)
            else:
                try:
                    sr = audio_input["sampling_rate"]
                except (KeyError, TypeError):
                    sr = self.sample_rate

            # Conversion numpy array 1D/2D vers int16 PCM
            if arr.dtype != np.int16:
                if np.issubdtype(arr.dtype, np.floating):
                    arr = (arr * 32767).astype(np.int16)
                else:
                    arr = arr.astype(np.int16)
            
            segment = AudioSegment(
                data=arr.tobytes(),
                sample_width=2,
                frame_rate=sr,
                channels=1 if arr.ndim == 1 else arr.shape[1],
            )
        elif isinstance(audio_input, bytes):
            segment = AudioSegment.from_file(io.BytesIO(audio_input))
        elif isinstance(audio_input, (str, Path)):
            segment = AudioSegment.from_file(str(audio_input))
        elif isinstance(audio_input, AudioSegment):
            segment = audio_input
        else:
            raise ValueError(f"Type d'entrée audio non supporté: {type(audio_input)}")

        # Application de la fréquence d'échantillonnage et des canaux cibles
        if segment.frame_rate != self.sample_rate or segment.channels != self.channels:
            segment = segment.set_frame_rate(self.sample_rate).set_channels(self.channels)

        return segment

    def compute_snr(self, segment: AudioSegment) -> float:
        """Calcule le rapport signal/bruit (SNR) estimé en dB.

        Args:
            segment: Segment audio pydub.

        Returns:
            SNR estimé en dB.
        """
        samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
        if len(samples) == 0:
            return 0.0

        # Découpage en trames de 20ms
        frame_len = int(self.sample_rate * 0.02)
        if len(samples) < frame_len:
            return 0.0

        num_frames = len(samples) // frame_len
        frames = samples[: num_frames * frame_len].reshape((num_frames, frame_len))
        frame_energies = np.mean(frames ** 2, axis=1)

        # Estimation de la puissance du bruit (15% des trames les moins énergétiques)
        # et du signal (25% des trames les plus énergétiques)
        sorted_energies = np.sort(frame_energies)
        noise_cutoff = max(1, int(num_frames * 0.15))
        signal_cutoff = max(1, int(num_frames * 0.25))

        noise_power = np.mean(sorted_energies[:noise_cutoff]) + 1e-10
        signal_power = np.mean(sorted_energies[-signal_cutoff:]) + 1e-10

        if signal_power <= noise_power:
            return 0.0

        snr_db = 10.0 * np.log10(signal_power / noise_power)
        return float(snr_db)

    def vad_filter(self, segment: AudioSegment) -> float:
        """Évalue la proportion de trames vocales via WebRTC VAD.

        Args:
            segment: Segment audio pydub 16kHz mono.

        Returns:
            Ratio de parole détecté (0.0 à 1.0).
        """
        raw_pcm = segment.raw_data
        frame_duration_ms = 30  # 30 ms
        frame_bytes = int(self.sample_rate * (frame_duration_ms / 1000.0) * 2)  # 16-bit = 2 octets

        if len(raw_pcm) < frame_bytes:
            return 0.0

        num_frames = len(raw_pcm) // frame_bytes
        voiced_frames = 0

        for i in range(num_frames):
            frame = raw_pcm[i * frame_bytes : (i + 1) * frame_bytes]
            try:
                if self.vad.is_speech(frame, self.sample_rate):
                    voiced_frames += 1
            except Exception:
                continue

        return voiced_frames / float(num_frames) if num_frames > 0 else 0.0

    def detect_music(self, segment: AudioSegment) -> bool:
        """Détecte la présence de musique dans l'audio via YAMNet (si activé).

        Args:
            segment: Segment audio pydub.

        Returns:
            True si de la musique est détectée avec forte probabilité, False sinon.
        """
        if not self.enable_yamnet or self.yamnet_model is None:
            return False

        try:
            samples = np.array(segment.get_array_of_samples(), dtype=np.float32) / 32768.0
            scores, embeddings, spectrogram = self.yamnet_model(samples)
            prediction = np.mean(scores.numpy(), axis=0)

            # Indices YAMNet typiques pour la musique / musique de fond (132 à 140 environ)
            # Indice 132: "Music", 137: "Musical instrument"
            music_score = float(prediction[132])
            return music_score > 0.4
        except Exception as e:
            logger.debug(f"Erreur détection musique YAMNet: {e}")
            return False

    def split_audio(self, segment: AudioSegment) -> List[AudioSegment]:
        """Découpe un segment audio en sous-segments au niveau des silences (Legacy helper).

        Args:
            segment: Segment audio source.

        Returns:
            Liste de sous-segments pydub.
        """
        try:
            chunks = split_on_silence(
                segment,
                min_silence_len=self.min_silence_len,
                silence_thresh=self.silence_threshold_dbfs,
                keep_silence=150,
            )
            return chunks if chunks else [segment]
        except Exception as e:
            logger.warning(f"Erreur lors du découpage audio: {e}")
            return [segment]

    def process_audio_file(
        self,
        audio_input: Union[bytes, str, Path, Dict],
        raw_text: str,
        text_cleaner_fn,
        output_audio_dir: Path,
        speaker_id: str = "unknown",
    ) -> List[Dict]:
        """Exécute le traitement complet pour un fichier ou échantillon audio.

        Args:
            audio_input: Fichier, octets, ou dictionnaire audio HF.
            raw_text: Transcription associée.
            text_cleaner_fn: Fonction de nettoyage textuel.
            output_audio_dir: Dossier d'enregistrement des fichiers WAV segmentés.
            speaker_id: Identifiant du locuteur.

        Returns:
            Liste de dictionnaires contenant les métadonnées des segments valides.
        """
        results = []
        try:
            # 1. Nettoyage initial du texte
            clean_text = text_cleaner_fn(raw_text)
            if not clean_text:
                return []

            # 2. Conversion audio vers 16kHz Mono WAV
            segment = self.convert_audio(audio_input)
            total_duration = len(segment) / 1000.0

            # Extraction robuste de l'identifiant pour le logging
            input_name = "raw_audio"
            if isinstance(audio_input, (str, Path)):
                input_name = Path(audio_input).name
            elif isinstance(audio_input, dict) and "path" in audio_input:
                input_name = Path(audio_input["path"]).name

            logger.info(f"[{input_name}] Début du traitement - Durée du fichier source : {total_duration:.2f}s")

            # Vérification de l'énergie RMS minimale
            samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
            max_val = float(1 << (8 * segment.sample_width - 1))
            samples_float = samples / max_val
            rms = np.sqrt(np.mean(samples_float ** 2)) if len(samples_float) > 0 else 0.0

            if rms < 0.001:
                logger.warning(f"[{input_name}] Audio rejeté : trop silencieux (RMS = {rms:.5f} < 0.001)")
                return []

            # 3. Découpage :
            audio_chunks = []
            use_librosa = False

            if self.use_rms_split:
                try:
                    import librosa
                    intervals = librosa.effects.split(samples_float, top_db=20)
                    librosa_chunks = []
                    durations = []
                    for start_idx, end_idx in intervals:
                        start_ms = (start_idx / self.sample_rate) * 1000.0
                        end_ms = (end_idx / self.sample_rate) * 1000.0
                        dur = (end_ms - start_ms) / 1000.0
                        librosa_chunks.append(segment[start_ms:end_ms])
                        durations.append(dur)

                    if not librosa_chunks:
                        logger.info(f"[{input_name}] Librosa n'a trouvé aucun segment. Fallback vers pydub.")
                    elif len(librosa_chunks) > 0 and (np.mean(durations) < 2.0 or sum(1 for d in durations if d < 2.0) / len(durations) > 0.5):
                        logger.info(f"[{input_name}] Librosa a produit des segments trop courts (durée moyenne: {np.mean(durations):.2f}s). Fallback vers pydub.")
                    else:
                        audio_chunks = librosa_chunks
                        use_librosa = True
                        logger.debug(f"[{input_name}] Découpage Librosa réussi : {len(audio_chunks)} segments.")
                except Exception as e:
                    logger.warning(f"[{input_name}] Erreur lors du split librosa ({e}). Fallback vers pydub.")

            if not use_librosa:
                try:
                    from pydub.silence import detect_nonsilent
                    intervals = detect_nonsilent(
                        segment,
                        min_silence_len=self.min_silence_len,
                        silence_thresh=self.silence_threshold_dbfs,
                    )
                    if intervals:
                        audio_chunks = []
                        for start_ms, end_ms in intervals:
                            start_padded = max(0, start_ms - 150)
                            end_padded = min(len(segment), end_ms + 150)
                            audio_chunks.append(segment[start_padded:end_padded])
                        logger.debug(f"[{input_name}] Découpage pydub réussi : {len(audio_chunks)} segments.")
                    else:
                        audio_chunks = [segment]
                except Exception as e:
                    logger.warning(f"[{input_name}] Erreur lors de detect_nonsilent pydub ({e}). Utilisation du segment entier.")
                    audio_chunks = [segment]

            # 4. Filtrer les segments par durée et qualité
            for chunk in audio_chunks:
                duration = len(chunk) / 1000.0
                segment_id = f"waxal_asr_{uuid.uuid4().hex}"

                # Filtrer par durée (2s à 25s)
                if duration < self.min_segment_duration or duration > self.max_segment_duration:
                    logger.info(f"[{input_name}] Segment rejeté : durée {duration:.2f}s hors limites [{self.min_segment_duration}, {self.max_segment_duration}]")
                    continue

                # Calcul du SNR
                snr_db = self.compute_snr(chunk)
                if snr_db < self.min_snr_db:
                    logger.info(f"[{input_name}] Segment rejeté : SNR trop faible ({snr_db:.2f} dB < {self.min_snr_db} dB)")
                    continue

                # VAD Filter
                voiced_ratio = self.vad_filter(chunk)
                if voiced_ratio < 0.35:
                    if duration < 3.0:
                        logger.info(f"[{input_name}] Segment rejeté : VAD faible ({voiced_ratio:.2f} < 0.35) et durée courte ({duration:.2f}s < 3.0s)")
                        continue
                    else:
                        logger.warning(f"[{input_name}] Avertissement VAD : Segment {segment_id} conservé malgré un faible VAD ({voiced_ratio:.2f}) car sa durée ({duration:.2f}s) est >= 3.0s")

                # Détection de musique (si configurée)
                if self.detect_music(chunk):
                    logger.info(f"[{input_name}] Segment rejeté : musique détectée")
                    continue

                # Sauvegarde du segment
                filename = f"{segment_id}.wav"
                filepath = output_audio_dir / filename
                chunk.export(str(filepath), format="wav")

                words = clean_text.split()
                logger.info(f"[{input_name}] Segment accepté : ID {segment_id}, durée {duration:.2f}s, SNR {snr_db:.2f} dB, VAD {voiced_ratio:.2f}")

                results.append(
                    {
                        "segment_id": segment_id,
                        "audio_filepath": str(filepath),
                        "duration": round(duration, 3),
                        "snr_db": round(snr_db, 2),
                        "vad_ratio": round(voiced_ratio, 2),
                        "text_raw": raw_text,
                        "text_clean": clean_text,
                        "word_count": len(words),
                        "speaker_id": speaker_id,
                    }
                )

        except Exception as e:
            logger.error(f"Erreur lors du traitement d'un fichier audio: {e}", exc_info=False)
            return []

        return results
