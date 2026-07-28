"""Tests unitaires Pytest pour le nettoyeur de texte et le processeur audio."""
import numpy as np
import pytest
from pydub import AudioSegment

from src.text_cleaner import TextCleaner
from src.audio_processor import AudioProcessor


class TestTextCleaner:
    """Tests unitaires pour la classe TextCleaner."""

    def test_unicode_and_lowercasing(self):
        cleaner = TextCleaner(min_word_count=1)
        text = "MBÒTÉ BISÓ NA BISÓ"  # Décomposé NFD
        result = cleaner.clean(text)
        assert result is not None
        assert "mbóté" in result or "mbòté" in result or "mbote" in result

    def test_lingala_special_characters_preservation(self):
        cleaner = TextCleaner(min_word_count=1)
        text = "Nazalí na ndáko, nazalí komela mái ɛ na ɔ."
        result = cleaner.clean(text)
        assert result is not None
        assert "ɛ" in result
        assert "ɔ" in result
        assert "nazalí" in result

    def test_metadata_removal(self):
        cleaner = TextCleaner(min_word_count=2)
        text = "[rires] Mbote na bino (applause) <noise> botandi 123!"
        result = cleaner.clean(text)
        assert result is not None
        assert "rires" not in result
        assert "applause" not in result
        assert "noise" not in result
        assert "mbote na bino" in result

    def test_min_word_count_filter(self):
        cleaner = TextCleaner(min_word_count=3)
        assert cleaner.clean("moke") is None
        assert cleaner.clean("moke mpenza") is None
        assert cleaner.clean("moke mpenza mingi") is not None


class TestAudioProcessor:
    """Tests unitaires pour la classe AudioProcessor."""

    @pytest.fixture
    def processor(self):
        return AudioProcessor(
            sample_rate=16000,
            channels=1,
            enable_yamnet=False,
        )

    def test_convert_audio_format(self, processor):
        # Création d'un AudioSegment synthétique 44.1kHz Stereo
        sample_rate = 44100
        duration_ms = 1000
        samples = (np.sin(2 * np.pi * 440 * np.linspace(0, 1, sample_rate)) * 32767).astype(np.int16)
        stereo_samples = np.column_stack((samples, samples))

        raw_segment = AudioSegment(
            data=stereo_samples.tobytes(),
            sample_width=2,
            frame_rate=sample_rate,
            channels=2,
        )

        converted = processor.convert_audio(raw_segment)
        assert converted.frame_rate == 16000
        assert converted.channels == 1

    def test_compute_snr_pure_signal_vs_noise(self, processor):
        # 0.5s sinusoïde forte (parole) + 0.5s silence (bruit de fond)
        t_speech = np.linspace(0, 0.5, 8000)
        speech_signal = (np.sin(2 * np.pi * 440 * t_speech) * 20000).astype(np.int16)
        silence_signal = np.zeros(8000, dtype=np.int16)
        combined_signal = np.concatenate([speech_signal, silence_signal])
        high_snr_segment = AudioSegment(data=combined_signal.tobytes(), sample_width=2, frame_rate=16000, channels=1)

        snr_high = processor.compute_snr(high_snr_segment)
        assert snr_high > 10.0

    def test_vad_filter_pure_silence(self, processor):
        # Silence pur (octets à 0)
        silence_bytes = b"\x00" * (16000 * 2)  # 1 seconde de silence 16-bit
        silence_segment = AudioSegment(data=silence_bytes, sample_width=2, frame_rate=16000, channels=1)

        voiced_ratio = processor.vad_filter(silence_segment)
        assert voiced_ratio == 0.0
