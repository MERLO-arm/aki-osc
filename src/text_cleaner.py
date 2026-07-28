"""Module de nettoyage et normalisation textuelle multilingue (avec support pour les langues africaines)."""
import re
import unicodedata
from typing import Optional


class TextCleaner:
    """Classe de nettoyage et normalisation de texte adaptée aux spécificités des langues du dataset WAXAL."""

    def __init__(self, min_word_count: int = 2):
        """Initialise le nettoyeur de texte.

        Args:
            min_word_count: Nombre de mots minimum requis pour valider une transcription.
        """
        self.min_word_count = min_word_count
        # Pattern pour supprimer les métadonnées entre crochet, parenthèse ou chevron [..., (...), <...>
        self.meta_pattern = re.compile(r"\[.*?\]|\(.*?\)|<.*?>|\{.*?\}")
        # Conservation des lettres Lingala (y compris ɛ, ɔ, ŋ, et voyelles accentuées)
        # Supprime la ponctuation inutile mais garde les espaces et tirets entre mots
        self.punct_pattern = re.compile(r"[^\w\s\-\'ɛɔŋáéíóúàèìòùâêîôûãẽĩõũäëïöüñ]")
        self.multi_space_pattern = re.compile(r"\s+")

    def clean(self, text: str) -> Optional[str]:
        """Nettoie et normalise une chaîne de texte.

        Args:
            text: Transcription brute.

        Returns:
            Texte nettoyé ou None si le texte est invalide/trop court.
        """
        if not text or not isinstance(text, str):
            return None

        # 1. Normalisation Unicode (NFC)
        cleaned = unicodedata.normalize("NFC", text)

        # 2. Suppression des balises de métadonnées [rires], (silence), etc.
        cleaned = self.meta_pattern.sub(" ", cleaned)

        # 3. Conversion en minuscules
        cleaned = cleaned.lower()

        # 4. Suppression de la ponctuation hors caractères autorisés
        cleaned = self.punct_pattern.sub(" ", cleaned)

        # 5. Normalisation des espaces
        cleaned = self.multi_space_pattern.sub(" ", cleaned).strip()

        # 6. Filtrage sur le nombre de mots minimal
        words = cleaned.split()
        if len(words) < self.min_word_count:
            return None

        return cleaned
