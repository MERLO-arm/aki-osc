# Sources de Données pour le Fulfuldé de l'Adamaoua, Cameroun (code : fub / fuv)

Ce document récapitule les gisements de données textuelles et audio disponibles pour enrichir les modèles de traduction (NLLB-200) et de reconnaissance vocale (Whisper ASR), tout en respectant les contraintes juridiques et de registre de langue.

---

## 🚫 Recommandation Légale et Registre de Langue
* **Watch Tower (JW.org)** : Ne **pas** scraper massivement en raison des restrictions strictes de copyright ( Watch Tower Bible and Tract Society). Utiliser uniquement pour de la validation lexicale manuelle. De plus, son registre ultra-formel et écrit risque de biaiser le voice-bot par rapport à la parole spontanée.
* **Priorité** : Privilégier les corpus sous licence libre (Mozilla, Meta MMS, Google Fleurs) et la collecte directe de terrain via **KoboToolbox** pour capturer la parole naturelle.

---

## 1. Répertoires de Recherche Linguistique et Archives Académiques
* **Pangloss Collection (CNRS/LACITO)** : Enregistrements audio haute fidélité de récits traditionnels fulfulde avec transcriptions interlinéaires au format XML.
* **ELAR (Endangered Languages Archive)** : Dépôts de collectes de terrain au Cameroun/Nigeria (heures de parole naturelle annotée).
* **SIL Language & Culture Archives (sil.org)** : Dictionnaires, guides grammaticaux et livrets d'alphabétisation en PDF.
* **CABTAL** : Textes communautaires, lexiques spécialisés pour le Grand-Nord Cameroun.
* **SUDOC / HAL Open Science** : Mémoires et thèses des universités de Ngaoundéré et Maroua contenant des contes retranscrits.

---

## 2. Dépôts NLP et Apprentissage Automatique (ASR / Traduction)

| Nom de la Source | Type de données | Format & Accès |
| :--- | :--- | :--- |
| **Meta MMS** | Audio + Transcriptions alignées | Hugging Face (`fub_cm` Cameroun) |
| **Google Fleurs** | Audio studio + Texte lu | Hugging Face (`fub` ou `ff_sn`) |
| **Common Voice (Mozilla)** | Audio participatif + Transcriptions | Téléchargement direct TSV + MP3 |
| **AfriSpeech-200** | Parole spontanée et lue | Dépôts GitHub Masakhane |
| **NLLB-200 (Meta AI)** | Corpus bilingues texte (Parallèle) | Hugging Face (Données d'entraînement) |
| **OPUS Corpus** | Paires de phrases multilingues | Téléchargement (TMX, MOSES) |
| **Leipzig Corpora Collection**| Phrases monolingues du web | Fichiers texte brut TXT |

---

## 3. Plateformes Religieuses et Littérature Orale (Licences Spécifiques)
* **Bible.is (Faith Comes By Hearing)** : Audio de l'intégralité du Nouveau et Ancien Testament en Fulfulde de l'Adamaoua, découpé par versets. Plus permissif pour usage de recherche.
* **ScriptureEarth (iso=fub)** : PDF de contes, guides de santé, chants traditionnels et enregistrements associés.
* **Global Recordings Network (GRN)** : Séries d'audios thématiques (Paroles de Vie, Bonnes Nouvelles) avec transcriptions textuelles.
* **YouVersion Bible App** : Plusieurs versions du texte biblique en Fulfulde camerounais.
* **Jesus Film Project (API / Web)** : Pistes vocales synchronisées de doublages en Fulfulde.

---

## 4. Médias d'Information, Radio et Audio Spontané (Grand-Nord Cameroun)
* **CRTV Régionales (Garoua, Maroua, Ngaoundéré)** : Journaux parlés et magazines ruraux diffusés en Fulfulde (fichiers audio capturables via web-radio). Excellent registre de parole naturelle.
* **RFI Fulfulde** : Articles d'actualité textuels et podcasts quotidiens d'information.
* **BBC News Hausa / Fulfulde** : Dépêches d'actualité, reportages vidéo et transcriptions.
* **Deutsche Welle (DW) Fulfulde** : Émissions radiophoniques et articles web d'actualité.
* **Kassida TV / Web-télés locales du Grand-Nord** : Débats de société, causeries éducatives et sermons diffusés sur Facebook et YouTube.

---

## 5. Projets Encyclopédiques et Communautaires (Monolingue)
* **Wikipédia en Fulfulde (ff.wikipedia.org)** : Dumps complets XML/SQL téléchargeables contenant des milliers d'articles encyclopédiques.
* **Wiktionnaire (ff.wiktionary.org)** : Dictionnaires collaboratifs avec transcriptions phonétiques (API Wikimedia).
* **Tatoeba Project** : Base de données de phrases courantes traduites et enregistrées par des locuteurs natifs.
* **Resulam** : Livres numériques, lexiques illustrés et vidéos pédagogiques axés sur les langues du Cameroun.
