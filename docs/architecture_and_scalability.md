# Architecture et Stratégie de Scalabilité : Voicebot & Agent IA Puissant

Ce document détaille les choix d'architecture technique, d'optimisation de latence et de scalabilité pour déployer le voicebot multilingue (Fulfuldé/Lingala/Français) en production pour des millions d'utilisateurs.

---

## 1. Vue d'Ensemble de l'Architecture (Flux Temps Réel)

Pour garantir une expérience utilisateur fluide, la latence globale (Time-To-First-Byte audio) doit rester sous la barre des **1,5 seconde**.

```
[Utilisateur (Voix)]
       │
       ▼
 ┌───────────┐         ┌────────────────────────┐
 │ Téléphone │ ──────> │  Passerelle SIP/VoIP   │ (ex: Twilio / Vapi)
 └───────────┘         └────────────────────────┘
                                  │
                                  ▼ (Flux Audio en continu gRPC / WebSockets)
                       ┌────────────────────────┐
                       │   ASR (Speech-to-Text) │ ──> Whisper-Base (LoRA)
                       └────────────────────────┘
                                  │
                                  ▼ (Texte localisé : Fulfuldé)
                       ┌────────────────────────┐
                       │   Traduction (NLP)     │ ──> NLLB-200 (LoRA)
                       └────────────────────────┘
                                  │
                                  ▼ (Texte standard : Français)
                       ┌────────────────────────┐
                       │  Dialogue (Agent RASA) │
                       └────────────────────────┘
                                  │
                                  ▼ (Texte Réponse : Français)
                       ┌────────────────────────┐
                       │   Traduction (NLP)     │ ──> NLLB-200 (Français ➔ Fulfuldé)
                       └────────────────────────┘
                                  │
                                  ▼ (Texte Réponse : Fulfuldé)
                       ┌────────────────────────┐
                       │   TTS (Text-to-Speech) │ ──> XTTS / Deepgram
                       └────────────────────────┘
                                  │
                                  ▼ (Flux Audio MP3 / PCM)
[Utilisateur (Voix)] <────────────────┘
```

---

## 2. Stratégies de Scalabilité par Composant

### 🚀 A. Reconnaissance Vocale (ASR - Whisper)
* **Serveur d'Inference Dédié** : Ne pas exécuter Whisper directement dans l'application web. Déployer Whisper sur **Triton Inference Server** (NVIDIA) ou utiliser **Faster-Whisper** compilé en C++ (CTranslate2) pour diviser la latence par 4.
* **Streaming ASR** : Analyser le flux audio par blocs de 500ms (chunk-based processing) via WebSockets pour commencer la transcription avant même que l'utilisateur ait fini de parler.
* **Batching Dynamique** : Configurer Triton pour regrouper les requêtes audio simultanées sur les GPU afin d'augmenter le débit (throughput).

### 🔄 B. Traduction (NLP - NLLB-200)
* **Mise en cache Redis (Crucial)** : 80% des requêtes des utilisateurs concernent 20% des phrases courantes (salutations, questions médicales fréquentes). Nous mettons en place un cache **Redis** ultra-rapide. Si la traduction existe en cache, la latence est de **<5ms** (pas de passage par le GPU).
* **Optimisation de modèle** : Convertir NLLB-200 au format **ONNX Runtime** ou **TensorRT-LLM** avec quantification FP16 ou INT8 pour réduire l'empreinte VRAM et diviser la latence de traduction par 3.

### 🧠 C. Gestionnaire de Dialogue (Rasa Agent)
* **Déploiement Apatride (Stateless)** : Rasa doit fonctionner de manière totalement horizontale (sans état local) derrière un Load Balancer.
* **Tracker Store externe (PostgreSQL)** : L'historique des conversations est stocké dans une base PostgreSQL managée pour que n'importe quel conteneur Rasa puisse répondre à n'importe quel utilisateur.
* **Lock Store Redis** : Pour éviter les collisions lorsque l'utilisateur parle rapidement (double requêtes), nous utilisons Redis comme verrou de session.

### 🗣️ D. Synthèse Vocale (TTS)
* **Pré-génération du Statique (CDN)** : Toutes les réponses statiques du bot (ex: *"Bonjour, comment puis-je vous aider ?"*) sont générées à l'avance et stockées sur un **CDN (Cloudflare / AWS S3)**. Le bot renvoie directement le fichier audio statique sans calcul GPU (latence zéro).
* **Streaming TTS** : Pour les réponses dynamiques (ex: diagnostic personnalisé), envoyer le flux audio généré par morceaux (chunk streaming) pour que l'utilisateur commence à entendre la voix pendant que la fin du texte est encore en cours de synthèse.

---

## 3. Stratégie de Monitoring & Alignement Continu
1. **Observabilité (Prometheus + Grafana)** : Suivre en temps réel le temps de calcul de chaque étape (ASR, Traduction, NLU, TTS).
2. **Collecte de données de terrain (KoboToolbox)** : Les échecs de compréhension sont stockés automatiquement dans une base de données d'amélioration. Chaque mois, le modèle Whisper et NLLB sont ré-entraînés automatiquement (pipeline CI/CD) sur ces nouvelles données réelles pour augmenter continuellement le score BLEU et abaisser le WER.
