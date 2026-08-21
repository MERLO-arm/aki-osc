# 🤖 Rasa Agent — Assistant Conversationnel CALM

Un assistant conversationnel intelligent construit avec **Rasa Pro 3.16+** et l'architecture **CALM** (Conversational AI with Language Models). Il utilise des LLMs via Groq pour comprendre le langage naturel et générer des réponses contextuelles.

---

## ✨ Fonctionnalités

| Fonctionnalité | Description |
|---|---|
| 🗣️ Salutations | Accueil et présentation de l'assistant |
| ❓ Aide | Guide l'utilisateur sur les capacités du bot |
| 📋 FAQ | Affichage des questions fréquentes |
| 💬 Feedback | Collecte et traitement des retours utilisateur |
| 👤 Transfert humain | Escalade vers un agent humain avec résumé automatique de la conversation (via GPT-4o) |
| 🔁 Reformulation contextuelle | Rephrase intelligent des réponses pour un ton naturel et professionnel |

---

## 📁 Structure du projet

```
rasa-agent/
├── config.yml              # Pipeline NLU et politiques (CALM)
├── credentials.yml         # Canaux de communication (REST, Socket.IO…)
├── endpoints.yml           # Points de terminaison (actions, NLG, modèles LLM)
├── .env                    # Variables d'environnement (RASA_LICENSE, GROQ_API_KEY)
├── pyproject.toml          # Dépendances Python (uv)
│
├── data/                   # 🧠 Logique métier — Flows conversationnels
│   ├── general/            # Flows de base (hello, goodbye, help, feedback…)
│   └── system/             # Patterns système (correction, annulation…)
│
├── domain/                 # 🧩 Domaine — Slots, réponses, actions
│   ├── general/            # Domaine des flows généraux
│   └── system/             # Patterns système
│
├── actions/                # ⚙️ Actions personnalisées (Python)
│   └── action_human_handoff.py  # Résumé IA + transfert vers agent humain
│
├── prompts/                # 📝 Templates de prompts LLM
│   └── rephraser_demo_personality_prompt.jinja2  # Prompt de reformulation
│
├── models/                 # 🏋️ Modèles entraînés (.tar.gz)
├── docs/                   # 📚 Documentation et templates
└── tests/                  # 🧪 Tests end-to-end
```

---

## 🛠️ Prérequis

- **Python** ≥ 3.13
- **[uv](https://docs.astral.sh/uv/)** — gestionnaire de paquets Python
- **Clé de licence Rasa Pro** — [obtenir une Developer Edition gratuite](https://rasa.com/rasa-pro/developer-edition/)
- **Clé API Groq** — [créer un compte gratuit](https://console.groq.com)

---

## 🚀 Installation

### 1. Cloner le projet

```bash
git clone <url-du-repo>
cd rasa-agent
```

### 2. Installer les dépendances

```bash
uv sync
```

### 3. Configurer les variables d'environnement

Créer un fichier `.env` à la racine du projet :

```env
RASA_LICENSE=<votre-clé-de-licence-rasa-pro>
GROQ_API_KEY=<votre-clé-api-groq>
```

---

## ▶️ Utilisation

### Entraîner le modèle

```bash
export $(cat .env | xargs) && uv run rasa train
```

### Lancer le serveur avec l'Inspector

```bash
export $(cat .env | xargs) && uv run rasa inspect
```

Puis ouvrir dans le navigateur :
**http://localhost:5005/webhooks/inspector/inspect.html**

### Lancer le serveur en production

```bash
export $(cat .env | xargs) && uv run rasa run
```

### Tester en ligne de commande

```bash
export $(cat .env | xargs) && uv run rasa shell
```

---

## ⚙️ Configuration technique

### Pipeline NLU (`config.yml`)

| Composant | Rôle |
|---|---|
| `CompactLLMCommandGenerator` | Comprend l'intention de l'utilisateur via LLM (Groq) |
| `FlowPolicy` | Exécute les flows CALM définis dans `data/` |

### Fournisseurs LLM (`endpoints.yml`)

| Groupe de modèles | Fournisseur | Modèle | Utilisation |
|---|---|---|---|
| `groq-llama-3-3` | Groq | `openai/gpt-oss-20b` | NLU + Reformulation |
| `groq-llama-3-fast` | Groq | `openai/gpt-oss-20b` | Enterprise Search (optionnel) |
| `free-embeddings` | HuggingFace | `BAAI/bge-small-en-v1.5` | Embeddings (optionnel) |

---

## 🧪 Tests

Lancer les tests end-to-end :

```bash
export $(cat .env | xargs) && uv run rasa test e2e tests/e2e_test_cases
```

---

## 📖 Ressources

- [Documentation Rasa Pro](https://rasa.com/docs/rasa-pro/)
- [Guide CALM (Flows)](https://rasa.com/docs/pro/build/writing-flows)
- [Référence des slots](https://rasa.com/docs/reference/primitives/slots/)
- [Actions personnalisées](https://rasa.com/docs/reference/primitives/custom-actions)
- [Contextual Response Rephraser](https://rasa.com/docs/rasa-pro/concepts/contextual-response-rephraser)

---

## 📄 Licence

Ce projet utilise **Rasa Pro** sous licence commerciale. Une [Developer Edition gratuite](https://rasa.com/rasa-pro/developer-edition/) est disponible pour les développeurs individuels.
