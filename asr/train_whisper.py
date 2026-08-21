import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
import torchaudio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# 1. Configuration
MODEL_ID = "openai/whisper-base"  # Alignement sur le modèle de base que vous avez entraîné hier
AUDIO_DIR = "data/fulfulde_run/raw/Adamawa-Fulfulde-TTS-Dataset/audio_files_mp3"
TSV_PATH = "data/fulfulde_run/raw/Adamawa-Fulfulde-TTS-Dataset/Mapping_MP3_translated.tsv"
OUTPUT_DIR = "outputs/whisper_fulfulde_lora"

print("Initialisation des processeurs Whisper...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
# On utilise le token de langue "ha" (Hausa) ou "en" comme fallback car le Fulfulde utilise l'alphabet latin 
# et n'est pas nativement supporté dans la liste figée des langues par défaut de Whisper.
tokenizer = WhisperTokenizer.from_pretrained(MODEL_ID, language="ha", task="transcribe")
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="ha", task="transcribe")

# 2. Chargement du dataset
print("Chargement des données depuis le fichier TSV...")
if not os.path.exists(TSV_PATH):
    raise FileNotFoundError(f"Fichier TSV introuvable : {TSV_PATH}")

df = pd.read_csv(TSV_PATH, sep="\t")
# Supprimer les lignes sans audio ou sans transcription
df = df.dropna(subset=["audio_filename", "sentence"])

# Créer le chemin complet vers l'audio
df["audio_path"] = df["audio_filename"].apply(lambda x: os.path.join(AUDIO_DIR, x))

# Filtrer pour s'assurer que les fichiers audio existent réellement
df = df[df["audio_path"].apply(os.path.exists)]
print(f"Nombre d'échantillons audio trouvés : {len(df)}")

# Diviser en Train (90%) et Validation (10%)
df_train = df.sample(frac=0.9, random_state=42)
df_eval = df.drop(df_train.index)

train_dataset = Dataset.from_pandas(df_train)
eval_dataset = Dataset.from_pandas(df_eval)

# 3. Fonction de prétraitement
def prepare_dataset(batch):
    # Charger l'audio et le ré-échantillonner à 16kHz
    audio_path = batch["audio_path"]
    try:
        speech, sr = torchaudio.load(audio_path)
        speech = torchaudio.functional.resample(speech, sr, 16000).squeeze().numpy()
    except Exception as e:
        print(f"Erreur de lecture audio {audio_path}: {e}")
        speech = np.zeros(16000)
    
    # Extraire les caractéristiques log-mel
    batch["input_features"] = feature_extractor(speech, sampling_rate=16000).input_features[0]
    
    # Tokeniser la transcription en Fulfuldé
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

# Pour un dry-run rapide local, on peut restreindre le dataset
if os.environ.get("DRY_RUN", "False") == "True":
    print("Mode DRY_RUN activé : réduction du dataset à 4 échantillons.")
    train_dataset = train_dataset.select(range(4))
    eval_dataset = eval_dataset.select(range(2))

print("Prétraitement du dataset (extraction audio et tokenisation)...")
train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(prepare_dataset, remove_columns=eval_dataset.column_names)

# 4. Data Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Séparer les caractéristiques d'entrée et les labels de transcription
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Remplacer le rembourrage de labels par -100 pour l'ignorer dans la perte
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Si le bos token est présent au début, le retirer car le modèle l'ajoute automatiquement
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 5. Métriques (WER)
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Remplacer -100 par pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Décoder les prédictions et les labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# 6. Chargement et configuration du modèle
print("Chargement du modèle de base Whisper...")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Configuration LoRA
print("Configuration de LoRA...")
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

eval_strategy = "steps" if os.environ.get("DRY_RUN", "False") == "False" else "no"
# Convertir pour compatibilité locale si nécessaire
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_steps=100,
    max_steps=1000 if os.environ.get("DRY_RUN", "False") == "False" else 5,
    evaluation_strategy=eval_strategy,
    eval_steps=200,
    save_steps=200,
    logging_steps=25,
    report_to="none",
    predict_with_generate=True,
    generation_max_length=225,
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
    label_names=["labels"],
)

# 8. Entraînement
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

print("Démarrage de l'entraînement Whisper LoRA...")
trainer.train()

print("Sauvegarde du modèle Whisper LoRA...")
model.save_pretrained(f"{OUTPUT_DIR}_final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}_final")
print("Entraînement terminé avec succès !")
