import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
import evaluate
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# 1. Configuration des métriques
bleu_metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
        
    # Remplacer les -100 dans les preds et les labels pour éviter les erreurs d'overflow de type dans le décodeur rust
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Nettoyage
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    
    result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

# 2. Paramètres et Chargement du Tokenizer
model_id = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Spécification des langues
tokenizer.src_lang = "fuv_Latn"  # Fulfuldé
tokenizer.tgt_lang = "fra_Latn"  # Français

def preprocess_function(examples):
    inputs = [ex for ex in examples["fulfulde"]]
    targets = [ex for ex in examples["french"]]
    
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

def main():
    print("=== Fine-tuning NLLB-200 (Fulfuldé ➔ Français) ===")
    
    # 3. Charger le corpus parallèle compilé
    csv_path = "data/compiled/nllb_parallel_corpus.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Veuillez d'abord générer le corpus compilé à l'aide de compile_dataset.py.")

    df = pd.read_csv(csv_path)
    print(f"Dataset chargé : {len(df)} paires uniques.")
    
    # Si un deuxième CSV de Hugging Face traduit existe, on le fusionne
    hf_csv = "data/compiled/hf_adamawa_parallel.csv"
    if os.path.exists(hf_csv):
        df_hf = pd.read_csv(hf_csv)
        df = pd.concat([df, df_hf], ignore_index=True).drop_duplicates(subset=["french", "fulfulde"])
        print(f"Fusion avec le dataset Hugging Face effectué. Total : {len(df)} paires uniques.")

    # Mélanger les données
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Convertir en Dataset Hugging Face
    dataset = Dataset.from_pandas(df)
    
    # Split Train/Val
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
    
    # Prétraitement / Tokenisation
    print("Tokenisation du dataset...")
    train_tokenized = train_dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    val_tokenized = val_dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

    # 4. Charger le modèle et appliquer LoRA
    print("Chargement du modèle de base NLLB-200...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)

    # Configuration LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
    )
    
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # 5. Configuration de l'entraînement
    training_args = Seq2SeqTrainingArguments(
        output_dir="outputs/nllb_fulfulde_lora",
        eval_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),  # fp16 sur GPU uniquement
        use_cpu=not torch.cuda.is_available(),  # Forcer le CPU sur Mac pour éviter MPS OOM
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        report_to="none"  # Désactiver mlflow/wandb pour éviter les blocages
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 6. Lancer l'entraînement
    print("Démarrage du Fine-tuning...")
    trainer.train()
    
    # 7. Sauvegarder l'adaptateur LoRA entraîné
    print("Sauvegarde du modèle entraîné...")
    trainer.save_model("outputs/nllb_fulfulde_lora_final")
    print("Terminé avec succès !")

if __name__ == "__main__":
    main()
