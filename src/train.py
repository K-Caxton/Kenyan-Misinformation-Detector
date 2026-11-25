import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "model" / "afro_xlmr_fakenews"
RESULTS_DIR = BASE_DIR / "results"

MODEL_NAME = "Davlan/afro-xlmr-base"
LABEL2ID = {"neutral": 0, "fact": 1, "fake": 2}
ID2LABEL = {0: "neutral", 1: "fact", 2: "fake"}
MAX_LEN = 128


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


def train_and_evaluate():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_path = DATA_DIR / "train_clean.csv"
    test_path = DATA_DIR / "test_clean.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Run 'python src/preprocess.py' first to generate cleaned datasets.")

    full_train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df, val_df = train_test_split(
        full_train_df,
        test_size=0.1,
        random_state=42,
        stratify=full_train_df["label_id"],
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=train_df["label_id"].values,
    )
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"Computed Class Weights (neutral, fact, fake): {weights.round(3)}")

    print(f"Loading tokenizer & model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = TweetDataset(train_df["clean_text"], train_df["label_id"], tokenizer, MAX_LEN)
    val_dataset = TweetDataset(val_df["clean_text"], val_df["label_id"], tokenizer, MAX_LEN)
    test_dataset = TweetDataset(test_df["clean_text"], test_df["label_id"], tokenizer, MAX_LEN)

    use_fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(BASE_DIR / "checkpoints"),
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=use_fp16,
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting fine-tuning...")
    trainer.train()

    print(f"Saving fine-tuned model to {MODEL_DIR}...")
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    print("\nEvaluating on clean test set...")
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    y_true = test_df["label_id"].values

    report_str = classification_report(
        y_true,
        preds,
        target_names=["neutral", "fact", "fake"],
        digits=4,
    )
    cm = confusion_matrix(y_true, preds)

    print("\nTEST SET CLASSIFICATION REPORT")
    print("=" * 55)
    print(report_str)
    print("CONFUSION MATRIX (Rows=True, Cols=Pred: neutral, fact, fake)")
    print(cm)

    with open(RESULTS_DIR / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write("TEST SET CLASSIFICATION REPORT\n")
        f.write("=" * 55 + "\n")
        f.write(report_str + "\n\n")
        f.write("CONFUSION MATRIX (neutral, fact, fake)\n")
        f.write(np.array2string(cm) + "\n")

    test_df["predicted_label"] = [ID2LABEL[p] for p in preds]
    test_df["correct"] = test_df["label"] == test_df["predicted_label"]
    test_df.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    summary = {
        "model_name": MODEL_NAME,
        "test_accuracy": float(accuracy_score(y_true, preds)),
        "test_macro_f1": float(f1_score(y_true, preds, average="macro")),
        "test_weighted_f1": float(f1_score(y_true, preds, average="weighted")),
        "confusion_matrix": cm.tolist(),
    }
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll evaluation artifacts saved to {RESULTS_DIR}")


if __name__ == "__main__":
    train_and_evaluate()