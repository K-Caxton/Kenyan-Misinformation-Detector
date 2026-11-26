import re
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_MODEL_DIR = BASE_DIR / "model" / "afro_xlmr_fakenews"
HF_REPO_ID = "YOUR_HF_USERNAME/ukwelicheck-afroxlmr"  # <-- Replace YOUR_HF_USERNAME

MODEL_SOURCE = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else HF_REPO_ID


def clean_input(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@user\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class FakeNewsPredictor:
    def __init__(self, model_path: str = MODEL_SOURCE):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label
        self.label2id = {v: k for k, v in self.id2label.items()}

    def predict(self, text: str, explain: bool = False) -> dict:
        cleaned = clean_input(text)
        inputs = self.tokenizer(
            cleaned,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu().tolist()

        scores = {self.id2label[i]: round(probs[i] * 100, 2) for i in range(len(probs))}
        best_label = max(scores, key=scores.get)

        word_impacts = []
        if explain and len(cleaned.split()) > 1:
            word_impacts = self._explain_words(
                cleaned, self.label2id[best_label], probs[self.label2id[best_label]]
            )

        return {
            "original_text": text,
            "cleaned_text": cleaned,
            "prediction": best_label,
            "confidence": scores[best_label],
            "probabilities": scores,
            "word_impacts": word_impacts,
        }

    def _explain_words(
        self, cleaned_text: str, target_idx: int, base_prob: float
    ) -> list[tuple[str, float]]:
        """Leave-one-out perturbation to find words driving the predicted class."""
        words = cleaned_text.split()
        if len(words) > 40:
            return []

        occluded_texts = [
            " ".join(words[:i] + words[i + 1 :]) for i in range(len(words))
        ]
        inputs = self.tokenizer(
            occluded_texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[:, target_idx].cpu().tolist()

        impacts = []
        for word, new_prob in zip(words, probs):
            drop = round((base_prob - new_prob) * 100, 2)
            impacts.append((word, drop))
        return impacts

    def predict_batch(
        self, texts: list[str], batch_size: int = 32, progress_callback=None
    ) -> list[dict]:
        """Fast batched inference for CSV auditing."""
        results = []
        total = len(texts)

        for start in range(0, total, batch_size):
            batch_raw = texts[start : start + batch_size]
            batch_clean = [
                clean_input(t) if isinstance(t, str) else "" for t in batch_raw
            ]

            inputs = self.tokenizer(
                batch_clean,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs_batch = torch.softmax(outputs.logits, dim=-1).cpu().tolist()

            for cleaned, probs in zip(batch_clean, probs_batch):
                if not cleaned:
                    results.append({"prediction": "neutral", "confidence": 0.0})
                else:
                    scores = {
                        self.id2label[i]: round(probs[i] * 100, 2)
                        for i in range(len(probs))
                    }
                    best_label = max(scores, key=scores.get)
                    results.append(
                        {"prediction": best_label, "confidence": scores[best_label]}
                    )

            if progress_callback:
                progress_callback(min((start + batch_size) / total, 1.0))

        return results


if __name__ == "__main__":
    predictor = FakeNewsPredictor()
    sample = "Ignore the fake sponsored polls, we are winning this by over 63 percent!"
    print(predictor.predict(sample, explain=True))