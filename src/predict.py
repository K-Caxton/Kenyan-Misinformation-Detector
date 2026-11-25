import re
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model" / "afro_xlmr_fakenews"


def clean_input(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@user\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class FakeNewsPredictor:
    def __init__(self, model_path: Path = MODEL_DIR):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Fine-tuned model not found at {model_path}. Run 'python src/train.py' first."
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict(self, text: str) -> dict:
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

        return {
            "original_text": text,
            "cleaned_text": cleaned,
            "prediction": best_label,
            "confidence": scores[best_label],
            "probabilities": scores,
        }


if __name__ == "__main__":
    predictor = FakeNewsPredictor()
    sample = "Ignore the fake sponsored polls, we are winning this by over 63 percent!"
    print(predictor.predict(sample))