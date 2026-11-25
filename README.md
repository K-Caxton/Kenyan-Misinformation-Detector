# 🇰🇪 Kenyan Political Misinformation Detector

An AI-powered NLP system fine-tuned on **`Davlan/afro-xlmr-base`** to detect political misinformation in Kenyan social media posts written in **Kiswahili, Sheng, and English**.


## Problem & Motivation
Political misinformation spreads rapidly across Kenyan platforms like X (Twitter) and WhatsApp during major political events. Most existing fact-checking models only understand standard English and struggle with Kenyan code-switching, Sheng slang, and Kiswahili morphology. This system classifies posts into three categories:
* **Neutral:** General political opinion, commentary, or banter
* **Fact:** Verifiable, accurate political statements
* **Fake:** Fabricated or misleading claims

## Dataset & Preprocessing
* **Source:** `lusamaki/Fake_News_Detection_System_29.5k` (`29,517` raw Kenyan political tweets)
* **Cleaning & Leakage Prevention:** Stripped URLs, `@mentions`, spreadsheet `#ERROR!` artifacts, and posts shorter than two words. Removed exact duplicate tweets and cross-split train/test overlaps to prevent data leakage.
* **Test Split:** `2,951` strictly unseen tweets (`1,766` Neutral, `729` Fact, `456` Fake).

## Model & Training Approach
* **Backbone:** `Davlan/afro-xlmr-base` (pre-trained across 17 African languages including Kiswahili + English).
* **Handling Class Imbalance:** Applied dampened square-root inverse class weights (`[0.75, 1.16, 1.47]`) and label smoothing (`0.1`) so the model catches minority **Fake** and **Fact** tweets without over-flagging normal **Neutral** opinions.

## Evaluation Results (Unseen Test Set)
| Metric | Score |
| :--- | :---: |
| **Test Accuracy** | **61.10%** |
| **Weighted F1-Score** | **0.6217** |
| **Macro F1-Score** | **0.5608** |

## Project Structure
```text
├── app/
│   └── app.py          # Streamlit web interface (Single-tweet & Batch CSV auditing)
├── src/
│   └── predict.py      # Offline inference engine
├── model/              # Saved fine-tuned model weights & tokenizer
└── results/
    └── metrics.json    # Evaluation metrics