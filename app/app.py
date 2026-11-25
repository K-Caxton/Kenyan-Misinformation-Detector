import json
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from predict import FakeNewsPredictor  # noqa: E402

st.set_page_config(
    page_title="Kenyan Political Misinformation Detector",
    page_icon="🇰🇪",
    layout="centered",
)

st.title("🇰🇪 African Language Fake News Detector")
st.caption(
    "Fine-tuned **AfroXLMR-Base** for detecting misinformation in code-switched "
    "**Kiswahili, Sheng, and Kenyan English** social media posts."
)


@st.cache_resource
def load_predictor():
    return FakeNewsPredictor()


try:
    predictor = load_predictor()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

tab1, tab2 = st.tabs(["Single-Tweet Fact-Checker", "Batch CSV Auditing"])

# --- TAB 1: SINGLE-TWEET FACT-CHECKER ---
with tab1:
    SAMPLES = {
        "Select a sample post...": "",
        "Sample 1 (Sheng / Kiswahili Commentary)": "after elections nipateni brazil, sii angechoma id basi",
        "Sample 2 (Misinformation / Fake Claim)": "ignore the fake sponsored polls by tifa, we are winning this by over 63 percent of total votes",
        "Sample 3 (Civic / Factual Statement)": "we have to say no to violence, let your neighbors peace be your responsibility during elections",
    }

    selected_sample = st.selectbox("Try a sample Kenyan social media post:", list(SAMPLES.keys()))
    user_text = st.text_area(
        "Or enter any tweet / news claim (Kiswahili, Sheng, or English):",
        value=SAMPLES[selected_sample],
        height=120,
        placeholder="Type or paste text here...",
    )

    if st.button("Analyze Claim", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            result = predictor.predict(user_text)
            pred = result["prediction"].upper()
            conf = result["confidence"]

            st.subheader("Classification Result")
            if pred == "FAKE":
                st.error(f"**Prediction: FAKE / MISINFORMATION** ({conf}% confidence)")
            elif pred == "FACT":
                st.success(f"**Prediction: FACT / VERIFIABLE** ({conf}% confidence)")
            else:
                st.info(f"**Prediction: NEUTRAL / OPINION** ({conf}% confidence)")

            st.write("**Class Probability Breakdown:**")
            prob_df = pd.DataFrame(
                {
                    "Category": [k.capitalize() for k in result["probabilities"].keys()],
                    "Confidence (%)": list(result["probabilities"].values()),
                }
            ).set_index("Category")
            st.bar_chart(prob_df)

# --- TAB 2: BATCH CSV AUDITING ---
with tab2:
    st.subheader("Batch CSV Auditing")
    st.write("Upload a CSV file of social media posts to classify all rows at once and download a report.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV file: {e}")
            st.stop()

        if df.empty:
            st.warning("The uploaded CSV file is empty.")
        else:
            st.write("**Preview of uploaded data:**", df.head(5))
            text_col = st.selectbox("Select the column containing the tweet / post text:", df.columns)

            if st.button("Classify All Posts", type="primary", use_container_width=True):
                predictions = []
                confidences = []
                progress_bar = st.progress(0)
                total_rows = len(df)

                for idx, raw_text in enumerate(df[text_col]):
                    clean_str = "" if pd.isna(raw_text) else str(raw_text).strip()
                    if not clean_str:
                        predictions.append("NEUTRAL")
                        confidences.append("0.00%")
                    else:
                        res = predictor.predict(clean_str)
                        predictions.append(res["prediction"].upper())
                        confidences.append(f"{res['confidence']}%")
                    progress_bar.progress((idx + 1) / total_rows)

                result_df = df.copy()
                result_df["Predicted_Label"] = predictions
                result_df["Confidence"] = confidences

                st.success(f"Classified {total_rows} posts successfully!")

                # Summary counts
                counts = result_df["Predicted_Label"].value_counts()
                c1, c2, c3 = st.columns(3)
                c1.metric("Neutral", int(counts.get("NEUTRAL", 0)))
                c2.metric("Fact", int(counts.get("FACT", 0)))
                c3.metric("Fake", int(counts.get("FAKE", 0)))

                st.dataframe(result_df, use_container_width=True)

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Classified CSV Report",
                    data=csv_bytes,
                    file_name="classified_kenyan_tweets.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

with st.expander("Model & Project Details"):
    st.markdown(
        """
- **Base Model:** `Davlan/afro-xlmr-base` (17 African Languages + English)
- **Dataset:** `lusamaki/Fake_News_Detection_System_29.5k` (Cleaned & Deduplicated)
- **Imbalance Handling:** Dampened square-root class-weighted Cross-Entropy Loss + Label Smoothing
        """
    )
    metrics_file = BASE_DIR / "results" / "metrics.json"
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        col1, col2, col3 = st.columns(3)
        col1.metric("Test Accuracy", f"{metrics['test_accuracy']*100:.2f}%")
        col2.metric("Macro F1-Score", f"{metrics['test_macro_f1']*100:.2f}%")
        col3.metric("Weighted F1", f"{metrics['test_weighted_f1']*100:.2f}%")