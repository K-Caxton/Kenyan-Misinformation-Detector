import html
import json
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from predict import FakeNewsPredictor  # noqa: E402

st.set_page_config(
    page_title="UkweliCheck | Kenyan Misinformation Lab",
    page_icon="🇰🇪",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- CUSTOM KENYAN EDITORIAL THEME ---
st.markdown(
    """
    <style>
    /* Hide default sidebar toggle for a clean centered editorial layout */
    [data-testid="collapsedControl"] { display: none; }

    .block-container {
        max-width: 860px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    /* Editorial Masthead Banner */
    .masthead {
        border-left: 5px solid #dc2626;
        padding: 0.4rem 0 0.4rem 1.1rem;
        margin-bottom: 1.5rem;
    }
    .masthead-tag {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.75rem;
        font-weight: 700;
        color: #dc2626;
        margin-bottom: 0.2rem;
    }
    .masthead-title {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.15;
        margin: 0;
    }
    .masthead-sub {
        font-size: 0.95rem;
        opacity: 0.8;
        margin-top: 0.35rem;
    }

    /* Primary Crimson/Kenyan Red Action Button */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #b91c1c 0%, #dc2626 100%);
        border: none;
        color: #ffffff;
        border-radius: 6px;
        font-weight: 700;
        letter-spacing: 0.03em;
        padding: 0.55rem 1.2rem;
        transition: transform 0.1s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #991b1b 0%, #b91c1c 100%);
        color: #ffffff;
    }

    /* Verdict Scorecard */
    .verdict-banner {
        padding: 1.2rem 1.4rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 6px solid;
        background-color: rgba(128, 128, 128, 0.08);
    }
    .verdict-fake { border-left-color: #ef4444; background-color: rgba(239, 68, 68, 0.1); }
    .verdict-fact { border-left-color: #10b981; background-color: rgba(16, 185, 129, 0.1); }
    .verdict-neutral { border-left-color: #f59e0b; background-color: rgba(245, 158, 11, 0.1); }

    .verdict-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
        opacity: 0.85;
    }
    .verdict-main {
        font-size: 1.45rem;
        font-weight: 800;
        margin: 0.15rem 0;
    }

    /* Token Explainability Box */
    .token-box {
        padding: 1rem;
        border-radius: 8px;
        border: 1px dashed rgba(128, 128, 128, 0.4);
        line-height: 1.9;
        font-size: 0.98rem;
        margin-top: 0.5rem;
    }

    /* Footer */
    .editorial-footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(128, 128, 128, 0.2);
        font-size: 0.82rem;
        opacity: 0.7;
        display: flex;
        justify-content: space-between;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor():
    return FakeNewsPredictor()


try:
    predictor = load_predictor()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# --- TOP EDITORIAL MASTHEAD ---
st.markdown(
    """
    <div class="masthead">
        <div class="masthead-tag">🇰🇪 Kenyan Civic NLP &bull; AfroXLMR-Base</div>
        <h1 class="masthead-title">UkweliCheck Misinformation Lab</h1>
        <div class="masthead-sub">
            Automated claim verification for code-switched <b>Kiswahili</b>, <b>Sheng</b>, and <b>Kenyan English</b> social media posts.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- NAVIGATION TABS ---
tab_verify, tab_audit, tab_specs = st.tabs(
    ["Verify a Claim", "Bulk CSV Audit", "Model & Benchmarks"]
)

# ==========================================
# TAB 1: SINGLE CLAIM VERIFICATION
# ==========================================
with tab_verify:
    SAMPLES = {
        "Sheng / Political": "after elections nipateni brazil, sii angechoma id basi",
        "Unverified Poll Claim": "ignore the fake sponsored polls by tifa, we are winning this by over 63 percent of total votes",
        "Civic Peace Message": "we have to say no to violence, let your neighbors peace be your responsibility during elections",
    }

    if "claim_text" not in st.session_state:
        st.session_state.claim_text = ""

    def load_preset():
        picked = st.session_state.preset_pills
        if picked in SAMPLES:
            st.session_state.claim_text = SAMPLES[picked]

    def reset_claim():
        st.session_state.claim_text = ""
        st.session_state.preset_pills = None

    st.caption("Quick-load a sample post or paste your own below:")
    st.pills(
        "Sample Presets",
        options=list(SAMPLES.keys()),
        key="preset_pills",
        on_change=load_preset,
        label_visibility="collapsed",
    )

    user_text = st.text_area(
        "Post or Claim Text",
        key="claim_text",
        height=130,
        placeholder="Paste a tweet, WhatsApp forward, or headline in Kiswahili, Sheng, or English...",
        label_visibility="collapsed",
    )

    ctrl1, ctrl2, ctrl3 = st.columns([2.2, 1, 2.3], vertical_alignment="center")
    with ctrl1:
        verify_btn = st.button("Run Fact-Check", type="primary", use_container_width=True)
    with ctrl2:
        st.button("Reset", use_container_width=True, on_click=reset_claim)
    with ctrl3:
        show_explain = st.toggle("Show trigger-word highlights", value=True)

    if verify_btn:
        if not user_text.strip():
            st.warning("Enter or select a post above before running verification.")
        else:
            with st.spinner("Evaluating linguistic cues..."):
                # Works even if predict() doesn't have 'explain' yet
                try:
                    res = predictor.predict(user_text, explain=show_explain)
                except TypeError:
                    res = predictor.predict(user_text)

            pred = res["prediction"].upper()
            conf = res["confidence"]
            probs = res["probabilities"]

            # Verdict Banner
            if pred == "FAKE":
                css_cls = "verdict-fake"
                headline = "LIKELY MISINFORMATION / FAKE"
                sub_note = "Contains linguistic patterns strongly associated with fabricated or misleading claims."
            elif pred == "FACT":
                css_cls = "verdict-fact"
                headline = "VERIFIABLE / FACTUAL FRAMING"
                sub_note = "Aligns with factual reporting or verifiable civic statements."
            else:
                css_cls = "verdict-neutral"
                headline = "OPINION / NEUTRAL COMMENTARY"
                sub_note = "Reads as personal commentary, humor, or general discussion rather than a factual claim."

            st.markdown(
                f"""
                <div class="verdict-banner {css_cls}">
                    <div class="verdict-label">Classification Verdict &bull; {conf}% Confidence</div>
                    <div class="verdict-main">{headline}</div>
                    <div style="font-size: 0.9rem; opacity: 0.85;">{sub_note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Clean 3-Column Score Breakdown
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Fake / Misinfo", f"{probs.get('fake', 0):.1f}%")
            sc2.metric("Fact / Verifiable", f"{probs.get('fact', 0):.1f}%")
            sc3.metric("Neutral / Opinion", f"{probs.get('neutral', 0):.1f}%")

            # Word-Level Trigger Highlights
            if show_explain and res.get("word_impacts"):
                st.markdown("**Linguistic Trigger Map** *(hover over highlighted words for impact)*")
                rgb = "239, 68, 68" if pred == "FAKE" else "16, 185, 129" if pred == "FACT" else "245, 158, 11"
                tokens_html = []
                for word, impact in res["word_impacts"]:
                    safe_w = html.escape(word)
                    if impact > 1.0:
                        alpha = min(0.18 + (impact / 35.0), 0.65)
                        tokens_html.append(
                            f'<span style="background-color: rgba({rgb}, {alpha:.2f}); '
                            f'padding: 2px 6px; border-radius: 4px; font-weight: 700;" '
                            f'title="Confidence contribution: +{impact}%">{safe_w}</span>'
                        )
                    else:
                        tokens_html.append(safe_w)
                st.markdown(f'<div class="token-box">{" ".join(tokens_html)}</div>', unsafe_allow_html=True)

# ==========================================
# TAB 2: BULK CSV AUDITING
# ==========================================
with tab_audit:
    st.markdown("#### Dataset & Social Media Feed Auditing")
    st.write("Upload a CSV export of posts to classify all rows and export an audited report.")

    uploaded_file = st.file_uploader("Drop CSV file here", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")
            st.stop()

        if df.empty:
            st.warning("Uploaded CSV has no rows.")
        else:
            c_col, c_run = st.columns([2.5, 1], vertical_alignment="bottom")
            with c_col:
                text_col = st.selectbox("Text column to audit:", df.columns)
            with c_run:
                audit_clicked = st.button("Audit Dataset", type="primary", use_container_width=True)

            if audit_clicked:
                bar = st.progress(0)
                texts = df[text_col].fillna("").astype(str).tolist()

                if hasattr(predictor, "predict_batch"):
                    batch_out = predictor.predict_batch(texts, batch_size=32, progress_callback=bar.progress)
                    preds = [r["prediction"].upper() for r in batch_out]
                    confs = [r["confidence"] for r in batch_out]
                else:
                    preds, confs = [], []
                    for i, t in enumerate(texts):
                        if not t.strip():
                            preds.append("NEUTRAL")
                            confs.append(0.0)
                        else:
                            r = predictor.predict(t)
                            preds.append(r["prediction"].upper())
                            confs.append(r["confidence"])
                        bar.progress((i + 1) / len(texts))

                audited = df.copy()
                audited["Verdict"] = preds
                audited["Confidence (%)"] = confs
                st.session_state.audited_df = audited

            if "audited_df" in st.session_state:
                audited = st.session_state.audited_df
                counts = audited["Verdict"].value_counts()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Rows Audited", len(audited))
                m2.metric("Fake", int(counts.get("FAKE", 0)))
                m3.metric("Fact", int(counts.get("FACT", 0)))
                m4.metric("Neutral", int(counts.get("NEUTRAL", 0)))

                chosen = st.pills(
                    "Filter by verdict:",
                    options=["FAKE", "FACT", "NEUTRAL"],
                    selection_mode="multi",
                    default=["FAKE", "FACT", "NEUTRAL"],
                )
                view_df = audited[audited["Verdict"].isin(chosen)] if chosen else audited
                st.dataframe(view_df, use_container_width=True)

                st.download_button(
                    label="Export Audited CSV",
                    data=view_df.to_csv(index=False).encode("utf-8"),
                    file_name="ukwelicheck_audit_report.csv",
                    mime="text/csv",
                    type="primary",
                )

# ==========================================
# TAB 3: MODEL & BENCHMARKS
# ==========================================
with tab_specs:
    st.markdown("#### Architecture & Training Pipeline")
    st.markdown(
        """
- **Backbone:** `Davlan/afro-xlmr-base` (Cross-lingual RoBERTa pre-trained across 17 African languages + English)
- **Corpus:** `lusamaki/Fake_News_Detection_System_29.5k` (Leakage-filtered & conflicting duplicates removed)
- **Optimization:** Inverse-frequency class-weighted Cross-Entropy loss with early stopping on Macro F1
        """
    )

    metrics_file = BASE_DIR / "results" / "metrics.json"
    if metrics_file.exists():
        st.divider()
        st.markdown("#### Holdout Test Set Performance")
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        b1, b2, b3 = st.columns(3)
        b1.metric("Test Accuracy", f"{metrics['test_accuracy']*100:.2f}%")
        b2.metric("Macro F1-Score", f"{metrics['test_macro_f1']*100:.2f}%")
        b3.metric("Weighted F1-Score", f"{metrics['test_weighted_f1']*100:.2f}%")

# --- FOOTER ---
st.markdown(
    """
    <div class="editorial-footer">
        <span>UkweliCheck NLP Lab</span>
        <span>Built for Kenyan Code-Switched Discourse (Kiswahili &bull; Sheng &bull; English)</span>
    </div>
    """,
    unsafe_allow_html=True,
)