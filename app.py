
import os
import gradio as gr
import numpy as np
import scipy.sparse as sp
import pickle
import re
import json
import tensorflow as tf
from tensorflow.keras.models    import Sequential
from tensorflow.keras.layers    import (Embedding, Bidirectional, LSTM,
                                         Dense, Dropout, SpatialDropout1D)
from tensorflow.keras.optimizers import Adam
import nltk
import spacy
import warnings
warnings.filterwarnings("ignore")

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE       = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE, "models")

import spacy
import subprocess
import sys

# Download the English model if not already present
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

def load(filename):
    return os.path.join(MODELS_DIR, filename)

with open(load("svm_model.pkl"),        "rb") as f: svm_model  = pickle.load(f)
with open(load("tfidf.pkl"),            "rb") as f: tfidf      = pickle.load(f)
with open(load("scaler.pkl"),           "rb") as f: scaler     = pickle.load(f)
with open(load("tokenizer.pkl"),        "rb") as f: tokenizer  = pickle.load(f)
with open(load("final_comparison.json"))      as f: comparison = json.load(f)

MAX_VOCAB    = 13000
vocab_size   = min(MAX_VOCAB, len(tokenizer.word_index) + 1)
MAX_LEN_LSTM = 101

def build_bilstm(vocab_size, max_len):
    model = Sequential([
        Embedding(vocab_size, 128,
                  input_length=max_len,   name="embedding"),
        SpatialDropout1D(0.3,             name="spatial_dropout"),
        Bidirectional(
            LSTM(64,
                 dropout=0.3,
                 recurrent_dropout=0.3,
                 return_sequences=False), name="bilstm"),
        Dense(32, activation="relu",      name="dense_hidden"),
        Dropout(0.6,                      name="dense_dropout"),
        Dense(1,  activation="sigmoid",   name="output")
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

lstm_model = build_bilstm(vocab_size, MAX_LEN_LSTM)
lstm_model.build(input_shape=(None, MAX_LEN_LSTM))
print("All models loaded.")

nlp   = spacy.load("en_core_web_sm", disable=["parser", "ner"])
STOP  = set(stopwords.words("english")) - {
            "not","no","never","dont","doesnt","didnt",
            "wont","cant","isnt","wasnt","neither","nor"}
NEG_P = (r"\b(not|never|no|dont|doesnt|didnt|"
         r"wont|cant|isnt|wasnt|neither|nor)\b")

def preprocess_svm(text):
    t = str(text).lower()
    t = re.sub(r"<.*?>",      "", t)
    t = re.sub(r"http\S+",    "", t)
    t = re.sub(r"'",          "", t)
    t = re.sub(NEG_P + r"\s+(\w+)",
               lambda m: m.group(1)+"_"+m.group(2), t)
    t = re.sub(r"[^a-z\s_]", "", t)
    t = " ".join(w for w in t.split() if w not in STOP)
    t = " ".join(tok.lemma_ for tok in nlp(t))
    return t

def preprocess_lstm(text):
    t = str(text).lower()
    t = re.sub(r"<.*?>",   "", t)
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"\s+",     " ", t).strip()
    return t

def predict_svm(text):
    clean = preprocess_svm(text)
    vec   = tfidf.transform([clean])
    eng   = scaler.transform([[
        text.count("!"),
        sum(1 for c in text if c.isupper()) / (len(text)+1),
        len(text.split()), 0
    ]])
    x    = sp.hstack([vec, sp.csr_matrix(eng)])
    pred = svm_model.predict(x)[0]
    prob = svm_model.predict_proba(x)[0]
    return int(pred), float(prob[1]), float(prob[0])

def predict_lstm(text):
    clean  = preprocess_lstm(text)
    seq    = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=MAX_LEN_LSTM,
                           padding="post", truncating="post")
    p      = float(lstm_model.predict(padded, verbose=0)[0][0])
    return (1 if p >= 0.5 else 0), p, 1-p

def make_bar(prob, width=20):
    filled = int(round(prob * width))
    return f"[{'█'*filled}{'░'*(width-filled)}] {prob*100:.1f}%"

def build_stats(text):
    return (f"📊 {len(text.split())} words  |  "
            f"{len(text)} chars  |  "
            f"{text.count('!')} exclamation marks  |  "
            f"{sum(1 for c in text if c.isupper())} capitals")

def build_interpretation(text, pred, prob_pos,
                         lstm_pred=None, lstm_pos=None,
                         both=False, model_name=""):
    lines = []
    neg_words = ["not","never","no","dont","doesnt",
                 "didnt","wont","cant","isnt","wasnt"]
    found_neg = [w for w in neg_words if w in text.lower()]
    if both:
        svm_l  = "positive" if pred      == 1 else "negative"
        lstm_l = "positive" if lstm_pred == 1 else "negative"
        if pred == lstm_pred:
            lines.append(f"Both models classified this as **{svm_l}**.")
        else:
            lines.append(
                f"The SVM predicted **{svm_l}** while the BiLSTM "
                f"predicted **{lstm_l}**. This typically occurs on "
                f"mixed-sentiment reviews.")
    else:
        label     = "positive" if pred == 1 else "negative"
        conf      = max(prob_pos, 1-prob_pos)*100
        certainty = ("highly confident" if conf >= 85
                     else "moderately confident" if conf >= 70
                     else "uncertain — mixed sentiment likely")
        lines.append(f"The {model_name} classified this as **{label}** "
                     f"and is {certainty} ({conf:.1f}%).")
    if found_neg:
        lines.append(
            f"⚡ Negation detected ({', '.join(found_neg[:3])}).")
    if len(text.split()) < 10:
        lines.append(
            "ℹ️ Short review — confidence may be lower than usual.")
    return "  \n".join(lines)

def analyse_sentiment(review_text, model_choice):
    if not review_text or len(review_text.strip()) < 5:
        return ("⚠️ Please enter a review with at least 5 characters.",
                "", "", "")
    text = review_text.strip()
    if model_choice == "Both models":
        sp_, sp_pos, sp_neg = predict_svm(text)
        lp_, lp_pos, lp_neg = predict_lstm(text)
        s_label = "POSITIVE 😊" if sp_ == 1 else "NEGATIVE 😞"
        l_label = "POSITIVE 😊" if lp_ == 1 else "NEGATIVE 😞"
        agree   = ("✅ Both models agree" if sp_ == lp_
                   else "⚡ Models disagree — mixed sentiment likely")
        result  = (f"**SVM:**    {s_label}  ({max(sp_pos,sp_neg)*100:.1f}%)\n"
                   f"**BiLSTM:** {l_label}  ({max(lp_pos,lp_neg)*100:.1f}%)\n\n"
                   f"{agree}")
        bars    = (f"SVM  → P(positive): {make_bar(sp_pos)}\n"
                   f"LSTM → P(positive): {make_bar(lp_pos)}")
        interp  = build_interpretation(text, sp_, sp_pos,
                                       lp_, lp_pos, both=True)
    elif model_choice == "SVM (LinearSVC + TF-IDF)":
        pred, p_pos, p_neg = predict_svm(text)
        label  = "POSITIVE 😊" if pred == 1 else "NEGATIVE 😞"
        result = f"**SVM:** {label}  ({max(p_pos,p_neg)*100:.1f}%)"
        bars   = (f"P(positive): {make_bar(p_pos)}\n"
                  f"P(negative): {make_bar(p_neg)}")
        interp = build_interpretation(
            text, pred, p_pos, model_name="SVM")
    else:
        pred, p_pos, p_neg = predict_lstm(text)
        label  = "POSITIVE 😊" if pred == 1 else "NEGATIVE 😞"
        result = f"**BiLSTM:** {label}  ({max(p_pos,p_neg)*100:.1f}%)"
        bars   = (f"P(positive): {make_bar(p_pos)}\n"
                  f"P(negative): {make_bar(p_neg)}")
        interp = build_interpretation(
            text, pred, p_pos, model_name="BiLSTM")
    return result, bars, interp, build_stats(text)

svm_r  = comparison["svm"]
lstm_r = comparison["lstm"]

EXAMPLES = [
    ["This product is absolutely amazing! Works perfectly.", "Both models"],
    ["Complete waste of money. Broke after two days.",       "Both models"],
    ["Great camera but the battery life is disappointing.",  "Both models"],
    ["Not what I expected. Does not work as described.",     "Both models"],
    ["Decent product for the price. Does the job.",          "Both models"],
]

with gr.Blocks(
    title="Sentiment Analysis — Group 15",
    theme=gr.themes.Base(primary_hue="red", neutral_hue="slate"),
    css="footer { display: none !important; }"
) as demo:
    gr.HTML(f"""
        <div style="text-align:center; padding:1.2rem 0 0.5rem">
            <h1 style="color:#C0392B">
                🔍 Product Review Sentiment Analyser
            </h1>
            <p style="color:#888">
                Group 15 · TechCrunch Cohort 6 · SVM vs BiLSTM
            </p>
            <hr style="border-color:#C0392B;opacity:0.3;margin:0.8rem 0 0.4rem">
        </div>
    """)
    with gr.Row():
        with gr.Column(scale=3):
            review_input = gr.Textbox(
                label="Enter a product review",
                placeholder=(
                    "e.g. This product is amazing! "
                    "or Broke after two days."),
                lines=5, max_lines=10)
            model_choice = gr.Radio(
                choices=["SVM (LinearSVC + TF-IDF)",
                         "Bidirectional LSTM",
                         "Both models"],
                value="Both models",
                label="Select model")
            with gr.Row():
                submit_btn = gr.Button("Analyse Sentiment",
                                       variant="primary")
                clear_btn  = gr.Button("Clear",
                                       variant="secondary")
        with gr.Column(scale=2):
            gr.HTML(f"""
                <div style="background:#1a0a0a;border:1px solid #C0392B;
                            border-radius:8px;padding:1rem;
                            font-size:0.88rem;color:#ccc">
                    <b style="color:#E57373">Model Guide</b><br><br>
                    <b style="color:#fff">SVM</b> — Accuracy
                    {svm_r['accuracy']*100:.1f}% ·
                    F1 {svm_r['weighted_f1']:.4f}<br><br>
                    <b style="color:#fff">BiLSTM</b> — Neg recall
                    {lstm_r['neg_recall']*100:.1f}%
                    (better for complaints)<br><br>
                    <b style="color:#fff">Both</b> —
                    Compare both models
                </div>
            """)
    gr.HTML("<hr style='border-color:#C0392B;opacity:0.2'>")
    result_label = gr.Markdown(label="Prediction")
    prob_bar     = gr.Markdown(label="Probability")
    interp_text  = gr.Markdown(label="Interpretation")
    stats_text   = gr.Markdown(label="Review Stats")
    gr.HTML("<hr style='border-color:#C0392B;opacity:0.2'>")
    gr.Examples(examples=EXAMPLES,
                inputs=[review_input, model_choice],
                label="Try these examples")

    # ── Model performance summary accordion
    with gr.Accordion("📊 Model Performance Summary", open=False):
        svm_r  = comparison['svm']
        lstm_r = comparison['lstm']
        gr.HTML(f"""
            <table style="width:100%; border-collapse:collapse;
                          font-size:0.9rem; color:#ddd">
                <thead>
                    <tr style="background:#6D0000; color:white">
                        <th style="padding:8px; text-align:left">Metric</th>
                        <th style="padding:8px; text-align:center">
                            SVM</th>
                        <th style="padding:8px; text-align:center">
                            BiLSTM</th>
                        <th style="padding:8px; text-align:center">
                            Winner</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(f"""
                    <tr style=\"background:{'#1a0a0a' if i%2==0 else '#110505'}\">
                        <td style="padding:7px">{m}</td>
                        <td style="padding:7px; text-align:center">{sv}</td>
                        <td style="padding:7px; text-align:center">{lv}</td>
                        <td style="padding:7px; text-align:center">{w}</td>
                    </tr>"""
                    for i,(m,sv,lv,w) in enumerate([
                        ('Accuracy',
                         f"{svm_r['accuracy']:.4f}",
                         f"{lstm_r['accuracy']:.4f}", '🔴 SVM'),
                        ('Weighted F1',
                         f"{svm_r['weighted_f1']:.4f}",
                         f"{lstm_r['weighted_f1']:.4f}", '🔴 SVM'),
                        ('ROC-AUC',
                         f"{svm_r['roc_auc']:.4f}",
                         f"{lstm_r['roc_auc']:.4f}", '🔴 SVM'),
                        ('Negative Recall',
                         f"{svm_r['tn']/(svm_r['tn']+svm_r['fp']):.4f}",
                         f"{lstm_r['neg_recall']:.4f}", '🟢 BiLSTM'),
                        ('F1 — Negative',
                         f"{svm_r['f1_negative']:.4f}",
                         f"{lstm_r['f1_negative']:.4f}", '🔴 SVM'),
                    ])}
                </tbody>
            </table>
            <p style="color:#888; font-size:0.82rem; margin-top:8px">
                Key finding: SVM wins on aggregate metrics.
                BiLSTM wins on negative recall (+11.02pp) —
                the metric that matters most for complaint detection.
            </p>
        """)

    # ── Wire up buttons
    submit_btn.click(
        fn=analyse_sentiment,
        inputs=[review_input, model_choice],
        outputs=[result_label, prob_bar, interp_text, stats_text]
    )
    clear_btn.click(
        fn=lambda: ("", "Both models", "", "", "", ""),
        outputs=[review_input, model_choice,
                 result_label, prob_bar, interp_text, stats_text]
    )
    review_input.submit(
        fn=analyse_sentiment,
        inputs=[review_input, model_choice],
        outputs=[result_label, prob_bar, interp_text, stats_text]
    )

print("Interface built successfully.")


# Launch outside the with block
if __name__ == "__main__":
    demo.launch(
        share=True,
        debug=False,
        show_error=True
    )
