from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "saved_model"
FALLBACK_MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"


@st.cache_resource
def load_model_and_tokenizer(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def predict_sentiment(text: str, tokenizer, model):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]

    predicted_label = int(torch.argmax(probabilities).item())
    sentiment = "Positive" if predicted_label == 1 else "Negative"
    confidence = float(probabilities[predicted_label].item())
    return sentiment, confidence


def main():
    st.set_page_config(page_title="Sentiment Analysis with BERT", page_icon="💬")
    st.title("💬 Sentiment Analysis with BERT")

    st.markdown("Enter a movie review and get a sentiment prediction from your trained model.")

    model_dir = st.text_input("Model directory", value=str(DEFAULT_MODEL_DIR))
    review_text = st.text_area("Review text", height=180, placeholder="Type a movie review here...")

    if st.button("Predict sentiment"):
        if not review_text.strip():
            st.warning("Please enter some review text.")
            return

        model_source = model_dir
        if not Path(model_dir).exists():
            model_source = FALLBACK_MODEL_ID
            st.warning(
                f"Model directory not found: {model_dir}. Using fallback model: {FALLBACK_MODEL_ID}"
            )

        tokenizer, model = load_model_and_tokenizer(model_source)
        sentiment, confidence = predict_sentiment(review_text, tokenizer, model)

        st.success(f"Prediction: {sentiment}")
        st.write(f"Confidence: {confidence:.2%}")

if __name__ == "__main__":
    main()