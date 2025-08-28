import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

# Add the CSS code right here
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #6a11cb, #2575fc);
}
</style>
""", unsafe_allow_html=True)

# --- Load the trained model and tokenizer ---
try:
    model = tf.keras.models.load_model("toxicity_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
except (IOError, OSError) as e:
    st.error(f"Error loading model or tokenizer: {e}. Make sure you run model_trainer.py first.")
    st.stop()

max_len = 100

# --- Text cleaning function (must match the one used for training) ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Streamlit UI ---
st.title("🛡️ Comment Toxicity Detector")
st.subheader("Deep Learning for Content Moderation")

st.write("Enter a comment below to check if it is toxic.")

user_input = st.text_area("Enter a comment:", height=150)

if st.button("Check Toxicity"):
    if user_input:
        # Preprocess the user input
        cleaned_text = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=max_len)

        # Make prediction
        prediction = model.predict(padded)[0][0]

        # Display result
        st.write("---")
        st.write(f"Prediction Score: `{prediction:.4f}`")

        if prediction > 0.5:
            st.error("⚠️ This comment is likely **toxic**.")
            st.markdown("This comment has a high probability of containing toxic language.")
        else:
            st.success("✅ This comment is likely **safe**.")
            st.markdown("This comment appears to be safe and non-toxic.")
    else:
        st.warning("Please enter some text to analyze.")