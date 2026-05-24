import streamlit as st
import pickle

st.set_page_config(page_title="Fake News Detector", page_icon="🧠")

st.sidebar.title("📌 About")
st.sidebar.info(
    "This AI model detects whether a news article is Real or Fake using Machine Learning (TF-IDF + Logistic Regression)."
)


# LOAD SAVED FILES
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# PREDICTION FUNCTION
def predict_news(news):
    vec = vectorizer.transform([news])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    confidence = prob[pred]

    if confidence < 0.6:
       status = "⚠️ Uncertain"
    elif confidence < 0.75:
       status = "🟡 Low Confidence"
    else:
       status = "✅ Real" if pred == 1 else "❌ Fake"

    return status, round(confidence * 100, 2)

# STREAMLIT UI
st.title("🧠 AI-Based Fake News Detection System")

st.write("Detect whether a news article is **Real or Fake** using AI.")

# Sample buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("🧪 Try Fake Example"):
        st.session_state.input_text = "A man claims he built a time machine using household items and successfully traveled to the future."

with col2:
    if st.button("🧪 Try Real Example"):
        st.session_state.input_text = "The government said on Tuesday that it will introduce new economic reforms to boost growth, according to official sources."

# Input box 
st.info("💡 Tip: Model works best with detailed news articles (2–3 lines with sources).")
user_input = st.text_area(
    "Enter News Text:",
    value=st.session_state.get("input_text", "")
)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        status, confidence = predict_news(user_input)

        st.subheader("Result")
        if "Real" in status:
         st.success(f"{status} ({confidence}%)")
        elif "Fake" in status:
         st.error(f"{status} ({confidence}%)")
        else:
         st.warning(f"{status} ({confidence}%)")

st.markdown("---")
st.caption("⚠️ This tool is for educational purposes only.")
