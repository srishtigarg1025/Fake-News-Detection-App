import streamlit as st
import pickle

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
    else:
        status = "✅ Real" if pred == 1 else "❌ Fake"

    return status, round(confidence * 100, 2)

# STREAMLIT UI
st.title("🧠 AI-Based Fake News Detection System")

st.write("This application uses NLP + Machine Learning to classify news as Real or Fake.")

user_input = st.text_area("Enter News Text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        status, confidence = predict_news(user_input)

        st.subheader("Result")
        st.write("Prediction:", status)
        st.write("Confidence:", confidence, "%")