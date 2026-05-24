
🧠 AI-Based Fake News Detection System

A Machine Learning + NLP powered web application that classifies news articles as **Real** or **Fake** with a confidence score.

🔗 Live App:  https://fake-news-detection-app-duo3p6gmursmat3g3k24gx.streamlit.app/

```
Features
```

- Classifies news as **Real / Fake / Uncertain**
- Displays **confidence score**
- Built-in **test example buttons**
- Fast predictions using trained ML model
- Deployed on Streamlit Cloud
- Clean and interactive UI

```
Tech Stack
```

- **Frontend:** Streamlit
- **Backend:** Python
- **ML Model:** Logistic Regression 
- **NLP:** TF-IDF Vectorization
- **Libraries:**

  * scikit-learn
  * pandas
  * numpy
  * pickle

```
Project Structure
```
Fake-News-App/


├── train_model.py      # Train model (run once)

├── app.py              # Streamlit web app

├── model.pkl           # Saved ML model

├── vectorizer.pkl      # Saved TF-IDF vectorizer

├── requirements.txt    # Dependencies

└── README.md
```

Limitations

* Works best with **formal news-style text (Reuters-like)**
* May give low confidence on:

  * Informal language
  * Very short text
  * Regional news (e.g., local Indian news)

Future Enhancements

* 🔍 Explainable AI (why prediction?)
* 🌍 Better support for Indian/local news
* 🧠 Upgrade to BERT / Deep Learning
* 📊 Confidence visualization (progress bar)
* 🌐 Multi-language support


