# 📰 News Authenticity Classifier

A web app that detects whether a news article is **Fake** or **Genuine** using Machine Learning.  
Built with **Python, Flask, scikit-learn, and ngrok** for deployment.

---

## ✨ Features
- Paste a news article and instantly classify it as **Fake** 🟥 or **Genuine** 🟩  
- Simple, clean web interface with a background image  
- Uses multiple ML models (Logistic Regression, Random Forest, Gradient Boosting)  
- Deployable via **ngrok** for live demos  

---

## 🚀 Tech Stack
- **Backend:** Python (Flask)  
- **Frontend:** HTML, CSS  
- **ML:** scikit-learn  
- **Deployment:** ngrok + Colab  

---

## 📸 Screenshot
<img src="static/fake.png" width="600">

---

## ⚙️ Getting Started

**Cloning the Repository**
```bash
git clone https://github.com/txm19/NewsAuthenticityClassifier.git
cd NewsAuthenticityClassifier
Installation

bash
Copy
Edit
pip install -r requirements.txt
Run the App

bash
Copy
Edit
python app.py
The app will be available at:
👉 http://127.0.0.1:5000

Live Demo (with ngrok)

bash
Copy
Edit
ngrok http 5000
You’ll get a shareable link like:
👉 https://<random-id>.ngrok-free.app

📂 Project Structure
bash
Copy
Edit
NewsAuthenticityClassifier/
│
├── app.py                 # Main Flask app
├── requirements.txt       # Python dependencies
│
├── static/                # Static files (images, CSS, etc.)
│   └── fake.png           # Background image
│
├── templates/             # HTML templates
│   └── index.html         # Web UI
│
├── True.csv               # Dataset (True news)
├── Fake.csv               # Dataset (Fake news)
│
├── gbc_model.pkl          # Trained Gradient Boosting model
├── logistic_model.pkl     # Trained Logistic Regression model
├── rfc_model.pkl          # Trained Random Forest model
├── vectorizer.pkl         # TF-IDF vectorizer
│
└── README.md              # Project documentation
