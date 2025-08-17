from flask import Flask, request, render_template
import pickle
import pandas as pd
import re

# Load your saved models (make sure you've pickled them earlier in Colab)
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
LR = pickle.load(open("logistic_model.pkl", "rb"))
gbc = pickle.load(open("gbc_model.pkl", "rb"))
rfc = pickle.load(open("rfc_model.pkl", "rb"))

app = Flask(__name__)

def wordopt(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\n', ' ', text)
    return text

def output_label(n):
    return "✅ Genuine News" if n == 1 else "❌ Fake News"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]
    df = pd.DataFrame({"text":[news]})
    df["text"] = df["text"].apply(wordopt)
    new_xv_test = vectorizer.transform(df["text"])

    pred_lr = LR.predict(new_xv_test)[0]
    pred_gbc = gbc.predict(new_xv_test)[0]
    pred_rfc = rfc.predict(new_xv_test)[0]

    return render_template("index.html", news=news,
        result=f"LR: {output_label(pred_lr)} | GBC: {output_label(pred_gbc)} | RFC: {output_label(pred_rfc)}")

if __name__ == "__main__":
    from pyngrok import ngrok
    public_url = ngrok.connect(5000)
    print("Public URL:", public_url)
    app.run(port=5000)

