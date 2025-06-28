# 🌐 Milestone 6: Flask Integration for Liver Cirrhosis Prediction

This milestone focuses on deploying the trained model as a web application using the Flask framework.

---

## 🏗️ Folder Structure for Deployment

```
📦 Project Root
├── 📂 templates
│   └── index.html                # Front-end UI for user input & result display
├── 📂 static
│   └── style.css                 # Optional CSS styling for the UI
├── app.py                        # Flask application backend
├── rf_model.pkl                  # Trained machine learning model
├── normalizer.pkl                # Saved normalizer/scaler used for input
└── requirements.txt              # List of required Python libraries
```

---

## 💻 Step-by-Step Flask Integration

### 1. Create `app.py` – the backend Flask server

```python
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and normalizer
model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("normalizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    final_input = scaler.transform([features])
    prediction = model.predict(final_input)
    output = "Liver Cirrhosis Detected" if prediction[0] == 1 else "No Cirrhosis Detected"
    return render_template("index.html", prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)
```

### 2. Create `index.html` – the front-end form

Place this in the `templates/` folder.

```html
<!DOCTYPE html>
<html>
<head>
    <title>Liver Cirrhosis Prediction</title>
</head>
<body>
    <h2>Enter Patient Details</h2>
    <form action="/predict" method="post">
        <input type="text" name="Total_Bilirubin" placeholder="Total Bilirubin" required><br>
        <input type="text" name="Direct_Bilirubin" placeholder="Direct Bilirubin" required><br>
        <input type="text" name="ALT" placeholder="ALT (SGPT)" required><br>
        <input type="text" name="AST" placeholder="AST (SGOT)" required><br>
        <input type="text" name="Albumin" placeholder="Albumin" required><br>
        <input type="text" name="A/G_Ratio" placeholder="A/G Ratio" required><br>
        <button type="submit">Predict</button>
    </form>
    {% if prediction_text %}
        <h3>{{ prediction_text }}</h3>
    {% endif %}
</body>
</html>
```

---

## 📦 3. Create `requirements.txt`

```txt
flask
numpy
scikit-learn
pandas
xgboost
```
Install using `pip install -r requirements.txt`

---

## 🚀 Running the App

```bash
python app.py
```
Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## ✅ Summary

- Model and scaler are integrated into a Flask app.
- Simple HTML form accepts inputs and shows prediction.
- Ready for deployment and demo.

🎯 You're now ready for project recording, testing, and submission!