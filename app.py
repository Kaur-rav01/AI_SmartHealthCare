from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import csv
from datetime import datetime
import os
from groq import Groq
import numpy as np
import pickle
from PIL import Image
from werkzeug.utils import secure_filename
import tensorflow as tf
from functools import wraps

# ------------------ APP CONFIG ------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.secret_key = "ravjot234"

# ------------------ LOAD MODELS ------------------
chest_model = tf.keras.models.load_model(
    r"C:\Users\DELL\OneDrive\Desktop\internship_\final_ai_smarthealthcare\models\best_model.h5"
)

scaler = pickle.load(open(
    r"C:\Users\DELL\OneDrive\Desktop\internship_\final_ai_smarthealthcare\models\heart_scaler.pkl", "rb"
))
heart_model = pickle.load(open(
    r"C:\Users\DELL\OneDrive\Desktop\internship_\final_ai_smarthealthcare\models\heart_model.pkl", "rb"
))
diabetes_model = pickle.load(open(
    r"C:\Users\DELL\OneDrive\Desktop\internship_\final_ai_smarthealthcare\models\diabetes_model.pkl", "rb"
))

# ------------------ API CLIENT ------------------

import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ------------------ FILES ------------------
CSV_FILE = "reminders.csv"
REPORTS_FILE = "reports.csv"

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["user", "task", "datetime"])
        writer.writeheader()

if not os.path.exists(REPORTS_FILE):
    with open(REPORTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["user", "report_type", "prediction", "recommendation", "datetime"])
        writer.writeheader()

# ------------------ HELPERS ------------------
def check_patient():
    return "patient_name" in session and "patient_age" in session

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not check_patient():
            flash("Please enter your name and age first!", "danger")
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return wrapper

# ------------------ ROUTES ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    if request.method == "POST":
        name = request.form.get("name")
        age = request.form.get("age")
        if not name or not age:
            error = "Please enter your Name and Age"
        else:
            session["patient_name"] = name
            session["patient_age"] = age
            return redirect(url_for("index"))
    return render_template("index.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

# ------------------ CHEST X-RAY ------------------
@app.route("/chest", methods=["GET", "POST"])
@login_required
def chest():
    prediction, image_path, recommendation = None, None, None
    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename.strip():
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            img = Image.open(image_path).resize((224, 224)).convert("RGB")
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

            result = chest_model.predict(img_array)[0][0]
            if result > 0.5:
                prediction = "Higher Chances of Pneumonia"
                recommendation = """
                <h5><strong>✅ Recommended Actions:</strong></h5>
                <ul><li>Consult a doctor</li>
                <li>Rest at home</li>
                <li>Avoid dust and cold drinks</li></ul>
                """
            else:
                prediction = "No Pneumonia Detected"
                recommendation = """
                <h5><strong>✅ Healthy Tips:</strong></h5>
                <ul><li>Practice Deep Breathing Exercise</li>
                <li>Avoid Smoking or Secondhand Smoke</li>
                <li>Rest & Recover Properly from Colds/Flu</li></ul>
                """

            # Save report
            with open(REPORTS_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["user", "report_type", "prediction", "recommendation", "datetime"])
                writer.writerow({
                    "user": session["patient_name"],
                    "report_type": "Chest",
                    "prediction": prediction,
                    "recommendation": recommendation,
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M")
                })

    return render_template("chest.html", prediction=prediction, image_path=image_path, recommendation=recommendation)

# ------------------ HEART ------------------
@app.route("/heart", methods=["GET", "POST"])
@login_required
def heart():
    prediction, recommendation = None, None
    if request.method == "POST":
        features = np.array([[ 
            int(request.form["age"]),
            1 if request.form["sex"] == "Male" else 0,
            int(request.form["cp"]),
            int(request.form["trestbps"]),
            int(request.form["chol"]),
            int(request.form["fbs"]),
            int(request.form["restecg"]),
            int(request.form["thalach"]),
            int(request.form["exang"]),
            float(request.form["oldpeak"]),
            int(request.form["slope"]),
            int(request.form["ca"]),
            int(request.form["thal"]),
        ]])

        scaled_features = scaler.transform(features)
        result = heart_model.predict(scaled_features)[0]

        if result == 1:
            prediction = "Higher Chances of Heart Disease"
            recommendation = """
            <h5><strong>✅ Recommended Actions:</strong></h5>
            <ul><li>Consult a cardiologist</li>
            <li>Control cholesterol</li>
            <li>Follow heart-healthy diet</li></ul>
            """
        else:
            prediction = "Low Chances of Heart Disease"
            recommendation = """
            <h5><strong>✅ Recommended Actions:</strong></h5>
            <ul><li>Keep up the good lifestyle</li>
            <li>Exercise regularly</li>
            <li>Monitor BP & sugar levels</li></ul>
            """

        # Save report
        with open(REPORTS_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["user", "report_type", "prediction", "recommendation", "datetime"])
            writer.writerow({
                "user": session["patient_name"],
                "report_type": "Heart",
                "prediction": prediction,
                "recommendation": recommendation,
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M")
            })

    return render_template("heart.html", prediction=prediction, recommendation=recommendation)

# ------------------ DIABETES ------------------
@app.route("/diabetes", methods=["GET", "POST"])
@login_required
def diabetes():
    prediction, recommendation = None, None
    if request.method == "POST":
        age = int(request.form["age"])
        gender = 1 if request.form["gender"] == "Male" else 0

        yes_no_fields = [
            'urination','thirst','weight_loss','hunger','fungal_infection','hair_loss',
            'irritability','slow_healing','muscle_weakness','obesity',
            'stiff_muscles','vision','itching','weakness'
        ]

        inputs = [age, gender]
        for field in yes_no_fields:
            inputs.append(1 if request.form[field] == "Yes" else 0)

        result = diabetes_model.predict([inputs])[0]

        if result == 1:
            prediction = "High Risk of Diabetes"
            recommendation = """
            <h5><strong>✅ Recommended Actions:</strong></h5>
            <ul><li>Maintain a balanced low-sugar diet</li>
            <li>Exercise regularly (30 mins/day)</li>
            <li>Monitor blood sugar regularly</li>
            <li>Take medications as prescribed</li>
            <li>Get regular checkups</li></ul>
            """
        else:
            prediction = "Low Risk of Diabetes"
            recommendation = """
            <h5><strong>✅ Recommended Actions:</strong></h5>
            <ul><li>Exercise regularly</li>
            <li>Maintain balanced diet</li>
            <li>Monitor blood sugar yearly</li></ul>
            """

        # Save report
        with open(REPORTS_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["user", "report_type", "prediction", "recommendation", "datetime"])
            writer.writerow({
                "user": session["patient_name"],
                "report_type": "Diabetes",
                "prediction": prediction,
                "recommendation": recommendation,
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M")
            })

    return render_template("diabetes.html", prediction=prediction, recommendation=recommendation)

# ------------------ CHATBOT ------------------
@app.route("/chatbot", methods=["GET", "POST"])
@login_required
def chatbot():
    if request.method == "POST":
        try:
            user_input = request.json.get("message", "Hello")
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a friendly medical chatbot."},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.7,
                max_tokens=300
            )
            reply = res.choices[0].message.content
            return jsonify({"reply": reply})
        except Exception as e:
            return jsonify({"reply": f"Error: {str(e)}"}), 500
    return render_template("chatbot.html")

# ------------------ REPORTS ------------------
@app.route("/reports")
@login_required
def reports():
    user_reports = []
    with open(REPORTS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["user"] == session["patient_name"]:
                user_reports.append(row)
    return render_template("reports.html", reports=user_reports)

# ------------------ REMINDERS ------------------
@app.route("/reminder")
@login_required
def reminders_page():
    return render_template("reminder.html")

@app.route("/add_reminder", methods=["POST"])
@login_required
def add_reminder():
    user = session["patient_name"]
    task = request.form.get("task")
    date = request.form.get("date")
    time_ = request.form.get("time")
    dt = f"{date} {time_}"

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["user", "task", "datetime"])
        writer.writerow({"user": user, "task": task, "datetime": dt})

    return jsonify({"status": "success"})

@app.route("/check_reminder")
@login_required
def check_reminders():
    now = datetime.now()
    due_tasks = []

    with open(CSV_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                reminder_time = datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M")
                if row["user"] == session["patient_name"] and reminder_time >= now:
                    due_tasks.append(f"{row['task']} at {row['datetime']}")
            except ValueError:
                continue
    return jsonify(due_tasks)

# ------------------ MAIN ------------------
if __name__ == "__main__":
    app.run(debug=True)
