from flask import Flask, render_template, request, send_file, redirect, url_for, session
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import tensorflow as tf
from reportlab.pdfgen import canvas
import requests
import numpy as np
from PIL import Image
import json
import gdown

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['REPORT_FOLDER'] = 'static'
app.secret_key = 'your_secret_key'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = 'best_model.h5'
if not os.path.exists(MODEL_PATH):
    
    print("Downloading the model from Google Drive...")
    # Replace with your Google Drive file ID
    file_id = '13GN1t5pSFiJ5XXSc4HeJ2AQour-cahfs'
    gdown.download(f'https://drive.google.com/uc?export=download&id={file_id}',MODEL_PATH, quiet=False)
else:
    print("Model already exists. Skipping download.")

# Load your model
model = tf.keras.models.load_model(MODEL_PATH)

# Classes
classes = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

USER_FILE = 'users.json'

# Load existing users from file
if os.path.exists(USER_FILE):
    with open(USER_FILE, 'r') as f:
        users_db = json.load(f)
else:
    users_db = {}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        username = request.form['email']
        password = request.form['password']

        # Check if username and password match
        if username in users_db and users_db[username] == password:
            # Redirect to index page after successful login
            return redirect(url_for('index'))
        else:
            # Return back to sign-in with error message if credentials are wrong
            return render_template('signin.html', error='Invalid credentials')

    return render_template('signin.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['email']
        password = request.form['password']

        if username in users_db:
            return render_template('signup.html', error='User already exists')

        users_db[username] = password

        # Save the new user to the file
        with open(USER_FILE, 'w') as f:
            json.dump(users_db, f)

        return redirect(url_for('signin'))  # Redirect to sign-in page after successful signup
    return render_template('signup.html')


def model_predict(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_tensor)
    predicted_class = classes[prediction.argmax()]
    return predicted_class


def get_suggestion_from_custom_model(diagnosis):
    suggestions = {
        'Mild Demented': [
            "Engage in regular mental stimulation activities like puzzles, reading, or learning new skills.",
            "Follow a structured daily routine to reduce confusion and build confidence.",
            "Participate in social activities to stay connected and supported.",
            "Eat a heart-healthy diet (e.g., Mediterranean diet) to support brain function.",
            "Discuss early-stage treatment options and planning with a healthcare provider."
        ],
        'Moderate Demented': [
            "Ensure a safe home environment—remove tripping hazards and use labels or reminders.",
            "Provide consistent emotional and physical support, including help with daily activities.",
            "Use memory aids like notes, apps, or caregiver support to manage confusion.",
            "Keep communication simple, clear, and reassuring.",
            "Talk to a doctor about medications or therapies that can help with mood and behavior."
        ],
        'Non Demented': [
            "Continue healthy habits like regular physical activity, a nutritious diet, and mental exercises.",
            "Engage in new learning opportunities or hobbies to build cognitive reserve.",
            "Stay socially active and maintain strong relationships with friends and family.",
            "Get regular sleep and manage stress for overall well-being.",
            "Have routine checkups and monitor cognitive health proactively."
        ],
        'Very Mild Demented': [
            "Focus on early detection and lifestyle changes to slow progression.",
            "Track symptoms and changes in memory with journals or apps.",
            "Encourage light physical activity like walking or yoga to boost cognitive function.",
            "Use simple reminders and organizational tools to assist with daily tasks.",
            "Start planning future care with family and medical professionals while decision-making is strong."
        ]
    }

    return "\n".join(suggestions.get(diagnosis, ["No suggestions available for this diagnosis."]))


def generate_pdf_report(prediction, suggestion):
    file_path = os.path.join(app.config['REPORT_FOLDER'], "report.pdf")
    c = canvas.Canvas(file_path)

    c.setFont("Helvetica-Bold", 20)
    c.drawString(100, 800, "Alzheimer's Disease Prediction Report")

    c.setFont("Helvetica", 14)
    c.drawString(100, 760, f"Prediction: {prediction}")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 730, "Suggestions:")

    text = c.beginText(100, 710)
    text.setFont("Helvetica", 12)
    for line in suggestion.split('\n'):
        text.textLine(line)
    c.drawText(text)

    disclaimer = "\nDisclaimer: This suggestion is generated by an AI language model (GPT-Neo) and should not be considered professional medical advice. Please consult a certified healthcare provider for an accurate diagnosis and guidance."
    text2 = c.beginText(100, 600)
    text2.setFont("Helvetica-Oblique", 10)
    for line in disclaimer.split('\n'):
        text2.textLine(line)
    c.drawText(text2)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(100, 50, f"Report generated on: {now}")

    c.save()
    return file_path


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    suggestion = None
    report_path = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            prediction = model_predict(filepath)
            suggestion = get_suggestion_from_custom_model(prediction)
            report_path = generate_pdf_report(prediction, suggestion)

            session['report_ready'] = True  # Flag to enable download button

    return render_template('predict.html', prediction=prediction, suggestion=suggestion, report_path=report_path)


@app.route('/download_report')
def download_report():
    report_file = os.path.join(app.config['REPORT_FOLDER'], "report.pdf")
    if os.path.exists(report_file):
        # Clear the session flag after download
        session.pop('report_ready', None)
        return send_file(report_file, as_attachment=True)
    return "No report available.", 404




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # use PORT from environment
    app.run(host="0.0.0.0", port=port)
    
