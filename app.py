import os
import json
import smtplib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import face_recognition
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DATA_FILE = 'data.json'

def load_data():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'w') as f:
            json.dump({"names": [], "emails": [], "filenames": []}, f)
    with open(DATA_FILE, 'r') as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f)

def send_email(to_email, subject, body, attachment_path=None):
    SENDER_EMAIL = 'bunnyjamuna2004@gmail.com'  # your email
    SENDER_PASSWORD = 'pveiktfefnhzykij'  # your app password

    try:
        message = MIMEMultipart()
        message['From'] = SENDER_EMAIL
        message['To'] = to_email
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))

        if attachment_path:
            with open(attachment_path, "rb") as attachment:
                part = MIMEApplication(attachment.read(), Name=os.path.basename(attachment_path))
                part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
                message.attach(part)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(message)
        server.quit()

    except Exception as e:
        print("❌ Failed to send email:", str(e))

def load_known_faces():
    data = load_data()
    known_encodings = []
    names = []
    for name, filename in zip(data['names'], data['filenames']):
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            names.append(name)
    return names, known_encodings

def evaluate_model(known_encodings, known_names, uploaded_encoding, threshold=0.6):
    # Compare the uploaded encoding to known encodings
    distances = face_recognition.face_distance(known_encodings, uploaded_encoding)
    min_distance = np.min(distances)
    min_index = np.argmin(distances)

    # Simple binary classification for demonstration (Known vs Unknown)
    y_true = ['Known']  # We know uploaded image is "Known" if matched, else Unknown
    y_pred = []

    if min_distance < threshold:
        y_pred.append('Known')
        matched_name = known_names[min_index]
    else:
        y_pred.append('Unknown')
        matched_name = None

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='Known', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=['Known', 'Unknown'])

    # Plot confusion matrix
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Known', 'Unknown'], yticklabels=['Known', 'Unknown'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    cm_path = os.path.join('static', 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()

    return accuracy, precision, cm_path, matched_name

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        file = request.files['file']

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            data = load_data()
            data['names'].append(name)
            data['emails'].append(email)
            data['filenames'].append(filename)
            save_data(data)

            return redirect(url_for('index'))
    return render_template('add.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            uploaded_filename = secure_filename(file.filename)
            uploaded_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)
            file.save(uploaded_path)

            uploaded_image = face_recognition.load_image_file(uploaded_path)
            uploaded_encodings = face_recognition.face_encodings(uploaded_image)
            if not uploaded_encodings:
                return "No face found in the uploaded image."

            uploaded_encoding = uploaded_encodings[0]

            known_names, known_encodings = load_known_faces()

            accuracy, precision, cm_path, matched_name = evaluate_model(
                known_encodings, known_names, uploaded_encoding)

            if matched_name:
                # Find email & filename for matched name
                data = load_data()
                idx = data['names'].index(matched_name)
                parent_email = data['emails'][idx]
                matched_image = data['filenames'][idx]

                # Send email notification
                send_email(parent_email, "Child Found",
                           f"Your child {matched_name} was found.",
                           attachment_path=uploaded_path)

                return render_template('result.html',
                                       matched=True,
                                       name=matched_name,
                                       email=parent_email,
                                       uploaded_image=uploaded_filename,
                                       matched_image=matched_image,
                                       accuracy=accuracy,
                                       precision=precision,
                                       confusion_matrix_image='confusion_matrix.png')

            else:
                return render_template('result.html', matched=False,
                                       accuracy=accuracy,
                                       precision=precision,
                                       confusion_matrix_image='confusion_matrix.png')

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
