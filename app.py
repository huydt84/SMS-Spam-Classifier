from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl", 'rb'))
vector = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST", "GET"])
def predict():
    msg = request.form['message']
    msg_cv = vector.transform([msg])
    classification = model.predict(msg_cv)
    if classification[0]==0:
        prediction = "This is a spam message"
    else:
        prediction = "This is a non-spam message"
    return render_template('index.html',prediction=prediction)

if __name__ == '__main__':
    app.run()