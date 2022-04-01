from flask import Flask, render_template, request
#import joblib
import os
import numpy as np
import pickle

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/result", methods=['POST', 'GET'])
def result():
    hypertension = float(request.form['hypertension'])
    heart_disease = float(request.form['heart_disease'])
    age = float(request.form['age'])
    x = np.array([hypertension, heart_disease, age]).reshape(1, -1)
    #pickled_model1 = pickle.load(open('model.pkl', 'rb'))
    #pickled_model2 = pickle.load(open('bestParam.pkl', 'rb'))
    pickled_model3 = pickle.load(open('bestModel.pkl', 'rb'))
    Y_pred = pickled_model3[0].predict(x)

    # for No Stroke Risk
    if Y_pred[0][0]<=5.50:
        return render_template('nostroke1.html')
    elif Y_pred[0][0]>5.50 and Y_pred[0][0]<=9.0:
        return render_template('nostroke2.html')
    elif Y_pred[0][0]>9.0 and Y_pred[0][0]<=13.10:
         return render_template('stroke1.html')
    else: 
          return render_template('stroke2.html')


if __name__ == "__main__":
    #app.run(debug=True, port=7385)
    app.run(debug=True)
