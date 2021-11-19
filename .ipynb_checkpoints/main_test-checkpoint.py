import numpy as np
from flask import Flask,request,render_template,redirect,url_for
import joblib
from tensorflow.keras.models import load_model

flower_model = load_model('flower_model.h5')
flower_scaler = joblib.load('flower_scaler.pkl')



app = Flask(__name__)


@app.route('/<result>')
def flower_prediction(result):
    return f"<h1>flower is: {result}</h1>"

@app.route('/',methods=["POST","GET"])
def home():
    if request.method == 'POST':
        plen = request.form['plen']
        pwed = request.form['pwid']
        slen = request.form['slen']
        swed = request.form['swid']

        flower = np.array([[plen,pwed,slen,swed]])

        classes = np.array(['setosa','versicolor','verginica'])

        flower_scaled = flower_scaler.transform(flower)
    
        class_ind = flower_model.predict_classes(flower_scaled)[0]

        return redirect(url_for("flower_prediction",result=classes[class_ind]))
        

    else:


        return render_template("home.html")

if __name__ == '__main__':
    app.run()
