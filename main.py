import numpy as np
from flask import Flask,request,jsonify,render_template
import joblib
from tensorflow.keras.models import load_model

flower_model = load_model('flower_model.h5')
flower_scaler = joblib.load('flower_scaler.pkl')

def prediction_fun(model,scaler,data):
    s_len = data["sepal_length"]
    s_wid = data["sepal_width"]
    p_len = data["petal_length"]
    p_wid = data["petal_width"]
    
    flower = [[s_len,s_wid,p_len,p_wid]]
    classes = np.array(['setosa','versicolor','verginica'])
    
    #flower = scaler.fit_transform(flower)
    flower_scaled = scaler.transform(flower)
    
    class_ind = model.predict_classes(flower_scaled)[0]
    #class_indx = np.argmax(model.predict(flower_scaled),axis=-1)
    
    return classes[class_ind]


app = Flask(__name__)


@app.route('/api/flower',methods=['POST'])
def flower_prediction():
    content = request.json
    results = prediction_fun(flower_model,flower_scaler,content)
    return jsonify(results)

@app.route('/')
def home():
    return render_template("home.html")

if __name__ == '__main__':
    app.run()
