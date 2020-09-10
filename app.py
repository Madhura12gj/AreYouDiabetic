

import flask
from flask import Flask, jsonify, request, render_template
import json
#from data_input import data_in
import numpy as np
import pickle 

def load_models():
    file_name = "models/Finalized_model123.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    return model

app = Flask(__name__)


def predict(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,-1)
    # load model
    print(to_predict)
    model = load_models()
    prediction = model.predict(to_predict)
    response = json.dumps({'response': str(prediction)})
    print(int(prediction))
    return int(prediction)

@app.route('/predict', methods = ['GET','POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = predict(to_predict_list)
        if result == 0:
            response ='Non-Diabetic Patient Predicted'
        else:
            response ='Diabetic Patient Predicted'
        return render_template("result.html", prediction = response)
    else:
        return render_template("index.html")
        


if __name__ == '__main__':
    app.run(port=3000, debug=True, use_reloader = False)
    
    
    


