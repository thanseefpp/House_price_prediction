# Importing Libraries and Packages
from flask import Flask, render_template, jsonify, request, url_for
import pickle
import numpy as np
import pandas as pd
import json

app = Flask(__name__)

# loading the pickle ML files
regression_model = pickle.load(open("house_price_predict_ml_model.pkl", 'rb'))
standardization = pickle.load(open("standardization.pkl", 'rb'))


@app.route('/')
def landing_page():
    return render_template("index.html")


# creating an api for testing
@app.route('/predict_api', methods=['POST'])
def predict_score():
    data = request.json['data']
    # print(data)
    reshaped_data = np.array(list(data.values())).reshape(1, -1)
    standardizing_data = standardization.transform(reshaped_data)
    predicted_result = regression_model.predict(standardizing_data)
    # print(predicted_result)
    return jsonify(predicted_result[0])


@app.route('/predict', methods=['POST'])
def predict_data():
    if request.method == "POST":
        CRIM    = request.form['CRIM']
        ZN      = request.form['ZN']
        INDUS   = request.form['INDUS'] 
        CHAS    = request.form['CHAS'] 
        NOX     = request.form['NOX'] 
        RM      = request.form['RM'] 
        AGE     = request.form['AGE'] 
        DIS     = request.form['DIS'] 
        RAD     = request.form['RAD'] 
        TAX     = request.form['TAX'] 
        PTRATIO = request.form['PTRATIO'] 
        B       = request.form['B'] 
        LSTAT   = request.form['LSTAT']
        # data = [float(x) for x in request.form.values()]
        data = [CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT]
        standardized_data = standardization.transform(np.array(data).reshape(1, -1))
        predicted_result = regression_model.predict(standardized_data)
        # return render_template('index.html', predicted_score="The House Price Predicted {}".format(predicted_result))
        return jsonify({'predicted_score':"Predicted Price : {}".format(round(predicted_result[0]))})
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug = True)


# {
#  "data":{
#    "CRIM" : 0.00632,
#    "ZN" : 18.0,
#    "INDUS" : 2.31,
#    "CHAS" : 0.0,
#    "NOX" : 0.538,
#    "RM" : 6.575,
#    "AGE" : 65.2,
#    "DIS" : 4.0900,
#    "RAD" : 1.0,
#    "TAX" : 296.0,
#    "PTRATIO" : 15.3,
#    "B" : 396.90,
#    "LSTAT" : 4.98,
#  }
#  }
