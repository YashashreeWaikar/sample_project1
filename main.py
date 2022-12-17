import pandas as pd
from flask import Flask, render_template, request
import pickle
import json
import numpy as np
import config
import sklearn

app= Flask(__name__)

#data= pd.read_csv('csv_file\cleaned_banglore_data.csv')
data= json.load(open(config.LOCATION_DATA_PATH,"r"))
pipe= pickle.load(open(config.MODEL_FILE_PATH,'rb'))

@app.route('/')
def index():

    #locations=sorted(data['location'].unique())
    locations= sorted(data['location'])
    return render_template('index_bang.html',locations=locations)

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method=='POST':
        location = request.form.get('location')
        bhk = float(request.form.get('bhk'))
        bath = float(request.form.get('bath'))
        sqft = request.form.get('total_sqft')

        print(location,bhk,bath,sqft)
        input= pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])
        prediction= np.round(pipe.predict(input)[0]* 1e5 ,2)

        return render_template('index_bang.html',prediction=prediction)
    
    else:
        location = request.form.get('location')
        bhk = float(request.form.get('bhk'))
        bath = float(request.form.get('bath'))
        sqft = request.form.get('total_sqft')

        print(location,bhk,bath,sqft)
        input= pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])
        prediction= np.round(pipe.predict(input)[0]* 1e5 ,2)

        return render_template('index_bang.html',prediction=prediction)


if __name__== '__main__':
    app.run(host= "0.0.0.0", port= config.PORT_NUMBER,debug=True)

    # 