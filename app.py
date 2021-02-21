# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 00:29:58 2021
@author: lalatendu
"""

import pickle
from flask import Flask, request, jsonify, render_template
from flasgger import Swagger
import numpy as np
import pandas as pd
#from keras.models import model_from_json
import json
import pickle
import xgboost 
import joblib

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import load_model
 
# load model
nlp_model = load_model('model_LSTM.h5')





dict_path='./dictiornay_data'
model_path='./'

with open('{}onehot_encoding.pkl'.format(model_path), 'rb') as model_file:
    one_hot_encode_model= pickle.load(model_file)

'''
with open('{}accident_model.pkl'.format(model_path), 'rb') as model_file:
    acc_model = pickle.load(model_file)
'''

#xgb = joblib.load(fname)
    
with open('{}poteltial_accident_model.pkl'.format(model_path), 'rb') as model_file:
    pacc_model = pickle.load(model_file)
    





   
#with open('{}\Accident_Level.json'.format(dict_path)) as json_file: 
#    data = json.load(json_file) 


app = Flask(__name__)
swagger = Swagger(app)

#Week_day1 = {"Sunday":0,"Monday":1,"Tuesday":2,"Wednesday":3,"Thrusday":4,"Friday":5,"Saturday":6}
#Season1 = {"Spring":1,"Summer":2,"Autumn":3,"Winter":4}

def read_dictionary_data(file_name):
    with open('{}/{}'.format(dict_path,file_name)) as json_file: 
            data = json.load(json_file) 
    return data

def input_dataframe(lst):
    np_array = np.array(lst)

    reshaped_array = np.reshape(np_array, (1, len(lst)))
    df = pd.DataFrame(reshaped_array,columns=['Country','Local','Industry Sector','Gender','Employee type','Critical Risk','Year','Month','Day','Weekday','Season'])
    #X_cat=df[['Country','Local','Industry Sector','Gender','Employee type','Critical Risk','Year','Month','Day','Weekday','Season']]
    # X_nlp=df[['Description']]
    #print(df)
    return df

def input_description_dataframe(lst):
    np_array = np.array(lst)

    reshaped_array = np.reshape(np_array, (1, len(lst)))
    df = pd.DataFrame(reshaped_array,columns=['Description'])
    #X_cat=df[['Country','Local','Industry Sector','Gender','Employee type','Critical Risk','Year','Month','Day','Weekday','Season']]
    # X_nlp=df[['Description']]
    #print(df)
    return df

def pot_input_datafarame(lst):
    np_array = np.array(lst)

    reshaped_array = np.reshape(np_array, (1, len(lst)))
    
    df = pd.DataFrame(reshaped_array,columns=['Country','Local','Industry Sector','Gender','Employee type','Critical Risk','Year','Month','Day','Weekday','Season','Accident Level'])
    #X_cat=df[['Country','Local','Industry Sector','Gender','Employee type','Critical Risk','Year','Month','Day','Weekday','Season']]
   # X_nlp=df[['Description']]
    print(df)
    return df
'''
def read_model():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model=loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model'''

def tokenizer_text(data):
    print("tokenizer_text")
    max_features=10000
    maxlen = 300
    tokenizer = Tokenizer(num_words=max_features,lower = False)
    tokenizer.fit_on_texts(data)
    data = tokenizer.texts_to_sequences(data)
    data = pad_sequences(data, maxlen = maxlen, padding='post')
    return data

def month2seasons(x):
    if x in [9, 10, 11]:
        season = 'Spring'
    elif x in [12, 1, 2]:
        season = 'Summer'
    elif x in [3, 4, 5]:
        season = 'Autumn'
    elif x in [6, 7, 8]:
        season = 'Winter'
    print("season in function:",season)
    return season_lvl[season]

def data_extact(x):
    print("Date Extract Started")
    date = pd.to_datetime(x)
    print("date:",date)
    print("year:",date.year)
    year = date.year
    
    month = date.month
    print("month:",month)
    day = date.day
    week_day = weekday_lvl[date.day_name()]
    print("week_day:",week_day)
    season = month2seasons(month)
    print("season:",season)
    print(year,month,day,week_day,season)
    return year,month,day,week_day,season

def get_key(my_dict,val):
    for key, value in my_dict.items():
         if val == value:
             return key
 
    return "No Result"


accident_lvl = read_dictionary_data('Accident_Level.json')    
country_lvl = read_dictionary_data('country.json') 
critical_Risk_lvl = read_dictionary_data('Critical_Risk.json')
employee_type_lvl = read_dictionary_data('Employee_type.json')
gender_lvl = read_dictionary_data('Gender.json')
industry_sector_lvl = read_dictionary_data('Industry_Sector.json')
local_lvl = read_dictionary_data('Local.json')
industry_sector_lvl = read_dictionary_data('Industry_Sector.json')
potential_Accident_Level_lvl = read_dictionary_data('Potential_Accident_Level.json')
season_lvl = read_dictionary_data('Season.json')
weekday_lvl = read_dictionary_data('Weekday.json')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict_json():
    input= request.get_json()
    
    print(input)
    #input_data = json.loads(input)
    input_data=input
    print(input_data["accident_deatils"])
    #print(input_data)
    
    
    data = input_data["accident_deatils"]
    country	= country_lvl[data.get("Country")]
    print("Country:",country)
    local   = local_lvl[data.get("Local")]
    industry_Sector = industry_sector_lvl[data.get("Industry_Sector")]
    gender = gender_lvl[data.get("Gender")]
    employee_type = employee_type_lvl[data.get("Employee_Type")]
    critical_risk = critical_Risk_lvl[data.get("Critical_Risk")]
    year,month,day,weekday,season=data_extact(data.get("Date"))	
    description = data.get("Description")
    
    print("Input data :")        
    print(country,local,industry_Sector,gender,employee_type,critical_risk,year,month,day,weekday,season)

    print(description)

    cat_data=[country,local,industry_Sector,gender,employee_type,critical_risk,year,month,day,weekday,season]
    print("\n\n\n")
    print(cat_data)
    #nlp_data,cat_data=input_datafarame([country,local,industry_Sector,gender,employee_type,critical_risk,year,month,day,weekday,season,description])
    
    #nlp_data = tokenizer_text(nlp_data)
    #print(nlp_data)
    inut_data=input_dataframe(cat_data)
    
    
    
    input_description = input_description_dataframe([description])
    input_des=tokenizer_text(input_description.Description)
    
    accident_lvl_pred= nlp_model.predict(input_des)
    
    
    
    acc_pred=one_hot_encode_model.inverse_transform(accident_lvl_pred)[0][0]
    
    print("*****************************************************************")
    print(one_hot_encode_model.inverse_transform(accident_lvl_pred)[0][0])
    print("*****************************************************************")
    
    
    
    print(inut_data)
    
    #acc_pred = acc_model.predict(inut_data)
    #acc_pred=1
    #print(acc_pred)
    
    pot_inut_data=pot_input_datafarame([country,local,industry_Sector,gender,employee_type,critical_risk,year,month,day,weekday,season,acc_pred])
    print(pot_inut_data)
    
    #acc_pred = get_key(accident_lvl,acc_pred)
    
    pot_acc_pred= pacc_model.predict(pot_inut_data)
    
    acc_lvl= get_key(accident_lvl,acc_pred)
    
    pot_acc_lvl= get_key(potential_Accident_Level_lvl,pot_acc_pred)
    #model = read_model()
    #print(model.summary())
    
    
    print(acc_lvl,pot_acc_lvl)
    
    
   # y_pred = model.predict(x=[nlp_data, cat_data], batch_size=1024, verbose=1)
    y_pred=1
    print("Lala")
    print(input)
    #print(jsonify(input))
    #input_data = pd.read_csv(request.files.get("input_file"), header=None)
    #prediction = model.predict(input_data)

    return jsonify({"Accident pedict":acc_lvl,"Potential Accident pedict":pot_acc_lvl})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    #app.run()
    