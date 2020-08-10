# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 06:54:18 2020

@author: sai
"""


from flask import Flask
from flask import request,render_template
from keras.models import load_model
 
model = load_model("PROJECT.h5")
import tensorflow  as tf
global graph
graph = tf.get_default_graph()
import numpy as np
app = Flask(__name__)
@app.route('/')
def hello_world():
    return render_template("index.html")
@app.route('/predict', methods = ["POST"])
def predict():
        
        encounter_id = request.form['encounter_id']
       
        patient_nbr = request.form.get('patient_number')
        admission_type_id = request.form.get('admission_type_id_Emergency')
        discharged_disposition_id = request.form.get('Discharged_to_home id')
        admission_source_id = request.form.get('admission__source_id_Emergency')
        medical_specialty = request.form.get('medical_specialty')
        time_in_hospital = request.form.get('time_in_hospital')
        num_lab_procedures = request.form.get('num_lab_procedures')
        num_procedures = request.form.get('num_procedures')
        num_medications = request.form.get('num_medications')
        number_outpatient = request.form.get('number_outpatient')
        number_emergency = request.form.get('number_emergency')
        number_inpatient = request.form.get('number_inpatient')
        number_diagnoses = request.form.get('number_diagnoses')
        ageGroup = request.form.get('ageGroup')
        has_weight = request.form.get('weight')
        race = request.form.get('race')
        gender = request.form.get('gender')
        max_glu_serum = request.form.get('max_glu_serum')
        A1Cresult = request.form.get('A1Cresult')
        metformin = request.form.get('metformin')
        repaglinide = request.form.get('repaglinide')
        nateglinide = request.form.get('nateglinide')
        chlorpropamide = request.form.get('chlorpropamide')
        glimepiride = request.form.get('glimepiride')
        acetohexamide = request.form.get('acetohexamide')
        glipizide = request.form.get('glipizide')
        glyburide = request.form.get('glyburide')
        tolbutamide = request.form.get('tolbutamide')
        pioglitazone = request.form.get('pioglitazone')
        rosiglitazone = request.form.get('rosiglitazone')
        acarbose = request.form.get('acarbose')
        miglitol = request.form.get('miglitol')
        troglitazone = request.form.get('troglitazone')
        tolazamide = request.form.get('tolazamide')
        examide = request.form.get('examide')
        citoglipton = request.form.get('citoglipton')
        insulin = request.form.get('insulin_no')
        glyburide_metformin = request.form.get('gluburide-metformin')
        glipizide_metformin = request.form.get('glipizide-metformin')
        glimepiride_pioglitazone = request.form.get('glimepiride-pioglitazone')
        metformin_rosiglitazone = request.form.get('metformin-rosiglitazone')
        metformin_pioglitazone  = request.form.get('metformin-pioglitazone')
        change = request.form.get('change_med_no')
        diabetesMed = request.form.get('insulin_Steady')
        
        total =  [encounter_id,
                 patient_nbr,
                 admission_type_id,
                 discharged_disposition_id,
                 admission_source_id,
                 medical_specialty,
                 time_in_hospital,
                 num_lab_procedures,
                 num_procedures,
                 num_medications,
                 number_outpatient,
                 number_emergency,
                 number_inpatient,
                 number_diagnoses,
                 ageGroup,
                 has_weight,
                 race,
                 gender,
                 max_glu_serum,
                 A1Cresult,
                 metformin,
                 repaglinide,
                 nateglinide,
                 chlorpropamide,
                 glimepiride,
                 acetohexamide,
                 glipizide,
                 glyburide,
                 tolbutamide,
                 pioglitazone,
                 rosiglitazone,
                 acarbose,
                 miglitol,
                 troglitazone,
                 tolazamide,
                 examide,
                 citoglipton,
                 insulin,
                 glyburide_metformin,
                 glipizide_metformin,
                 glimepiride_pioglitazone,
                 metformin_rosiglitazone,
                 metformin_pioglitazone,
                 change,
                 diabetesMed ]

        with graph.as_default():
            pred = model.predict_classes(np.matrix(total))
        index = ['readmittes more 30 days ','not readmitted',"readmittes less than 30 days"]
    
        return render_template("index.html",Submit = "patients will get readmitted = " + str(index[pred[0]]))

if __name__ == '__main__':
    app.run(debug = True)

    
    
    
    
