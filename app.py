# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template
import pickle
import logging

# Initialize Flask app
app = Flask("__name__")

# Load data and model
df_1 = pd.read_csv("first_telc.csv")
model = pickle.load(open("model.sav", "rb"))

# Define columns for new data input
columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
           'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
           'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
           'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
           'MonthlyCharges', 'TotalCharges']

# Define labels for tenure groups
labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]

# Logging configuration
logging.basicConfig(filename='error.log', level=logging.DEBUG)

@app.route("/")
def loadPage():
    return render_template('index.html')

@app.route("/", methods=['POST'])
def predict():
    try:
        # Extract and convert input queries with validation
        inputQuery1 = request.form['query1']  # gender
        inputQuery2 = int(request.form['query2'])  # SeniorCitizen
        if inputQuery2 not in [0, 1]:  # Check valid values
            raise ValueError("SeniorCitizen must be 0 or 1")
        inputQuery3 = request.form['query3']  # Partner
        inputQuery4 = request.form['query4']  # Dependents
        inputQuery5 = int(request.form['query5'])  # tenure
        inputQuery6 = request.form['query6']  # PhoneService
        inputQuery7 = request.form['query7']  # MultipleLines
        inputQuery8 = request.form['query8']  # InternetService
        inputQuery9 = request.form['query9']  # OnlineSecurity
        inputQuery10 = request.form['query10']  # OnlineBackup
        inputQuery11 = request.form['query11']  # DeviceProtection
        inputQuery12 = request.form['query12']  # TechSupport
        inputQuery13 = request.form['query13']  # StreamingTV
        inputQuery14 = request.form['query14']  # StreamingMovies
        inputQuery15 = request.form['query15']  # Contract
        inputQuery16 = request.form['query16']  # PaperlessBilling
        inputQuery17 = request.form['query17']  # PaymentMethod
        inputQuery18 = float(request.form['query18'])  # MonthlyCharges
        inputQuery19 = float(request.form['query19'])  # TotalCharges

        # Prepare the data for prediction
        data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, 
                 inputQuery6, inputQuery7, inputQuery8, inputQuery9, inputQuery10, 
                 inputQuery11, inputQuery12, inputQuery13, inputQuery14, 
                 inputQuery15, inputQuery16, inputQuery17, inputQuery18, 
                 inputQuery19]]
        
        new_df = pd.DataFrame(data, columns=columns)
        df_2 = pd.concat([df_1, new_df], ignore_index=True)

        # Preprocessing steps
        df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), 
                                       right=False, labels=labels)
        df_2.drop(columns=['tenure'], axis=1, inplace=True)

        # Get dummy variables and align columns
        new_df__dummies = pd.get_dummies(df_2[['gender', 'SeniorCitizen', 'Partner', 
                                                 'Dependents', 'PhoneService', 
                                                 'MultipleLines', 'InternetService', 
                                                 'OnlineSecurity', 'OnlineBackup', 
                                                 'DeviceProtection', 'TechSupport', 
                                                 'StreamingTV', 'StreamingMovies', 
                                                 'Contract', 'PaperlessBilling', 
                                                 'PaymentMethod', 'tenure_group']])
        model_columns = model.feature_names_in_
        new_df__dummies = new_df__dummies.reindex(columns=model_columns, fill_value=0)

        # Log the columns passed to the model for debugging
        logging.debug(f"Columns passed to the model: {new_df__dummies.columns.tolist()}")
        logging.debug(f"Input Data: {data}")

        # Make predictions
        single = model.predict(new_df__dummies.tail(1))
        probability = model.predict_proba(new_df__dummies.tail(1))[:, 1]

        # Prepare output based on prediction
        if single == 1:
            o1 = "This customer is likely to be churned!!"
        else:
            o1 = "This customer is likely to continue!!"
        o2 = f"Confidence: {probability[0] * 100:.2f}%"
        
        # Return the output to the HTML template
        return render_template('index.html', output1=o1, output2=o2)

    except Exception as e:
        logging.exception("Exception occurred during prediction")
        return render_template('index.html', error="An error occurred during prediction. Please check the logs for details."), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
