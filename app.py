import pandas as pd
from flask import Flask, request, render_template
import pickle
import traceback
import os

# Initialize Flask app
app = Flask(__name__)

# Load initial dataset and model artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df_1 = pd.read_csv(os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv"))
with open(os.path.join(BASE_DIR, "model.sav"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, "model_columns.pkl"), "rb") as f:
    model_columns = pickle.load(f)

@app.route("/")
def loadPage():
    """
    Render the initial home page with empty form fields.
    """
    return render_template('home.html', 
                          **{f"query{i}": "" for i in range(1, 20)},
                          error="")

@app.route("/", methods=['POST'])
def predict():
    """
    Handle form submission, process input, make prediction, and redirect to result page.
    """
    try:
        # Collect all 19 input fields from the form
        inputs = [request.form.get(f"query{i}", "").strip() for i in range(1, 20)]

        # Validate critical numeric fields (SeniorCitizen, MonthlyCharges, TotalCharges, tenure)
        if not inputs[0] or not inputs[1] or not inputs[2] or not inputs[18]:
            return render_template('home.html', 
                                 error="Please fill all numeric fields (SeniorCitizen, MonthlyCharges, TotalCharges, tenure)!",
                                 **{f"query{i}": inputs[i-1] for i in range(1, 20)})

        # Safely convert numeric inputs
        inputs[0] = int(float(inputs[0]))     # SeniorCitizen (0 or 1)
        inputs[1] = float(inputs[1])          # MonthlyCharges
        inputs[2] = float(inputs[2])          # TotalCharges
        inputs[18] = int(float(inputs[18]))   # tenure

        # Create new DataFrame with input data
        data = [inputs]
        new_df = pd.DataFrame(data, columns=[
            'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
            'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'tenure'
        ])

        # Concatenate with original dataset for consistent encoding
        df_2 = pd.concat([df_1, new_df], ignore_index=True)

        # Create tenure group
        labels = [f"{i} - {i+11}" for i in range(1, 72, 12)]
        df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
        df_2.drop(columns=['tenure'], inplace=True)

        # One-hot encode categorical variables
        df_encoded = pd.get_dummies(df_2[[
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group'
        ]])

        # Align with model columns and fill missing with 0
        final_input = df_encoded.tail(1).reindex(columns=model_columns, fill_value=0)

        # Make prediction
        prediction = model.predict(final_input)[0]
        prob_class_0 = model.predict_proba(final_input)[0][0]
        prob_class_1 = model.predict_proba(final_input)[0][1]

        # Prepare output messages
        if prediction == 1:
            prediction_text = "This customer is likely to be churned!!"
            confidence = prob_class_1 * 100
        else:
            prediction_text = "This customer is likely to continue!!"
            confidence = prob_class_0 * 100

        # Redirect to result page with prediction data
        return render_template('result.html', 
                              prediction_text=prediction_text, 
                              confidence=confidence)

    except Exception as e:
        traceback.print_exc()  # Print full stack trace to console
        return render_template('home.html', 
                              error=f"Internal Error: {str(e)}",
                              **{f"query{i}": request.form.get(f"query{i}", "") for i in range(1, 20)})

if __name__ == "__main__":
    app.run(debug=True)