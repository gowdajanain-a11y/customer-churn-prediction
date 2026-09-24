import pandas as pd
from flask import Flask, request, render_template
import pickle
import traceback

app = Flask(__name__)

df_1 = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
model = pickle.load(open("model.sav", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

# Default values for the fields we are NOT asking the user about.
# These are typical / most common values from the dataset.
DEFAULTS = {
    "SeniorCitizen": 0,
    "gender": "Female",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "PaperlessBilling": "Yes",
}

# --- Per-customer explanation setup -----------------------------------
# Instead of showing the model's fixed "global" importances (same for every
# customer), we compare THIS customer's specific inputs against real
# historical churn rates for that same value. This changes per customer.

df_1['is_churn'] = (df_1['Churn'] == 'Yes').astype(int)
OVERALL_CHURN_RATE = df_1['is_churn'].mean() * 100

CONTRACT_RATES = df_1.groupby('Contract')['is_churn'].mean() * 100
INTERNET_RATES = df_1.groupby('InternetService')['is_churn'].mean() * 100
PAYMENT_RATES = df_1.groupby('PaymentMethod')['is_churn'].mean() * 100

# Bucket MonthlyCharges into Low/Medium/High using the training data's own
# thirds, then compute the historical churn rate for each bucket.
_charge_bins = pd.qcut(df_1['MonthlyCharges'], q=3, labels=['Low', 'Medium', 'High'])
df_1['_charge_bucket'] = _charge_bins
CHARGE_RATES = df_1.groupby('_charge_bucket', observed=True)['is_churn'].mean() * 100
_charge_edges = pd.qcut(df_1['MonthlyCharges'], q=3, retbins=True)[1]

# Bucket tenure into the same groups used for training (1-12, 13-24, ...)
_tenure_labels = [f"{i} - {i+11}" for i in range(1, 72, 12)]
df_1['_tenure_bucket'] = pd.cut(df_1['tenure'], range(1, 80, 12), right=False, labels=_tenure_labels)
TENURE_RATES = df_1.groupby('_tenure_bucket', observed=True)['is_churn'].mean() * 100


def bucket_monthly_charge(value):
    if value <= _charge_edges[1]:
        return 'Low'
    elif value <= _charge_edges[2]:
        return 'Medium'
    return 'High'


def bucket_tenure(value):
    idx = min(value // 12, len(_tenure_labels) - 1)
    return _tenure_labels[int(idx)]


def get_customer_explanation(tenure, monthly_charges, contract, internet_service, payment_method):
    """For THIS customer's 5 inputs, compare their historical churn rate
    against the overall average. Positive = raises risk, negative = lowers it."""
    charge_bucket = bucket_monthly_charge(monthly_charges)
    tenure_bucket = bucket_tenure(tenure)

    factors = [
        ("Contract: " + contract, CONTRACT_RATES.get(contract, OVERALL_CHURN_RATE)),
        ("Internet: " + internet_service, INTERNET_RATES.get(internet_service, OVERALL_CHURN_RATE)),
        ("Payment: " + payment_method, PAYMENT_RATES.get(payment_method, OVERALL_CHURN_RATE)),
        ("Monthly charges: " + charge_bucket, CHARGE_RATES.get(charge_bucket, OVERALL_CHURN_RATE)),
        ("Tenure: " + tenure_bucket + " months", TENURE_RATES.get(tenure_bucket, OVERALL_CHURN_RATE)),
    ]

    results = []
    for label, group_rate in factors:
        diff = round(group_rate - OVERALL_CHURN_RATE, 1)
        results.append((label, diff))

    # Sort by strongest effect first (regardless of direction)
    results.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return results

@app.route("/")
def loadPage():
    return render_template('home.html',
                            query1="", query2="", query3="", query4="", query5="",
                            error="")

@app.route("/", methods=['POST'])
def predict():
    try:
        tenure = request.form.get("query1", "").strip()
        monthly_charges = request.form.get("query2", "").strip()
        contract = request.form.get("query3", "").strip()
        internet_service = request.form.get("query4", "").strip()
        payment_method = request.form.get("query5", "").strip()

        if not tenure or not monthly_charges or not contract or not internet_service or not payment_method:
            return render_template('home.html',
                                    error="Please fill in all 5 fields!",
                                    query1=tenure, query2=monthly_charges, query3=contract,
                                    query4=internet_service, query5=payment_method)

        tenure = int(float(tenure))
        monthly_charges = float(monthly_charges)
        total_charges = tenure * monthly_charges  # simple, reasonable estimate

        row = {
            "SeniorCitizen": DEFAULTS["SeniorCitizen"],
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "gender": DEFAULTS["gender"],
            "Partner": DEFAULTS["Partner"],
            "Dependents": DEFAULTS["Dependents"],
            "PhoneService": DEFAULTS["PhoneService"],
            "MultipleLines": DEFAULTS["MultipleLines"],
            "InternetService": internet_service,
            "OnlineSecurity": DEFAULTS["OnlineSecurity"],
            "OnlineBackup": DEFAULTS["OnlineBackup"],
            "DeviceProtection": DEFAULTS["DeviceProtection"],
            "TechSupport": DEFAULTS["TechSupport"],
            "StreamingTV": DEFAULTS["StreamingTV"],
            "StreamingMovies": DEFAULTS["StreamingMovies"],
            "Contract": contract,
            "PaperlessBilling": DEFAULTS["PaperlessBilling"],
            "PaymentMethod": payment_method,
            "tenure": tenure,
        }

        new_df = pd.DataFrame([row])
        df_2 = pd.concat([df_1, new_df], ignore_index=True)

        labels = [f"{i} - {i+11}" for i in range(1, 72, 12)]
        df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
        df_2.drop(columns=['tenure'], inplace=True)

        df_encoded = pd.get_dummies(df_2[[
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group'
        ]])

        final_input = df_encoded.tail(1).reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(final_input)[0]
        prob_class_0 = model.predict_proba(final_input)[0][0]
        prob_class_1 = model.predict_proba(final_input)[0][1]

        if prediction == 1:
            prediction_text = "This customer is likely to be churned!!"
            confidence = prob_class_1 * 100
        else:
            prediction_text = "This customer is likely to continue!!"
            confidence = prob_class_0 * 100

        explanation = get_customer_explanation(
            tenure, monthly_charges, contract, internet_service, payment_method
        )

        return render_template('result.html',
                                prediction_text=prediction_text,
                                confidence=confidence,
                                explanation=explanation,
                                overall_rate=round(OVERALL_CHURN_RATE, 1))

    except Exception as e:
        traceback.print_exc()
        return render_template('home.html',
                                error=f"Internal Error: {str(e)}",
                                query1=request.form.get("query1", ""),
                                query2=request.form.get("query2", ""),
                                query3=request.form.get("query3", ""),
                                query4=request.form.get("query4", ""),
                                query5=request.form.get("query5", ""))

if __name__ == "__main__":
    app.run(debug=True)