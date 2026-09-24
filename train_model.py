import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Step 1: Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Step 2: Preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Create tenure_group column
df['tenure_group'] = pd.cut(df['tenure'], range(1, 80, 12), right=False,
                            labels=["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)])

# Drop unnecessary columns
df.drop(columns=['customerID', 'tenure'], inplace=True)

# Step 3: Prepare features and labels
X = pd.get_dummies(df.drop('Churn', axis=1))
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Step 4: Save model columns to model_columns.pkl
with open("model_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

# Step 5: Train model and save it to model.sav
model = RandomForestClassifier()
model.fit(X, y)

with open("model.sav", "wb") as f:
    pickle.dump(model, f)

print("✅ Model and model_columns.pkl saved successfully!")
