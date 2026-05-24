import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib

print("Loading loan Approval dataset")
url = "https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/Loan-Approval-Prediction.csv"
df = pd.read_csv(url)

print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print("\nLoan Approval Distribution:")
print(df['Loan_Status'].value_counts())
print(f"Approval Rate: {df['Loan_Status'].value_counts(normalize=True)['Y']*100:.2f}%")

df = df.drop('Loan_ID', axis=1)

df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mode()[0])
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['LoanAmount_per_Income'] = df['LoanAmount'] /  (df['Total_Income'] + 1)

categorical_cols = ['Gender', 'Married', 'Dependents', 'Education',
                    'Self_Employed', 'Property_Area']

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

x = df.drop('Loan_Status', axis=1)
y= df['Loan_Status']

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(f"\nTraining samples: {x_train.shape[0]}")
print(f"Testing samples: {x_test.shape[0]}")

print("\n Training models....")

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(x_train_scaled, y_train)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(x_train_scaled, y_train)

print("Models trained successfully!")

def evaluate_mode(model, x_test, y_test, model_name):
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n {model_name} Performance:")
    print(f"Accuracy : {acc:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

print("="*50)
evaluate_mode(lr_model, x_test_scaled, y_test, "Logistic Regression")
evaluate_mode(rf_model, x_test_scaled, y_test, "Random Forest")

joblib.dump(rf_model, 'loan_approval_rf_model.pkl')
joblib.dump(scaler, 'loan_scaler.pkl')
joblib.dump(le_dict, 'loan_label_encoders.pkl')
print("\n Best model and preprocessors saved successfully!")

def predict_loan_approval(applicant_data):
    model = joblib.load('loan_approval_rf_model.pkl')
    scaler = joblib.load('loan_scaler.pkl')
    encoders = joblib.load('loan_label_encoders.pkl')

    df_pred = pd.DataFrame([applicant_data])

    df_pred['Total_Income'] = df_pred['ApplicantIncome'] + df_pred['CoapplicantIncome']
    df_pred['LoanAmount_per_Income'] = df_pred['LoanAmount'] / (df_pred['Total_Income'] +1)

    for col, le in encoders.items():
        if col in df_pred.columns:
            df_pred[col] = le.transform(df_pred[col])

    feature_order = [
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
        'Credit_History', 'Property_Area', 'Total_Income', 'LoanAmount_per_Income'
    ]    

    df_pred = df_pred[feature_order]

    features_scaled = scaler.transform(df_pred)
    probability = model.predict_proba(features_scaled)[0][1]
    prediction = "Approved" if probability >= 0.5 else "Rejected"

    return {
        "Loan_Status": prediction,
        "Approval_Probability": round(probability * 100,2)
    }

if __name__ == "__main__":
    print("\n" + "="*60)
    print("LOAN APPROVAL PREDICTION SYSTEM READY")
    print("="*60)


    example_applicant = {
        'Gender': 'Male',
        'Married':'Yes',
        'Dependents' :'1',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 4583,
        'CoapplicantIncome': 1508,
        'LoanAmount': 128,
        'Loan_Amount_Term': 360,
        'Credit_History': 1,
        'Property_Area': 'Rural'
    }

    result = predict_loan_approval(example_applicant)
    print("\nExample Applicant Prediction:")
    print(f"Loan Status: {result['Loan_Status']}")
    print(f"Approval Probablity: {result['Approval_Probability']}%")
