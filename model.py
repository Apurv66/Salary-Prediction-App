import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import numpy as np

df = pd.read_csv("clean_data.csv")

categorical_cols = ['Gender', 'Education Level', 'Job Title']
 
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop(columns="Salary").values
y = df['Salary'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
print(f"Mean Absolute Error: {mae}")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

for col, le in encoders.items():
    with open(f"{col}_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

print("Model and encoders have been saved successfully.")

def encode_and_predict(age, gender, edu_level, job_title, year_exp):
    gender_val = encoders['Gender'].transform([gender])[0]
    edu_val = encoders['Education Level'].transform([edu_level])[0]
    job_val = encoders['Job Title'].transform([job_title.strip().lower()])[0]
    input_data = np.array([[age, gender_val, edu_val, job_val, year_exp]])
    return model.predict(input_data)[0]
