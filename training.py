# load libraries
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
from pprint import pprint

# load data
data = pd.read_csv('database/diabetes.csv')
print(data.describe())

# clean the dataset replace zeros with NANs
data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

#function to impute the missing values with median based on Outcome class
def impute_median(data, var):
  temp = data[data[var].notnull()]
  temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median()
  data.loc[(data['Outcome'] == 0 ) & (data[var].isnull()), var] = temp.loc[0 ,var]
  data.loc[(data['Outcome'] == 1 ) & (data[var].isnull()), var] = temp.loc[1 ,var]
  return data

#impute values using the function
data = impute_median(data, 'Glucose')
data = impute_median(data, 'BloodPressure')
data = impute_median(data, 'SkinThickness')
data = impute_median(data, 'Insulin')
data = impute_median(data, 'BMI')


x = data.drop('Outcome', axis=1)
y = data['Outcome']
columns = x.columns

# scale the dataset
scaler = StandardScaler()
scaler.fit(x)
#joblib.dump(scaler, 'models/scaler.joblib')
X = scaler.transform(x)
X = pd.DataFrame(X, columns=columns)

# partition the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define a baseline model
model = RandomForestClassifier(random_state=42)
pprint(model.get_params())

# train and evaluate the model
model.fit(X_train, y_train)
#joblib.dump(model, 'models/rfc-model.joblib')
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))
