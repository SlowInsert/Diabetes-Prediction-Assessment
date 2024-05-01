# Imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score , recall_score
from sys import exit
# Deactivate future warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Load data

Diabeties_factors = pd.read_csv('Data.csv')

# Tests to see errors in data set to true to enable tests
test = True
if test:
    # print the first 5 rows of the data
    print(Diabeties_factors.head())
    # number of rows and column
    print(Diabeties_factors.shape)
    # info about data
    print(Diabeties_factors.info())
    # check for missing values
    print(Diabeties_factors.isnull().sum())
    print(Diabeties_factors.describe())

# Remove the 'id' column
Diabeties_factors = Diabeties_factors.drop(columns='ID', axis=1)
# remove patient number column
Diabeties_factors = Diabeties_factors.drop(columns='No_Pation', axis=1)

# fixing data by removing whitespace
Diabeties_factors['CLASS'] = Diabeties_factors['CLASS'].str.strip()
Diabeties_factors['Gender'] = Diabeties_factors['Gender'].str.upper()


# Replace 'M' with 0 and 'F' with 1 in the 'Gender' column
Diabeties_factors['Gender'] = Diabeties_factors['Gender'].replace({'M': 0, 'F': 1})

# Convert the data to numeric
Diabeties_factors['CLASS'] = Diabeties_factors['CLASS'].replace({'Y': 0, 'N': 1,'P': 2})

# splitting features and target
# features
X = Diabeties_factors.drop(columns='CLASS', axis=1)

# target
Y = Diabeties_factors['CLASS']

# scaling
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Model training
model = LogisticRegression(solver='saga', max_iter=1000000)
model.fit(X_train, Y_train)

# Training the Logistic Regression model with training data
model.fit(X_train, Y_train)

# Accuracy testing set to True to enable testing
Accuracy_testing = True
if Accuracy_testing:

    # Accuracy on training data
    X_train_prediction = model.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    print('Accuracy on training data : ', training_data_accuracy)

    # Accuracy on test data
    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    print('Accuracy on test data : ', test_data_accuracy)

    # Generate predictions
    Y_pred = model.predict(X_test)
    # Generate confusion matrix
    cm = confusion_matrix(Y_test, Y_pred)
    # Calculations of accuracy, precision and recall
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average='weighted')
    recall = recall_score(Y_test, Y_pred, average='weighted')
    print('Confusion Matrix: ')
    print(cm)
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)



while True:
    print('Enter data in the following format: "Gender(M/F) ,Age ,Creatinine ratio ,Body Mass Index (BMI) '
          ',Urea ,Cholesterol (Chol) ,Fasting lipid profile ,LDL ,VLDL ,Triglycerides(TG) and HDL '
          'Cholesterol ,HBA1C" ')
    Patient_Info = input("Enter patient Information('q' to quit): ")
    if Patient_Info.upper() == 'Q':
        exit()

    # split the data
    Patient_Info = Patient_Info.split(',')
    # convert F to 1 and M to 0
    if Patient_Info[0].upper() == 'M':
        Patient_Info[0] = 0
    elif Patient_Info[0].upper() == 'F':
        Patient_Info[0] = 1
    # change to numpy array
    Patient_Info = np.asarray(Patient_Info)
    # define column name
    column_names = ['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']

    # reshape the array
    Patient_Info_reshaped = Patient_Info.reshape(1, -1)
    # convert to dataframe
    Patient_Info_df = pd.DataFrame(Patient_Info_reshaped, columns=column_names)
    # standardize the data
    std_data = scaler.transform(Patient_Info_df)
    # making prediction
    prediction = model.predict(std_data)

    if prediction[0] == 1:
        print('The patient is not at risk of Diabeties')
        print("")
    elif prediction[0] == 0:
        print('The patient is at risk or already has diabetes')
        print("")
    elif prediction[0] == 2:
        print('The patient is at risk of Diabetes')
        print("")
#(1,50,4.7,46,4.9,4.2,0.9,2.4,1.4,0.5,24)

