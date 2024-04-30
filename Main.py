#imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load and fix data

Diabeties_factors = pd.read_csv('Data.csv')

#print the first 5 rows of the data
print(Diabeties_factors.head())
#number of rows and collums
print(Diabeties_factors.shape)
#info about data
print(Diabeties_factors.info())
#check for missing values
print(Diabeties_factors.isnull().sum())