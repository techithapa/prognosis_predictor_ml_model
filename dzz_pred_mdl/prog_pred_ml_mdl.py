# Import required libraries
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Training.csv and remove last empty column
DATA_PATH = "data\Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis = 1)

# check if the dataset is balanced or not
no_of_dizz = data["prognosis"].value_counts()
tmp_df = pd.DataFrame({
    "Diseases": no_of_dizz.index,
    "Counts" : no_of_dizz.values
})

# plot size
plt.figure(figsize = (18,8))

# populating variables for bar plot
sns.barplot(x = "Diseases", y = "Counts", data = tmp_df)

# rotate the plot
plt.xticks(rotation=90)

# print plot
# plt.show()

# encode the target variable using Lable encoder
encoder = LabelEncoder()
data['prognosis'] = encoder.fit_transform(data["prognosis"])

# split data for training and testing a model
X = data.iloc[:,:-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 24)

# output Train and Test shapes
print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")
