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
data_path = "data\dataset.csv"
data = pd.read_csv(data_path).dropna(axis = 1)

# Load and display the dataset
print(data.head())

# Removing the 'Unnamed: 133' column
# data_cleaned = data.drop(columns=['Unnamed: 133'])

# Removing duplicate rows
# data_cleaned = data.drop_duplicates()

# Check the shape of the data after cleaning
# data_shape_after_cleaning = data_cleaned.shape
# print(f"data shape after cleaning: {data_shape_after_cleaning}")

# Display a statistical summary of the dataset
# statistical_summary = data_cleaned.describe()
# print(f"statistical summary: {statistical_summary}")

# Check for missing values
missing_values = data.isnull().sum()
print(f"Missing values: {missing_values}")

# Check for duplicate rows
duplicate_rows = data.duplicated().sum()
print(f"Duplicate rows: {duplicate_rows}")

# check if the dataset is balanced or not
no_of_dizz = data["prognosis"].value_counts()
print(f"Number of diseases: {no_of_dizz}")

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

# output Train and Test shapes
print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")

# function that returns accuracy score
def cv_score(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))

# initialize models
models = {
    "SVC":SVC(),
    "Gaussian NB":GaussianNB(),
    "Random Forest":RandomForestClassifier(random_state=18)
}

# output cross validation of models
for mdl in models:
    model = models[mdl]
    scores = cross_val_score(model, X, y, cv = 10, 
                             n_jobs = -1, 
                             scoring = cv_score)
    print("=="*30)
    print(mdl)
    print(f"Scores: {scores}")
    print(f"Mean Score: {np.mean(scores)}")

# train and test data using svm classifier
svm_model = SVC()
svm_model.fit(X_train, y_train)
prediction = svm_model.predict(X_test)

print(f"Accuracy Score on Training data from SVM: {accuracy_score(y_train, svm_model.predict(X_train))*100}")
print(f"Accuracy Score on Test data from SVM: {accuracy_score(y_test, prediction)*100}")

cf_matrix = confusion_matrix(y_test, prediction)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for SVM Classifier on Test Data")
plt.show()

# Training and testing Naive Bayes Classifier
gn_model = GaussianNB()
gn_model.fit(X_train, y_train)
prediction = gn_model.predict(X_test)
print(f"Accuracy of train data using Naive Bayes Classifier: {accuracy_score(y_train, gn_model.predict(X_train))*100}")
print(f"Accuracy on test data using Naive Bayes Classifier: {accuracy_score(y_test, prediction)*100}")

cf_matrix = confusion_matrix(y_test, prediction)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix using Naive Bayes Classifier on Test Data")
plt.show()

# Training and testing using Random Forest Classifier
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)
prediction = rf_model.predict(X_test)
print(f"Accuracy of train data using Random Forest Classifier: {accuracy_score(y_train, rf_model.predict(X_train))*100}")
print(f"Accuracy on test data using Random Forest Classifier: {accuracy_score(y_test, prediction)*100}")
 
cf_matrix = confusion_matrix(y_test, prediction)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix using Random Forest Classifier on Test Data")
plt.show()