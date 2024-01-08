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
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold


# Load Training.csv and remove last empty column
data_path = "data\dataset.csv"
raw_data = pd.read_csv(data_path).dropna(axis = 1)
# display the dataset
# print(data)

# Check the shape of the data after cleaning
data_shape_before_cleaning = raw_data.shape
print(f"data shape before cleaning: {data_shape_before_cleaning}")  

# Check for missing values
missing_values = raw_data.isnull().sum()
# print(f"Missing values: {missing_values}")

# Check for duplicate rows
duplicate_rows = raw_data.duplicated().sum()
print(f"Duplicate rows: {duplicate_rows}")

# Removing the 'Unnamed: 133' column
for column in raw_data.columns:
    if raw_data[column].empty:
        print(f"The column '{column}' is empty.")
    else:
        None

# Removing duplicate rows
data = raw_data.drop_duplicates()
# print(f"data cleaned: {data}")

# Check the shape of the data after cleaning
data_shape_after_cleaning = data.shape
print(f"data shape after cleaning: {data_shape_after_cleaning}")

# Display a statistical summary of the dataset
statistical_summary = data.describe()
# print(f"statistical summary: {statistical_summary}")

# check if the dataset is balanced or not
no_of_dizz = data["prognosis"].value_counts()
# print(f"Number of diseases: {no_of_dizz}")

# Dataframe with number of diseases and their counts
tmp_df = pd.DataFrame({
    "Diseases": no_of_dizz.index,
    "Counts" : no_of_dizz.values
})

# plot size
plt.figure(figsize = (18,8))
# populating variables for bar plot
sns.barplot(x = "Diseases", y = "Counts", data = tmp_df)
plt.title(f"Bar plot showing counts per diseases")
# rotate the plot
plt.xticks(rotation=90)
# print plot
plt.show()

# encode the target variable using Label encoder
encoder = LabelEncoder()
en_prog = encoder.fit_transform(data["prognosis"])

# Separate features and target variable
X = data.drop('prognosis', axis=1)  # Features (symptoms)
y = data['prognosis']               # Target variable (prognosis)

# Split the dataset into the Training set and Validation set (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

# output Train and Test shapes
print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_val.shape}, {y_val.shape}")

# a common function for different models to train and test data
def model_train_test(model, X_train, y_train, X_test, y_test):
    """
    Train and test a machine learning model.

    Parameters:
    model: The machine learning model to be trained and tested.
    X_train: Training data features.
    y_train: Training data labels.
    X_test: Testing data features.
    y_test: Testing data labels.
    """

    # Model training
    model.fit(X_train, y_train)

    # Predictions on the test set
    pred = model.predict(X_test)

     # Evaluate Cross validation 
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    cross_val_results = cross_val_score(model, X, y, cv=kf)*100
    print(f'\n{model.__class__.__name__} Cross-Validation Results (Accuracy): {cross_val_results}')
    print(f'\n{model.__class__.__name__} Cross-Validation Mean Accuracy: {cross_val_results.mean()}')

    # Accuracy score
    Accuracy_train = accuracy_score(y_train, model.predict(X_train))*100
    Accuracy_test = accuracy_score(y_test, pred)*100
    print(f"\nAccuracy score on train data using {model.__class__.__name__}: {Accuracy_train}")
    print(f"\nAccuracy score on test data using {model.__class__.__name__}: {Accuracy_test}")
    
    # Classification report
    print(f"\n{model.__class__.__name__} Classification Report:")
    print(classification_report(y_test, pred, zero_division=1))

    # graphical representation of confusion matrix of different models
    cf_matrix = confusion_matrix(y_test, pred)
    plt.figure(figsize=(12,8))
    sns.heatmap(cf_matrix, annot=True)
    plt.title(f"{model.__class__.__name__}-Confusion Matrix on Test Data")
    plt.show()

    return model, Accuracy_train, Accuracy_test

# Creating the Decision Tree classifier
dt_classifier = DecisionTreeClassifier(max_depth=100, random_state=42)
model_train_test(dt_classifier, X_train, y_train, X_val, y_val)

# Instantiate the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
model_train_test(rf_classifier, X_train, y_train, X_val, y_val)

# Instantiate the SVC Classifier
svm_classifier = SVC(kernel='linear')
model_train_test(svm_classifier, X_train, y_train, X_val, y_val)

# Instantiate the GaussianNB Classifier
nb_classifier = GaussianNB()
model_train_test(nb_classifier, X_train, y_train, X_val, y_val)

