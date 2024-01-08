# Import required libraries
from collections import Counter
from collections import defaultdict
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
'''
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
'''
# encode the target variable using Label encoder
encoder = LabelEncoder()
en_prog = encoder.fit_transform(data["prognosis"])

# Separate features and target variable
X = data.drop('prognosis', axis=1)  # Features (symptoms)
y = en_prog               # Target variable (prognosis)

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

    # Accuracy score
    Accuracy_train = accuracy_score(y_train, model.predict(X_train))*100
    Accuracy_test = accuracy_score(y_test, pred)*100
    print(f"\nAccuracy score on train data using {model.__class__.__name__}: {Accuracy_train}")
    print(f"\nAccuracy score on test data using {model.__class__.__name__}: {Accuracy_test}")
    
     # Evaluate Cross validation 
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    cross_val_results = cross_val_score(model, X, y, cv=kf)*100
    print(f'\n{model.__class__.__name__} Cross-Validation Results (Accuracy): {cross_val_results}')
    print(f'\n{model.__class__.__name__} Cross-Validation Mean Accuracy: {cross_val_results.mean()}')

    # Classification report
    print(f"\n{model.__class__.__name__} Classification Report:")
    print(classification_report(y_test, pred, zero_division=1))
    '''
    # graphical representation of confusion matrix of different models
    cf_matrix = confusion_matrix(y_test, pred)
    plt.figure(figsize=(12,8))
    sns.heatmap(cf_matrix, annot=True)
    plt.title(f"{model.__class__.__name__}-Confusion Matrix on Test Data")
    plt.show()
    '''
    return model, Accuracy_train, Accuracy_test

# Instantiate Decision Tree classifier
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

# Now the good models are trained on whole data, combined and tested by using testing data
svm_model = SVC(kernel='linear')
nb_model = GaussianNB()
rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
svm_model.fit(X, y)
nb_model.fit(X, y)
rf_model.fit(X, y)

# Load test data
test_data = pd.read_csv("data/Testing.csv").dropna(axis=1)
 
test_X = test_data.drop('prognosis', axis=1)
test_Y = encoder.fit_transform(test_data["prognosis"])

# Predictions of all three models
svm_pred = svm_model.predict(test_X)
nb_pred = nb_model.predict(test_X)
rf_pred = rf_model.predict(test_X)
 
final_pred = [Counter([x, y, z]).most_common(1)[0][0] for x, y, z in zip(svm_pred, nb_pred, rf_pred)]
final_pred = [int(value) for value in final_pred]
print(f"Accuracy on Test dataset using the combined model: {accuracy_score(test_Y, final_pred)*100}")
 
cf_matrix = confusion_matrix(test_Y, final_pred)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot = True)
plt.title("Final model-Confusion Matrix on Test Dataset")
plt.show()

# Create functions that allow taking symptoms to predict disease
symptoms = X.columns.values

# The dictionary below encodes symptoms into numerical values
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

# Assuming you have a LabelEncoder named encoder
data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}

# Defining the function
def predictDisease(symptoms):
    symptoms = symptoms.split(",")

    # Input data for model
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"].get(symptom)
        if index is not None:
            input_data[index] = 1

    # Check the number of features in the input data
    if len(input_data) != len(data_dict["symptom_index"]):
        raise ValueError(f"Input data has {len(input_data)} features, "
                         f"but the model expects {len(data_dict['symptom_index'])} features.")

    # Convert input data into suitable format
    input_data = np.array(input_data).reshape(1, -1)

    # Generate individual output
    rf_prediction = data_dict["predictions_classes"][rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][svm_model.predict(input_data)[0]]

    # Use Counter to find the most common prediction
    final_prediction = Counter([rf_prediction, nb_prediction, svm_prediction]).most_common(1)[0][0]

    predictions = {
        "rf_model_prediction": rf_prediction,
        "nb_model_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
    }
    return predictions

# Test the function
print(predictDisease("Itching,Skin Rash,Nodal Skin Eruptions"))                                                