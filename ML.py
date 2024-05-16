import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor,GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from joblib import dump, load

# 1. Data Collection
columns = ['ID_Number','Diagnosis','Radius','Texture','Perimeter','Area','Smoothness','Compactness','Concavity','Concave_points','Symmetry','Fractal_Dimension','Radius_SE','Texture_SE','Perimeter_SE','Area_SE','Smoothness_SE','Compactness_SE','Concavity_SE','Concave_points_SE','Symmetry_SE','Fractal_Dimension_SE','Radius_Worst','Texture_Worst','Perimeter_Worst','Area_Worst','Smoothness_Worst','Compactness_Worst','Concavity_Worst','Concave_points_Worst','Symmetry_Worst','Fractal_Dimension_Worst']
df = pd.read_csv('Data/wdbc.data', names=columns)
print(df.head())

# 2. Exploratory Data Analysis
class_distribution = df['Diagnosis'].value_counts(normalize=True)
print(class_distribution)
# check the data types of the columns and check for missing values
df.info()
df.describe()
df.isnull().sum()

# 2. Data Cleaning

# I need to remove the ID_Number column, as it is not useful for the model.
df.drop('ID_Number', axis=1, inplace=True)

# change the Diagnosis column from 'B' and 'M' to 0 and 1
df['Diagnosis'] = df['Diagnosis'].map({'B':0,'M':1})

# check for outliers in the data set.
from scipy.stats import zscore
z_scores = zscore(df)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
num_outliers = (~filtered_entries).sum()
print(f"Number of outliers: {num_outliers}")
df_filtered = df[filtered_entries]
# I don't know if these outliers are crucial to detecting cancer, so I will keep them in for now, and run tests later. ***** DONT FORGET

# The dataset is imbalanced, so I will use SMOTE to balance the dataset.
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(sampling_strategy='minority')
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 2. Visualising the data
selected_columns = ['Diagnosis','Radius','Texture','Perimeter','Area','Smoothness','Compactness','Concavity','Concave_points','Symmetry','Fractal_Dimension','Radius_SE','Texture_SE','Perimeter_SE','Area_SE','Smoothness_SE','Compactness_SE','Concavity_SE','Concave_points_SE','Symmetry_SE','Fractal_Dimension_SE','Radius_Worst','Texture_Worst','Perimeter_Worst','Area_Worst','Smoothness_Worst','Compactness_Worst','Concavity_Worst','Concave_points_Worst','Symmetry_Worst','Fractal_Dimension_Worst']

# correlation matrix to see how the features are correlated with each other
corr_matrix = df[selected_columns].corr()

plt.figure(figsize=(18, 18))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", square=True, cmap='coolwarm')
plt.show()

# Feature importance

# Fit a random forest classifier
clf = RandomForestClassifier()
clf.fit(X_train_smote,y_train_smote)

# Get feature importances
importances = clf.feature_importances_
importance_series = pd.Series(importances, index=X.columns)
# Sort the series by importance
sorted_importances = importance_series.sort_values(ascending=False)

# Create a bar plot
plt.figure(figsize=(8,4))
sorted_importances.plot(kind='bar', color='blue')
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.show()

cols_to_drop = set()
corr_matrix = df.corr()

# Iterate over the correlation matrix and drop the feature with the lowest importance score in each pair of highly correlated features
for i in range(corr_matrix.shape[0]):
    for j in range(i+1, corr_matrix.shape[1]):
        if abs(corr_matrix.iloc[i, j]) > 0.9:
            print(f"Feature pair ({corr_matrix.columns[i]}, {corr_matrix.columns[j]}) has a correlation score of {corr_matrix.iloc[i, j]}")
            # If neither of the features in the pair have been removed before
            if corr_matrix.columns[i] not in cols_to_drop and corr_matrix.columns[j] not in cols_to_drop:
                # Add the feature with the lowest importance score to the set of features to remove
                if importance_series[corr_matrix.columns[i]] > importance_series[corr_matrix.columns[j]]:
                    cols_to_drop.add(corr_matrix.columns[j])
                else:
                    cols_to_drop.add(corr_matrix.columns[i])



# Drop the columns
# I am not going to drop the columns yet, I will run tests with and without the columns to see the difference in.
#df = df.drop(cols_to_drop, axis=1)
print(f"Columns to be dropped: {cols_to_drop}")

# Preliminary Model Training - Testing and scoring a selection of classifiers before feature engineering so I can compare results after.


scores = {}

# List of classifiers
classifiers = [
    (RandomForestClassifier(), "Random Forest Classifier"),
    (SVC(probability=True), "SVC"),
    (KNeighborsClassifier(), "KNeighbors Classifier"),
    (GradientBoostingClassifier(), "Gradient Boosting Classifier"),
    (MLPClassifier(max_iter=1000), "MLP Classifier"),
    (LogisticRegression(max_iter=10000), "Logistic Regression"),
    (AdaBoostClassifier(algorithm="SAMME"), "AdaBoost Classifier"),
    (ExtraTreesClassifier(), "Extra Trees Classifier"),
    (BaggingClassifier(), "Bagging Classifier")
]


# Train and score each classifier
for clf, clf_name in classifiers:
    clf.fit(X_train_smote, y_train_smote)
    scores[clf_name] = clf.score(X_test, y_test)

# Print scores in descending order
sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
for key, value in sorted_scores:
    print(f'{key} score: {value}')
    
    
# List of classifiers with balanced class weights
scores_balanced_classifiers = {}
classifiers_balanced = [
    (RandomForestClassifier(class_weight='balanced'), "Random Forest Classifier"),
    (SVC(class_weight='balanced'), "SVC"),
    (ExtraTreesClassifier(class_weight='balanced'), "Extra Trees Classifier"),
]

# Train and score each classifier
for clf, clf_name in classifiers_balanced:
    clf.fit(X_train_smote, y_train_smote)
    scores_balanced_classifiers[clf_name] = clf.score(X_test, y_test)
    
# Print scores in descending order
sorted_scores = sorted(scores_balanced_classifiers.items(), key=lambda x: x[1], reverse=True)
for key, value in sorted_scores:
    print(f'{key} score after class weight balanced: {value}. A difference of {value - scores[key]}')
    
#Printing scores after the removal of the columns
X_cols_dropped = df.drop(list(cols_to_drop) + ['Diagnosis'], axis=1)
print(X_cols_dropped.columns)

X_train_resampled, y_train_resampled = smote.fit_resample(X_cols_dropped, y)
scores_after_columns_removed = {}

# Train and score each classifier
for clf, clf_name in classifiers:
    clf.fit(X_train_smote, y_train_smote)
    scores_after_columns_removed[clf_name] = clf.score(X_test, y_test)

# Print scores in descending order
sorted_scores = sorted(scores_after_columns_removed.items(), key=lambda x: x[1], reverse=True)
for key, value in sorted_scores:
    print(f'{key} score after columns removed: {value}. A difference of {value - scores[key]}')
    
# print average difference of scores
average_difference = np.mean([value - scores[key] for key, value in sorted_scores])
print(f'\nAverage difference in scores after removing columns: {average_difference}')

from sklearn.model_selection import cross_val_score

# Dictionary to store cross-validation scores
cv_scores = {}

# Perform cross-validation for each classifier
for clf, clf_name in classifiers:
    scores = cross_val_score(clf, X, y, cv=5)
    cv_scores[clf_name] = scores.mean()

# Print cross-validation scores in descending order
sorted_scores = sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)
for key, value in sorted_scores:
    print(f'{key} cross-validation score: {value}')
    
# 4. Feature Engineering

# Scaling my numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_scaled_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train-test split on the resampled data
X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(X_scaled_resampled, y_resampled, test_size=0.2, random_state=42)

# Fit the model on the resampled training data
rfc_resampled = RandomForestClassifier()
rfc_resampled.fit(X_train_resampled, y_train_resampled)


etc_score = cross_val_score(rfc_resampled, X_scaled, y, cv=5)
#print(f"ExtraTreesClassifier score before removing features: {etc_score.mean()}")

# visualizing feature importance
importances = rfc_resampled.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(8, 3))
plt.bar(range(X_train_smote.shape[1]), importances[indices])
plt.xticks(range(X_train_smote.shape[1]), X_train_smote.columns[indices], rotation=90)
plt.show()

# Using pipeline and SelectKBest to find the optimal amount of usable features.

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import KFold

'''# shortened list of classifiers for quicker testing
classifiers = [
    (SVC(probability=True), "SVC"),
    (KNeighborsClassifier(), "KNeighbors Classifier"),
    (ExtraTreesClassifier(), "Extra Trees Classifier")
]
'''


plt.figure(figsize=(8, 5))

selected_features_dict = {}

for clf, clf_type in classifiers:
    features_scores = []
    for i in range(1,len(X.columns)+1):
        pipeline = Pipeline([
            ('selector', SelectKBest(score_func=f_classif, k=i)),
            ('classifier', clf)
        ])
        scores = cross_val_score(pipeline, X_scaled, y, cv=KFold(n_splits=10))
        mean_score = scores.mean()
        features_scores.append((i, mean_score))

    plt.plot([i[0] for i in features_scores], [i[1] for i in features_scores], label=clf_type)
    
    max_score = max(features_scores, key=lambda x: x[1])
    print(f'{clf_type} - Max score: {max_score[1]} with {max_score[0]} features')
    
    # Store the optimal number of features for this classifier in a dictionary
    selector = SelectKBest(score_func=f_classif, k=max_score[0])
    X_selected = selector.fit_transform(X_scaled, y)
    selected_features = X.columns[selector.get_support()]
    selected_features_dict[clf] = selected_features


plt.xlabel('Number of Features')
plt.ylabel('Score')
plt.title('Number of Features vs Score')
plt.legend()
plt.show()

# 6. Model Training - Testing and scoring a selection of classifiers after feature engineering 

scores = {}

for clf_name, features in selected_features_dict.items():
    
    X_selected = X_scaled[features]
    X_train_smote, y_train_smote = smote.fit_resample(X_selected, y)
    X_test_selected = X_test_resampled[features]
    
    clf_name.fit(X_train_smote, y_train_smote)   
    scores[clf_name] = clf_name.score(X_test_selected, y_test_resampled)
    
    print(f'{clf_name} - Score: {scores[clf_name]}')
    print(classification_report(y_test_resampled, clf_name.predict(X_test_selected)))
    
# 8. Hyperparameter Tuning - GridSearchCV on all the classifiers

param_grids = {
    'SVC': {
        'C': [0.1, 1, 10, 100], 
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'poly', 'sigmoid']
    },
    'KNeighbors Classifier': {
        'n_neighbors': list(range(1, 31)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    },
    'Extra Trees Classifier': {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
    'Random Forest Classifier': {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
    'Gradient Boosting Classifier': {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.1, 0.05, 0.02, 0.01],
        'max_depth': [4, 6, 8],
        'min_samples_leaf': [20, 50,100,150],
    },
    'MLP Classifier': {
        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],
    },
    'Logistic Regression': {
        'C': np.logspace(-4, 4, 20),
        'solver': ['liblinear'],
    },
    'AdaBoost Classifier': {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate' : [0.01,0.05,0.1,0.3,1],
    },
    'Bagging Classifier': {
        'n_estimators': [10, 20, 30, 40, 50],
        'max_samples': [0.5, 1.0],
        'max_features': [0.5, 1.0],
    }  
}


# Initialize a DataFrame to store the performance metrics
metrics_df = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score'])

for clf, clf_name in classifiers:
    X_selected = X_scaled[selected_features_dict[clf]]
    
    # Apply SMOTE to the selected features
    X_resampled, y_resampled = smote.fit_resample(X_selected, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    # Get the parameter grid for this classifier
    param_grid = param_grids[clf_name]

    # Initialize the GridSearchCV object
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)
 
    print(f'Best parameters for {clf_name} using GridSearchCV: {grid_search.best_params_}')
    print(f'Best score for {clf_name} using GridSearchCV: {grid_search.best_score_}')
    print(f"ROC AUC score for {clf_name}: {roc_auc_score(y_test, y_pred)}")
    print(f"Classification report for {clf_name}:\n{classification_report(y_test, y_pred)}")  
    
    # 9. Using the best parameters for the top 3 classifiers to train and test the models. (It was easier to do this in the same loop)
    clf.set_params(**grid_search.best_params_)
    clf.fit(X_train, y_train)
    
    # Save the model
    dump(clf, f"{clf_name}_model.joblib")
    
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]
    
    # Compute the performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_score)
    
    # Store the performance metrics in the DataFrame
    metrics_df.loc[clf_name] = [accuracy, precision, recall, f1, roc_auc]
    
    # Compute the ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_score)
    
    # Plot the ROC curve
    plt.plot(fpr, tpr, label=f'{clf_name} (AUC = {roc_auc:.2f})')

# Add labels and legend to the plot
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')

# Display the plot
plt.show()


# Display the performance metrics
print(metrics_df,'\n')

# Find the model with the highest score
best_model = metrics_df.idxmax()

# Print the best model for each metric and its score
for metric in metrics_df.columns:
    print(f"The best model for {metric} is {best_model[metric]} with a score of {metrics_df.loc[best_model[metric], metric]}")


# I will now compare the classifiers original scores vs their scores after hyperparameter tuning.
for key, value in sorted_scores:
    print(f'{key} score increase of: {metrics_df.loc[key, "Accuracy"] - value}')
    
# load the models back in
model1 = load('SVC_model.joblib')
model2 = load('KNeighbors Classifier_model.joblib')
model3 = load('Extra Trees Classifier_model.joblib')

X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


# Ensemble model - Voting Classifier
ensemble = VotingClassifier(estimators=[('svc', model1), ('knn', model2), ('etc', model3)], voting='hard')
ensemble.fit(X_train, y_train)
predictions1 = ensemble.predict(X_test)
roc_auc1 = roc_auc_score(y_test, predictions1)
print(f"Ensemble model accuracy: {accuracy_score(y_test, predictions1)}")
print(f"Ensemble model ROC AUC: {roc_auc}")

# Stacking Classifier
stacking_model = StackingClassifier(estimators=[('knn', model2), ('etc', model3)],final_estimator=model1)
stacking_model.fit(X_train, y_train)
predictions2 = stacking_model.predict(X_test)
roc_auc2 = roc_auc_score(y_test, predictions2)
print(f"Stacking model accuracy: {accuracy_score(y_test, predictions2)}")
print(f"Stacking model ROC AUC: {roc_auc2}")