import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor,GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# Read data from .data file
columns = ['ID_Number','Diagnosis','Radius','Texture','Perimeter','Area','Smoothness','Compactness','Concavity','Concave_points','Symmetry','Fractal_Dimension','Radius_SE','Texture_SE','Perimeter_SE','Area_SE','Smoothness_SE','Compactness_SE','Concavity_SE','Concave_points_SE','Symmetry_SE','Fractal_Dimension_SE','Radius_Worst','Texture_Worst','Perimeter_Worst','Area_Worst','Smoothness_Worst','Compactness_Worst','Concavity_Worst','Concave_points_Worst','Symmetry_Worst','Fractal_Dimension_Worst']
df = pd.read_csv('Data/wdbc.data', names=columns)
df['Diagnosis'] = df['Diagnosis'].map({'B':0,'M':1})
print(df.head())

y = df['Diagnosis']
X = df.drop('Diagnosis', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scores = {}

# List of classifiers
classifiers = [
    (RandomForestClassifier(), "RandomForestClassifier score"),
    (RandomForestRegressor(), "RandomForestRegressor score"),
    (SVC(), "SVC score"),
    (KNeighborsClassifier(), "KNeighborsClassifier score"),
    (GradientBoostingClassifier(), "GradientBoostingClassifier score"),
    (MLPClassifier(), "MLPClassifier score"),
    (LogisticRegression(), "LogisticRegression score"),
    (AdaBoostClassifier(), "AdaBoostClassifier score"),
    (ExtraTreesClassifier(), "ExtraTreesClassifier score"),
    (BaggingClassifier(), "BaggingClassifier score")
]


# Train and score each classifier
for clf, clf_name in classifiers:
    clf.fit(X_train, y_train)
    scores[clf_name] = clf.score(X_test, y_test)

# Print scores in descending order
sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
for key, value in sorted_scores:
    print(f'{key}: {value}')


# Hyperparameter tuning

param_distributions = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
param_distributions_LR = {
    'C': np.logspace(-4, 4, 20),
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

# Hyperparameter tuning
'''
random_search1 = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_distributions, cv=5, scoring='accuracy', n_iter=10)
random_search1.fit(X_train, y_train)
y_pred1 = random_search1.predict(X_test)
print(f'Best parameters for RandomForestClassifier: {random_search1.best_params_}')
print(f'Best score for RandomForestClassifier: {random_search1.best_score_}')

random_search2 = RandomizedSearchCV(ExtraTreesClassifier(), param_distributions=param_distributions, cv=5, scoring='accuracy', n_iter=10)
random_search2.fit(X_train, y_train)
y_pred2 = random_search2.predict(X_test)
print(f'Best parameters for ExtraTreesClassifier: {random_search2.best_params_}')
print(f'Best score for ExtraTreesClassifier: {random_search2.best_score_}')


# checking all scores, but we want to avoid false negatives as much as possible, so we will focus on recall
accuracys = [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]

print("RandomSearchCV RandomForestClassifier")
for accuracy in accuracys:
    print(accuracy.__name__, accuracy(y_test, y_pred1))
    
print("RandomSearchCV ExtraTreesClassifier")
for accuracy in accuracys:
    print(accuracy.__name__, accuracy(y_test, y_pred2))

   
# Confusion matrix
print("RandomSearchCV RandomForestClassifier")
print(confusion_matrix(y_test, y_pred1))
print("RandomSearchCV ExtraTreesClassifier")
print(confusion_matrix(y_test, y_pred2))
'''

'''
# Bagging with RandomForestClassifier
bag_clf = BaggingClassifier(
    RandomForestClassifier(), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1
)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
print("BaggingClassifier score: ", bag_clf.score(X_test, y_test))

# Boosting with AdaBoost
ada_clf = AdaBoostClassifier(
    RandomForestClassifier(), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5
)
ada_clf.fit(X_train, y_train)
y_pred = ada_clf.predict(X_test)
print("AdaBoostClassifier score: ", ada_clf.score(X_test, y_test))

'''