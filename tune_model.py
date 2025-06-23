# Hyperparameter tuning for a Random Forest classifier
# to predict mode (car, bus, train) using features_output.csv. 
# Use GridSearchCV to evaluate multiple model configurations 
# via 5-fold cross-validation, then select the best model.

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

#load in data
features_path = 'features_output.csv'
df = pd.read_csv(features_path)
df = df.dropna()

X = df.drop(columns=['label', 'window', 'source_id'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
#param grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

#grid search for the classifier
grid_search = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# save best model to the same name
joblib.dump(grid_search.best_estimator_, 'model_transport_mode.pkl')
print("\nBest model saved to 'model_transport_mode.pkl'")

y_pred = grid_search.best_estimator_.predict(X_test)

#prints
print("\nBest Parameters:")
print(grid_search.best_params_)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# conf mat
cm = confusion_matrix(y_test, y_pred, labels=grid_search.best_estimator_.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=grid_search.best_estimator_.classes_,
            yticklabels=grid_search.best_estimator_.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Tuned Random Forest)')
plt.tight_layout()
plt.show()
