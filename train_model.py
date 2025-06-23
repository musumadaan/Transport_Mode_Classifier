import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from feature_utils import extract_features
import joblib

# Load features CSV
def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna()  # drop NA rows
    if 'label' not in df.columns:
        raise ValueError("Input CSV must contain a 'label' column with transport mode classes.")
    return df

# Train/test split and model training
def train_and_evaluate(df):
    X = df.drop(columns=['label', 'window', 'source_id'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Confusion matrix result
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    return model

# Main function to run training pipeline
def main():
    features_path = 'features_output.csv'
    try:
        df = load_data(features_path)
        model = train_and_evaluate(df)
        joblib.dump(model, 'model_transport_mode.pkl') #save as the name.
        print("\nModel done and saved to 'model_transport_mode.pkl'.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()