import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib # for saving and loading the model
import os # For creating directories

def train_and_save_model(df, model_path='models/trained_shc_model.pkl', feature_columns_path='models/feature_columns.pkl', feature_importances_path='models/feature_importances.pkl'):
    """Trains a RandomForestClassifier and saves the model, feature columns, and feature importances."""
    # Define features (X) and target (y)
    X = df.drop('shc_risk', axis=1)
    y = df['shc_risk']

    # Split data into training (70%) and testing (30%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print(f"Original training set target distribution:\n{y_train.value_counts()}")

    # Apply SMOTE to the training data to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"\nResampled training set target distribution:\n{y_train_resampled.value_counts()}")

    # Initialize and train a RandomForestClassifier model with the resampled data and class weights
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_resampled, y_train_resampled)

    # Make predictions on the original (unresampled) test set
    y_pred = model.predict(X_test)

    # Evaluate the model and print statistics
    print("\nModel Evaluation:")
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Create the models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save the trained model
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

    # Save the feature columns for consistency during prediction
    joblib.dump(X.columns.tolist(), feature_columns_path)
    print(f"Feature columns saved to {feature_columns_path}")

    # Save feature importances
    joblib.dump(model.feature_importances_, feature_importances_path)
    print(f"Feature importances saved to {feature_importances_path}")

    return model, X_test, y_test
