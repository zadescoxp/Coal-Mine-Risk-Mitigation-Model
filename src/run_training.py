# In a script like run_training.py or directly in an interpreter
from data_preprocessing import load_and_preprocess_data
from model_training import train_and_save_model

# Assuming your CSV is in a 'data' folder relative to where you run this script (e.g., if run_training.py is in 'src/')
csv_file_path = '../data/coal_shc_dataset_synthetic_v1_1.csv'
processed_df = load_and_preprocess_data(csv_file_path)

# The model, feature columns, and feature importances will be saved in the 'models/' directory.
model, X_test, y_test = train_and_save_model(processed_df,
                                              model_path='../models/trained_shc_model.pkl',
                                              feature_columns_path='../models/feature_columns.pkl',
                                              feature_importances_path='../models/feature_importances.pkl')

print("\nTraining complete and model artifacts saved to '../models/'.")
