import pandas as pd
import numpy as np

def synthetically_balance_shc_risk(df, positive_percentage=0.01):
    """Synthetically introduces a small percentage of '1.0' values to 'shc_risk' for demonstration."""
    num_positive_samples = int(len(df) * positive_percentage)
    if num_positive_samples == 0 and len(df) > 0:
        num_positive_samples = 1

    zero_shc_indices = df[df['shc_risk'] == 0.0].index

    if len(zero_shc_indices) > num_positive_samples:
        indices_to_change = np.random.choice(zero_shc_indices, num_positive_samples, replace=False)
        df.loc[indices_to_change, 'shc_risk'] = 1.0
    else:
        # Fallback if not enough 0.0 values, set all to 1.0 for demonstration purposes
        df.loc[zero_shc_indices, 'shc_risk'] = 1.0
    
    print("Value counts after synthetic modification:")
    print(df['shc_risk'].value_counts())
    
    return df

def load_and_preprocess_data(csv_path):
    """Loads data, performs preprocessing steps, and returns the processed DataFrame."""
    df = pd.read_csv(csv_path)

    # Drop 'sample_id' column
    df = df.drop('sample_id', axis=1)

    # Handle missing values by dropping rows with any NaN values
    df = df.dropna()

    # Perform one-hot encoding on 'data_source' and 'vegetation_stress_index'
    df = pd.get_dummies(df, columns=['data_source', 'vegetation_stress_index'], prefix=['source', 'veg_stress'])

    # Ensure 'shc_risk' is integer type for classification
    df['shc_risk'] = df['shc_risk'].astype(int)

    # Apply synthetic balancing for demonstration purposes
    df = synthetically_balance_shc_risk(df)

    return df
