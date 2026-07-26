import panel as pn
pn.extension('tabulator', 'vega')
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

# --- Configuration and Model Loading ---
MODELS_DIR = '../models' # Corrected path to point to the parent directory's models folder
MODEL_PATH = os.path.join(MODELS_DIR, 'trained_shc_model.pkl')
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, 'feature_columns.pkl')
FEATURE_IMPORTANCES_PATH = os.path.join(MODELS_DIR, 'feature_importances.pkl')

# Load the trained model, feature columns, and feature importances
try:
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    feature_importances = joblib.load(FEATURE_IMPORTANCES_PATH)
except FileNotFoundError:
    print(f"Error: Model files not found in {MODELS_DIR}. Please run `run_training.py` first.")
    exit()

# Initialize SHAP explainer (using a sample of data to avoid issues with large datasets for background)
# For RandomForest, TreeExplainer is efficient.
# We'll use a dummy data point for the explainer's background if actual training data isn't available.
# A more robust solution would be to save X_train during training.
explainer = shap.TreeExplainer(model)

# Placeholder for a default DataFrame to get mean values for sliders.
# These should ideally come from the descriptive statistics of your training data.
# For now, using hardcoded sensible defaults.
slider_defaults = {
    'ambient_temp_C': {'start': 0.0, 'end': 50.0, 'step': 0.1, 'value': 25.0},
    'coal_surface_temp_C': {'start': 0.0, 'end': 100.0, 'step': 0.1, 'value': 50.0},
    'delta_T_C': {'start': 0.0, 'end': 50.0, 'step': 0.1, 'value': 20.0},
    'delta_T_rate_per_hr': {'start': 0.0, 'end': 10.0, 'step': 0.01, 'value': 1.0},
    'humidity_pct': {'start': 0.0, 'end': 100.0, 'step': 0.1, 'value': 60.0},
    'solar_radiation_Wm2': {'start': 0.0, 'end': 1000.0, 'step': 1.0, 'value': 300.0},
    'wind_speed_kmh': {'start': 0.0, 'end': 30.0, 'step': 0.1, 'value': 10.0},
    'rainfall_last_24h_mm': {'start': 0.0, 'end': 100.0, 'step': 0.1, 'value': 5.0},
    'soil_moisture_pct': {'start': 0.0, 'end': 30.0, 'step': 0.1, 'value': 10.0},
    'coal_surface_moisture_pct': {'start': 0.0, 'end': 20.0, 'step': 0.1, 'value': 5.0},
    'pile_age_days': {'start': 0, 'end': 365, 'step': 1, 'value': 90},
    'coal_type_volatile_pct': {'start': 0.0, 'end': 50.0, 'step': 0.1, 'value': 25.0},
    'crack_width_mm': {'start': 0.0, 'end': 10.0, 'step': 0.01, 'value': 2.0}
}

# --- Widgets for Input Features ---
ambient_temp_C = pn.widgets.FloatSlider(name='Ambient Temp (°C)', **slider_defaults['ambient_temp_C'])
coal_surface_temp_C = pn.widgets.FloatSlider(name='Coal Surface Temp (°C)', **slider_defaults['coal_surface_temp_C'])
delta_T_C = pn.widgets.FloatSlider(name='Delta T (°C)', **slider_defaults['delta_T_C'])
delta_T_rate_per_hr = pn.widgets.FloatSlider(name='Delta T Rate (per hr)', **slider_defaults['delta_T_rate_per_hr'])
humidity_pct = pn.widgets.FloatSlider(name='Humidity (%)', **slider_defaults['humidity_pct'])
solar_radiation_Wm2 = pn.widgets.FloatSlider(name='Solar Radiation (W/m²)', **slider_defaults['solar_radiation_Wm2'])
wind_speed_kmh = pn.widgets.FloatSlider(name='Wind Speed (km/h)', **slider_defaults['wind_speed_kmh'])
rainfall_last_24h_mm = pn.widgets.FloatSlider(name='Rainfall Last 24h (mm)', **slider_defaults['rainfall_last_24h_mm'])
soil_moisture_pct = pn.widgets.FloatSlider(name='Soil Moisture (%)', **slider_defaults['soil_moisture_pct'])
coal_surface_moisture_pct = pn.widgets.FloatSlider(name='Coal Surface Moisture (%)', **slider_defaults['coal_surface_moisture_pct'])
pile_age_days = pn.widgets.IntSlider(name='Pile Age (days)', **slider_defaults['pile_age_days'])
coal_type_volatile_pct = pn.widgets.FloatSlider(name='Coal Type Volatile (%)', **slider_defaults['coal_type_volatile_pct'])
pile_compaction = pn.widgets.RadioButtonGroup(name='Pile Compaction', options={'Compact': 1, 'Loose': 0}, button_type='success')
crack_width_mm = pn.widgets.FloatSlider(name='Crack Width (mm)', **slider_defaults['crack_width_mm'])
hotspot_detected = pn.widgets.Checkbox(name='Hotspot Detected', value=False)

source_synthetic_physics_based = pn.widgets.Checkbox(name='Source: Synthetic Physics Based', value=True)
source_synthetic_model = pn.widgets.Checkbox(name='Source: Synthetic Model', value=False)
source_field_measurement = pn.widgets.Checkbox(name='Source: Field Measurement', value=False)

veg_stress_0 = pn.widgets.Checkbox(name='Veg Stress: 0', value=False)
vg_stress_1 = pn.widgets.Checkbox(name='Veg Stress: 1', value=False)
vg_stress_2 = pn.widgets.Checkbox(name='Veg Stress: 2', value=True)

# Prediction button and output areas
predict_button = pn.widgets.Button(name='Predict SHC Risk', button_type='primary', width=200)
prediction_output = pn.pane.Markdown("", width=400, css_classes=['prediction-output'])
shap_explanation_plot = pn.pane.Matplotlib(None, width=600, height=300, sizing_mode='stretch_width', dpi=100)
shap_summary_text = pn.pane.Markdown("", width=600, sizing_mode='stretch_width')

# --- Global Feature Importance Plot ---
def create_feature_importance_plot():
    if feature_columns is None or feature_importances is None:
        return pn.pane.Markdown("Feature importances not available.")

    # Create a DataFrame for easy plotting
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    fig_importance, ax_importance = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), ax=ax_importance, palette='viridis')
    ax_importance.set_title('Top 15 Global Feature Importances')
    ax_importance.set_xlabel('Relative Importance')
    ax_importance.set_ylabel('')
    plt.tight_layout()
    plt.close(fig_importance) # Close the figure to prevent it from displaying twice
    return pn.pane.Matplotlib(fig_importance, dpi=100) # Wrap with pn.pane.Matplotlib

# --- Prediction Function ---
def make_prediction(event):
    input_data = {
        'ambient_temp_C': ambient_temp_C.value,
        'coal_surface_temp_C': coal_surface_temp_C.value,
        'delta_T_C': delta_T_C.value,
        'delta_T_rate_per_hr': delta_T_rate_per_hr.value,
        'humidity_pct': humidity_pct.value,
        'solar_radiation_Wm2': solar_radiation_Wm2.value,
        'wind_speed_kmh': wind_speed_kmh.value,
        'rainfall_last_24h_mm': rainfall_last_24h_mm.value,
        'soil_moisture_pct': soil_moisture_pct.value,
        'coal_surface_moisture_pct': coal_surface_moisture_pct.value,
        'pile_age_days': pile_age_days.value,
        'coal_type_volatile_pct': coal_type_volatile_pct.value,
        'pile_compaction': pile_compaction.value,
        'crack_width_mm': crack_width_mm.value,
        'hotspot_detected': int(hotspot_detected.value),
        'source_synthetic_physics_based': source_synthetic_physics_based.value,
        'source_synthetic_model': source_synthetic_model.value,
        'source_field_measurement': source_field_measurement.value,
        'veg_stress_0': veg_stress_0.value,
        'veg_stress_1': vg_stress_1.value,
        'veg_stress_2': vg_stress_2.value
    }

    input_series = pd.Series(input_data)
    final_input_df = pd.DataFrame([input_series.reindex(feature_columns, fill_value=0)]).astype(float)

    pred = model.predict(final_input_df)[0]
    prob = model.predict_proba(final_input_df)[0][1] # Probability of SHC Risk = 1

    risk_text = "High Risk (1)" if pred == 1 else "Low Risk (0)"
    risk_color = "red" if pred == 1 else "green"
    risk_html = f"<span style='color: {risk_color};'>{risk_text}</span>"
    result = f"**Predicted SHC Risk:** {risk_html}<br>"
    result += f"**Probability of High Risk:** {prob:.2f}"
    prediction_output.object = result

    # --- SHAP Explanation ---
    raw_shap_values = explainer.shap_values(final_input_df)

    # Handle cases where shap_values might be a list of arrays (standard) or a single 3D array (some versions/models)
    if isinstance(raw_shap_values, list):
        # For binary classification, raw_shap_values is a list like [shap_values_class0, shap_values_class1]
        # Each element is an array of shape (num_samples, num_features)
        shap_values_for_display = raw_shap_values[1][0] # Get positive class (index 1), first sample (index 0)
        base_value_for_display = explainer.expected_value[1]
    elif isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 3:
        # Some SHAP versions/explainers return a 3D array (num_samples, num_features, num_classes)
        # We need the SHAP values for the first sample (index 0) and the positive class (index 1)
        shap_values_for_display = raw_shap_values[0, :, 1]
        base_value_for_display = explainer.expected_value[1]
    else:
        shap_explanation_plot.object = None # Clear the plot if format is unexpected
        shap_summary_text.object = pn.pane.Markdown("Error: SHAP values returned in an unexpected format. Cannot generate explanation.")
        return

    # Create a SHAP Explanation object
    explanation = shap.Explanation(values=shap_values_for_display,
                                  base_values=base_value_for_display,
                                  data=final_input_df.iloc[0],
                                  feature_names=feature_columns)

    # Create a SHAP waterfall plot
    fig_shap = plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation, show=False) # show=False prevents immediate display, allowing Panel to handle it
    plt.tight_layout()
    # Removed plt.close(fig_shap) to ensure the figure is available for Panel to render
    shap_explanation_plot.object = fig_shap # Assign the matplotlib figure directly

    # Textual summary of SHAP values
    explanation_text = "**Key factors influencing this prediction:**\n\n"
    shap_df = pd.DataFrame({
        'Feature': feature_columns,
        'SHAP_Value': shap_values_for_display
    }).sort_values(by='SHAP_Value', ascending=False).set_index('Feature')

    # Top features pushing towards '1' (High Risk)
    positive_contributors = shap_df[shap_df['SHAP_Value'] > 0].head(3)
    if not positive_contributors.empty:
        explanation_text += "**Pushing towards High Risk:**\n"
        for feature, row in positive_contributors.iterrows():
            explanation_text += f"- **{feature}**: {row['SHAP_Value']:.2f} (positive influence)\n"
    
    # Top features pushing towards '0' (Low Risk)
    negative_contributors = shap_df[shap_df['SHAP_Value'] < 0].tail(3).sort_values(by='SHAP_Value', ascending=True)
    if not negative_contributors.empty:
        explanation_text += "\n**Pushing towards Low Risk:**\n"
        for feature, row in negative_contributors.iterrows():
            explanation_text += f"- **{feature}**: {row['SHAP_Value']:.2f} (negative influence)\n"
            
    if positive_contributors.empty and negative_contributors.empty:
        explanation_text += "*No strong individual feature influences detected for this prediction, or all influences are minor.*"

    shap_summary_text.object = explanation_text


predict_button.on_click(make_prediction)

# --- Dashboard Layout using FastListTemplate ---
template = pn.template.FastListTemplate(
    title='SHC Risk Prediction Dashboard with XAI',
    sidebar=[pn.Column(
        "## Input Parameters",
        ambient_temp_C,
        coal_surface_temp_C,
        delta_T_C,
        delta_T_rate_per_hr,
        humidity_pct,
        solar_radiation_Wm2,
        wind_speed_kmh,
        rainfall_last_24h_mm,
        soil_moisture_pct,
        coal_surface_moisture_pct,
        pile_age_days,
        coal_type_volatile_pct,
        pile_compaction,
        crack_width_mm,
        hotspot_detected,
        pn.Row(source_synthetic_physics_based, source_synthetic_model, source_field_measurement, sizing_mode='stretch_width', align='start'),
        pn.Row(veg_stress_0, vg_stress_1, vg_stress_2, sizing_mode='stretch_width', align='start'),
        pn.layout.Divider(),
        predict_button,
        pn.layout.Divider(),
        "## Prediction Result",
        prediction_output
    )],
    main=[
        pn.Column(
            "## Global Feature Importance",
            pn.pane.Markdown("This chart shows the overall importance of each feature to the model's predictions, calculated across the entire dataset."),
            create_feature_importance_plot()
        ),
        pn.Column(
            "## Local Prediction Explanation (SHAP Values)",
            pn.pane.Markdown("This section provides insights into *why* the model made a specific prediction for the currently entered input values. The plot below illustrates how each feature contributes to pushing the prediction towards High Risk (red) or Low Risk (blue). Below the plot, a summary highlights the most influential features."),
            shap_explanation_plot,
            shap_summary_text
        )
    ]
)

template.servable()
