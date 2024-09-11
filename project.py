import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
def load_data():
    data = pd.read_csv('file testing data.csv')
    return data

# Main function for the streamlit app
def main():
    st.title("Electric Load Prediction")

    # Load dataset
    data = load_data()
    data['Date Hour'] = pd.to_datetime(data['Date Hour'], format='%m/%d/%Y %H:%M')
    data['Hour'] = data['Date Hour'].dt.hour
    data['Day'] = data['Date Hour'].dt.day
    data['Month'] = data['Date Hour'].dt.month
    data['Year'] = data['Date Hour'].dt.year
    # Modify Day of Week: Shift to start from Sunday
    data['Day of Week'] = (data['Date Hour'].dt.dayofweek + 1) % 7
    data['Is Weekend'] = data['Day of Week'].apply(lambda x: 1 if x >= 6 else 0)  # Weekend is Saturday(6) and Sunday(0)

    # Feature Engineering
    features = ['temp', 'Hour', 'Day', 'Month', 'Day of Week', 'Is Weekend']
    X = data[features]
    y = data['Power Usage']

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model selection dropdown
    model_choice = st.selectbox("Select Model", ["RandomForest", "GradientBoosting", "XGBoost", "LightGBM"])

    if model_choice == "RandomForest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_choice == "GradientBoosting":
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_choice == "XGBoost":
        model = XGBRegressor(n_estimators=100, random_state=42)
    elif model_choice == "LightGBM":
        model = LGBMRegressor(n_estimators=100, random_state=42)

    # Train the selected model
    model.fit(X_train, y_train)

    # Test model performance
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Display model performance
    st.write(f"### Model Performance ({model_choice})")
    st.write(f"R-squared Score: {r2:.2f}")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")

    # Allow user input for prediction
    st.write("### Make a Prediction")
    temp_input = st.number_input("Enter Temperature (°C)", min_value=-30.0, max_value=50.0, value=25.0)
    hour_input = st.slider("Select Hour of Day", min_value=0, max_value=23, value=12)
    day_input = st.slider("Select Day of the Month", min_value=1, max_value=31, value=15)
    month_input = st.slider("Select Month", min_value=1, max_value=12, value=6)
    day_of_week_input = st.slider("Select Day of the Week (0=Sunday, 6=Saturday)", min_value=0, max_value=6, value=0)
    is_weekend_input = 1 if day_of_week_input >= 6 else 0

    # Scale user input
    user_input = scaler.transform([[temp_input, hour_input, day_input, month_input, day_of_week_input, is_weekend_input]])
    prediction = model.predict(user_input)
    st.write(f"Predicted Power Usage: {prediction[0]:.2f}")

if __name__ == '__main__':
    main()
