import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import History
import os
import joblib
import io
from datetime import timedelta

st.set_page_config(page_title="Time-Series Prediction Project", layout="wide")
st.markdown("""
    <style>
        .reportview-container { margin-top: -2em; }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
        @import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300;1,400;1,500;1,600;1,700&family=Parkinsans:wght@300..800&family=Yuji+Mai&display=swap');
        .center-header {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 20vh;
            text-align: center;
            font-size: 2em;
            font-weight: bold;
            color: blue;
            font-family: 'normal';
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="center-header">Time-Series Prediction Project</div>', unsafe_allow_html=True)
st.write('<p style="font-size:120%;color:orange;font-weight: bold">Team Members:</p>', unsafe_allow_html=True)
st.write('<p style="font-size:100%;font-weight: bold;color:#b233ff">1. Khaled Abd Elhanan Saad</p>', unsafe_allow_html=True)
st.write('<p style="font-size:100%;font-weight: bold;color:#ff6633">2. Mohammed Adel Emarah</p>', unsafe_allow_html=True)
st.write('<p style="font-size:100%;font-weight: bold;color:#33ff9c">3. Omar Ahmed Mohemed Aytta</p>', unsafe_allow_html=True)
st.write('<p style="font-size:100%;font-weight: bold;color:#33ffdc">4. Mohamed Momen Zidann</p>', unsafe_allow_html=True)
st.write('<p style="font-size:100%;font-weight: bold;color:#33ff41">5. Omar Samy Marey</p>', unsafe_allow_html=True)

st.markdown("---")
st.write('<p style="font-size:130%;color:orange;font-weight: bold">Project Overview</p>', unsafe_allow_html=True)
st.write('<p style="color:yellow; font-weight:bold;">This project uses neural networks (LSTM or MLP) to predict time-series data using regression (predicting exact values).</p>', unsafe_allow_html=True)

st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose section:", ["Info", "Preprocessing", "Train Model", "Prediction Plot", "Loss Curve", "Model Accuracy", "Future Prediction"])

SEQUENCE_LENGTH = 20
MODEL_PATH = "model.keras"
PREDICTION_OUTPUT = "prediction_results.csv"
HISTORY_PATH = "training_history.pkl"
ACCURACY_PATH = "model_accuracy.pkl"

def convert_series_to_matrix(vector_series, sequence_length, is_lstm=True):
    if is_lstm:
        matrix = []
        for i in range(len(vector_series) - sequence_length + 1):
            matrix.append(vector_series[i:i + sequence_length])
        return np.array(matrix)
    else:
        matrix = []
        for i in range(len(vector_series) - sequence_length + 1):
            slice_data = vector_series[i:i + sequence_length].to_numpy().flatten()
            matrix.append(slice_data)
        return np.array(matrix)

def preprocess_data(df, target_column, model_type="LSTM"):
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_column not in numerical_cols:
        raise ValueError(f"Target column '{target_column}' must be a numerical column.")

    if model_type == "LSTM":
        try:
            data = df[target_column].astype(float).values
        except ValueError:
            raise ValueError(f"Column '{target_column}' must contain numerical data.")

        if len(data) < SEQUENCE_LENGTH:
            raise ValueError(f"Dataset has {len(data)} rows, but at least {SEQUENCE_LENGTH} rows are required.")

        matrix_data = convert_series_to_matrix(data, SEQUENCE_LENGTH, is_lstm=True)
        shifted_value = matrix_data.mean()
        matrix_data -= shifted_value

        train_row = int(0.9 * matrix_data.shape[0])
        train_set = matrix_data[:train_row, :]
        np.random.shuffle(train_set)

        X_train = train_set[:, :-1]
        y_train = train_set[:, -1]
        X_test = matrix_data[train_row:, :-1]
        y_test = matrix_data[train_row:, -1]

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        return X_train, y_train, X_test, y_test, shifted_value

    else:  # MLP
        data = df[numerical_cols].astype(float)
        if len(data) < SEQUENCE_LENGTH:
            raise ValueError(f"Dataset has {len(data)} rows, but at least {SEQUENCE_LENGTH} rows are required.")

        matrix_data = convert_series_to_matrix(data, SEQUENCE_LENGTH, is_lstm=False)
        shifted_values = matrix_data.mean(axis=0)
        matrix_data -= shifted_values

        train_row = int(0.9 * matrix_data.shape[0])
        train_set = matrix_data[:train_row, :]
        np.random.shuffle(train_set)

        features_per_timestep = len(numerical_cols)
        target_idx_in_flattened = (SEQUENCE_LENGTH - 1) * features_per_timestep + numerical_cols.index(target_column)

        X_train = train_set[:, :-features_per_timestep]
        y_train = train_set[:, target_idx_in_flattened]
        X_test = matrix_data[train_row:, :-features_per_timestep]
        y_test = matrix_data[train_row:, target_idx_in_flattened]

        return X_train, y_train, X_test, y_test, shifted_values[numerical_cols.index(target_column)]

def build_model(input_shape, model_type="LSTM"):
    model = Sequential()
    if model_type == "LSTM":
        model.add(LSTM(50, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dropout(0.2))
        
    elif model_type == "MLP":
        model.add(Dense(64, input_shape=input_shape, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
    
    model.add(Dense(1, activation='linear'))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    return model

def calculate_mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    epsilon = np.finfo(float).eps
    mape = np.mean(np.abs((actual - predicted) / (actual + epsilon))) * 100
    return mape

uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df)
    st.write("Shape:", df.shape)

    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numerical_columns:
        st.error("No numerical columns found in the dataset. Please upload a dataset with at least one numerical column.")
    else:
        st.write('<p style="font-size:120%;color:orange;font-weight: bold">Select Target Column</p>', unsafe_allow_html=True)
        target_column = st.selectbox(
            "Choose the column to predict (only numerical columns are shown):",
            options=["-- Select a column --"] + numerical_columns,
            index=0
        )

        if target_column == "-- Select a column --":
            st.warning("Please select a numerical column to proceed.")

        else:
            st.success(f"Selected column for prediction: **{target_column}**")

            st.write('<p style="font-size:120%;color:orange;font-weight: bold">Select Model Type</p>', unsafe_allow_html=True)
            model_type = st.selectbox(
                "Choose the model type:",
                options=["LSTM", "MLP"],
                index=0
            )

            if option == "Info":
                st.subheader("Dataset Info")
                st.write("Summary:")
                st.write(df.describe())
                buffer = io.StringIO()
                df.info(buf=buffer)
                s = buffer.getvalue()
                st.text(s)

            elif option == "Preprocessing":
                try:
                    df_cleaned = df.dropna(how='any').dropna(how='any', axis=1)
                    st.success("Data preprocessed successfully: Removed rows and columns with any missing values.")
                    st.write("Dataset after removing rows and columns with any missing values:")
                    st.dataframe(df_cleaned)
                    st.write("Shape after cleaning:", df_cleaned.shape)

                    X_train, y_train, X_test, y_test, shifted_value = preprocess_data(df_cleaned, target_column, model_type)
                    st.write("Training data shape:", X_train.shape)
                    st.write("Test data shape:", X_test.shape)
                    st.write("Mean value used for shifting:", shifted_value)
                except Exception as e:
                    st.error(f"Error: {e}")

            elif option == "Train Model":
                try:
                    with st.spinner("Training model..."):
                        df_cleaned = df.dropna(how='any').dropna(how='any', axis=1)
                        X_train, y_train, X_test, y_test, shifted_value = preprocess_data(df_cleaned, target_column, model_type)

                        if model_type == "LSTM":
                            model = build_model((X_train.shape[1], 1), model_type)
                        else:
                            model = build_model((X_train.shape[1],), model_type)

                        history = model.fit(
                            X_train, y_train,
                            batch_size=512, epochs=50,
                            validation_split=0.05, verbose=0
                        )
                        model.save(MODEL_PATH)
                        joblib.dump(history.history, HISTORY_PATH)
                        y_pred = model.predict(X_test)

                        actual = y_test + shifted_value
                        predicted = y_pred.flatten() + shifted_value
                        mape = calculate_mape(actual, predicted)
                        accuracy = 100 - mape
                        accuracy_metrics = {"mape": mape, "accuracy": accuracy}
                        joblib.dump(accuracy_metrics, ACCURACY_PATH)

                        prediction = pd.DataFrame({
                            'Actual': actual,
                            'Predicted': predicted
                        })
                        prediction.to_csv(PREDICTION_OUTPUT, index=False)

                        st.success("Model trained and saved successfully!")
                        st.write("Prediction Results (first few rows):")
                        st.write(prediction.head())
                        st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                        st.write(f"Accuracy (100 - MAPE): {accuracy:.2f}%")
                except Exception as e:
                    st.error(f"Training failed: {e}")

            elif option == "Prediction Plot":
                try:
                    df_cleaned = df.dropna(how='any').dropna(how='any', axis=1)
                    X_train, y_train, X_test, y_test, shifted_value = preprocess_data(df_cleaned, target_column, model_type)
                    if not os.path.exists(MODEL_PATH):
                        st.error("Model not found. Please train the model first.")
                    else:
                        model = load_model(MODEL_PATH)
                        y_pred = model.predict(X_test)
                        fig, ax = plt.subplots()
                        ax.plot(y_test + shifted_value, label='True')
                        ax.plot(y_pred.flatten() + shifted_value, label='Predicted')
                        ax.legend()
                        ax.set_title(f"Prediction vs True Values for {target_column}")
                        ax.set_xlabel("Time Step")
                        ax.set_ylabel(target_column)
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

            elif option == "Loss Curve":
                try:
                    if os.path.exists(HISTORY_PATH):
                        history = joblib.load(HISTORY_PATH)
                        fig, ax = plt.subplots()
                        ax.plot(history['loss'], label='Train Loss')
                        ax.plot(history['val_loss'], label='Validation Loss')
                        ax.set_title("Loss Curve")
                        ax.set_xlabel("Epoch")
                        ax.set_ylabel("Loss")
                        ax.legend()
                        st.pyplot(fig)
                    else:
                        st.warning("No saved training history. Please train the model first.")
                except Exception as e:
                    st.error(f"Failed to load loss curve: {e}")

            elif option == "Model Accuracy":
                try:
                    if os.path.exists(ACCURACY_PATH):
                        accuracy_metrics = joblib.load(ACCURACY_PATH)
                        st.subheader("Model Accuracy")
                        mape = accuracy_metrics["mape"]
                        accuracy = accuracy_metrics["accuracy"]
                        st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                        st.write(f"Accuracy (100 - MAPE): {accuracy:.2f}%")
                    else:
                        st.warning("Accuracy metrics not found. Please train the model first.")
                except Exception as e:
                    st.error(f"Failed to load accuracy metrics: {e}")

            elif option == "Future Prediction":
                try:
                    if not os.path.exists(MODEL_PATH):
                        st.error("Model not found. Please train the model first.")
                    else:
                        model = load_model(MODEL_PATH)
                        df_cleaned = df.dropna(how='any').dropna(how='any', axis=1)
                        
                        if target_column not in df_cleaned.columns:
                            st.error(f"Column '{target_column}' not found in the dataset.")
                        else:
                            data = df_cleaned[target_column].astype(float).values
                            if len(data) < SEQUENCE_LENGTH:
                                st.error(f"Dataset has {len(data)} rows, but at least {SEQUENCE_LENGTH} rows are required.")
                            else:
                                # Get the last sequence of closing prices
                                last_sequence = data[-SEQUENCE_LENGTH:]
                                shifted_value = last_sequence.mean()
                                last_sequence = last_sequence - shifted_value
                                
                                # Predict the next 7 days
                                num_days = 7
                                predictions = []
                                current_sequence = last_sequence.copy()
                                
                                for _ in range(num_days):
                                    if model_type == "LSTM":
                                        current_sequence_reshaped = current_sequence.reshape(1, SEQUENCE_LENGTH, 1)
                                    else:  # MLP
                                        current_sequence_reshaped = current_sequence.flatten().reshape(1, -1)
                                    pred = model.predict(current_sequence_reshaped, verbose=0)
                                    pred_value = pred[0][0] + shifted_value
                                    predictions.append(pred_value)
                                    current_sequence = np.roll(current_sequence, -1)
                                    current_sequence[-1] = pred[0][0]
                                
                                # Generate dates for the next 7 days (assuming last date is May 4, 2025)
                                last_date = pd.to_datetime("2025-05-04")  # Adjust based on your data
                                dates = [last_date + timedelta(days=i+1) for i in range(num_days)]
                                
                                # Create a DataFrame for predictions
                                prediction_df = pd.DataFrame({
                                    "Date": dates,
                                    "Predicted Closing Price": predictions
                                })
                                
                                st.subheader("Future Predictions for the Next Week")
                                st.write("Predicted closing prices from May 5 to May 11, 2025:")
                                st.dataframe(prediction_df)
                                
                                # Plot the predictions
                                fig, ax = plt.subplots()
                                ax.plot(dates, predictions, label="Predicted", marker='o')
                                ax.set_title(f"Predicted Closing Prices for {target_column}")
                                ax.set_xlabel("Date")
                                ax.set_ylabel(target_column)
                                ax.legend()
                                ax.grid(True)
                                plt.xticks(rotation=45)
                                st.pyplot(fig)
                                
                                # Save predictions to CSV
                                prediction_df.to_csv("future_predictions.csv", index=False)
                                st.write("Future predictions saved to 'future_predictions.csv'")
                                
                except Exception as e:
                    st.error(f"Failed to generate future predictions: {e}")

else:
    st.info("Please upload a dataset to begin.")