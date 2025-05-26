from flask import Flask, render_template, request, jsonify  # Added render_template here
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import traceback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def create_model(input_shape):
    model = Sequential([

# lstm - long short term mem. It is a type of recurrent neural network. 
# why lstm - bec standard rnn have gradient vansishing problem which makes them forget long term dependency LSTM solves this using special memory cells and gates that decide what to remember and what to forget.

        LSTM(64, return_sequences=True, input_shape=input_shape,   #First LSTM layer with 64 units, returns sequences for next LSTM layer.
             #it is power of 2, good for gpu
             #It's large enough to capture patterns in time-series data.
             #Not too large to slow down training or cause overfitting.
             kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'),

        Dropout(0.3),  #Drops 30% of units during training to prevent overfitting.
        LSTM(32, return_sequences=False, #Second LSTM layer, now reduces output to one sequence.
             #Acts as a bottleneck to reduce the complexity and avoid overfitting.
             kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'),
        Dropout(0.3),
        Dense(16, activation='relu'), #A dense layer with 16 units and ReLU activation.
        Dense(1) #We're predicting a single value — the stock price at the next time step.
    ])
    model.compile(optimizer='adam', loss='mse') #Compiles the model using Adam optimizer and MSE (Mean Squared Error) loss.
    return model

@app.route('/')
#helps users to hit the homepage, ye serve karta h like index.html
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'train_file' not in request.files or 'test_file' not in request.files:
            return jsonify({"error": "Missing file uploads"}), 400
            
        train_file = request.files['train_file']
        test_file = request.files['test_file']
        
        if train_file.filename == '' or test_file.filename == '':
            return jsonify({"error": "No files selected"}), 400

        # used to save the files
        train_path = os.path.join(app.config['UPLOAD_FOLDER'], 'train_temp.csv')
        test_path = os.path.join(app.config['UPLOAD_FOLDER'], 'test_temp.csv')
        train_file.save(train_path)
        test_file.save(test_path)

        # helps toReads CSVs while handling numbers with commas like “1,000”.
        train_df = pd.read_csv(train_path, thousands=',')
        test_df = pd.read_csv(test_path, thousands=',')

        
        # lsit coloumn in csv and close checks if coloumn is not missing
        if 'Close' not in train_df.columns or 'Close' not in test_df.columns:
            return jsonify({"error": "CSV must contain 'Close' column"}), 400
            
        #tries to convert all values to numbers.
        #If any value is non-numeric (like "N/A", "abc"), it is replaced with NaN (Not a Number).
        train_df['Close'] = pd.to_numeric(train_df['Close'], errors='coerce')
        test_df['Close'] = pd.to_numeric(test_df['Close'], errors='coerce')
        
        # Check for invalid values
        if train_df['Close'].isnull().any() or test_df['Close'].isnull().any():
            return jsonify({"error": "Non-numeric values found in 'Close' column"}), 400
            
        if len(train_df) < 60:
            return jsonify({"error": "Training data needs at least 60 records"}), 400





        # Preprocess data
        #Scales Close prices between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_data = scaler.fit_transform(train_df[['Close']])
        test_data = scaler.transform(test_df[['Close']])

        # Create time-series sequences
        #X = sequences of 60 continuous past values
        #y = the value that comes right after each 60-step sequence
        def create_sequences(data, window_size=60):
            X, y = [], []
            for i in range(window_size, len(data)):
                X.append(data[i-window_size:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)
            

        #Reshapes X_train into 3D format required by LSTM.    
        X_train, y_train = create_sequences(train_data)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Build and train model
        #Builds and trains the model for 15 epochs with batch size 32.
        model = create_model((X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)

        # Prepare test inputs
        combined_data = pd.concat([train_df['Close'], test_df['Close']])   #Combines train and test for smoother transitions.
        inputs = combined_data[len(combined_data)-len(test_df)-60:].values
        inputs = scaler.transform(inputs.reshape(-1,1))
        

        #Creates test sequences (just like training).
        X_test = []
        for i in range(60, 60+len(test_df)):
            X_test.append(inputs[i-60:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        
        # Predicts using the trained model.
        #Inverse transforms to get actual price scale.
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions).flatten()
        actual = test_df['Close'].values

        # Calculates RMSE, MAE, and a custom accuracy.
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae = mean_absolute_error(actual, predictions)
        accuracy = 100 * (1 - np.mean(np.abs(actual - predictions)/np.mean(actual)))

        # Clean up
        os.remove(train_path)
        os.remove(test_path)

        return jsonify({ #Sends the result back to the frontend.
            "status": "success",
            "labels": [str(i) for i in range(len(actual))],
            "actual_prices": actual.tolist(),
            "predicted_prices": predictions.tolist(),
            "rmse": float(rmse),
            "mae": float(mae),
            "accuracy": float(accuracy)
        })

    except Exception as e:
        #Prints and logs detailed error in server logs.
        error_trace = traceback.format_exc()
        print(f"Error occurred: {error_trace}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "trace": error_trace
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
































