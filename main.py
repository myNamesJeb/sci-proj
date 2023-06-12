import yfinance as yf
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

avgs = []
numbers_for_avg = []
activation_functions = ['sigmoid']
for i in range(10 * len(activation_functions)):
    # Set the seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Define the stock symbol and download the historical data
    stock_symbol = "NVDA"  # Change this to the desired stock symbol
    data = yf.download(stock_symbol, start="2010-01-01", end="2023-06-01")

    # Preprocess the data
    data["Close"] = data["Close"].fillna(method="ffill")  # Forward fill missing values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

    # Split the data into training and testing sets
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    # Define the window size for the LSTM model
    window_size = 20

    # Create the input features and target labels
    def create_dataset(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i : (i + window_size)])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(train_data, window_size)
    X_test, y_test = create_dataset(test_data, window_size)

    # Build the LSTM model
    activation_function = activation_functions[i // 10]  # Change this to select the desired activation function
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(window_size, 1), activation=activation_function),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(optimizer="adam", loss="mse")

    # Train the model
    model.fit(X_train, y_train, epochs=25, batch_size=80)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Inverse transform the predictions and actual values
    predictions = scaler.inverse_transform(predictions)
    actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Get the latest predicted price
    latest_prediction = predictions[-1][0]

    # Get the latest actual price
    latest_actual_price = actual_values[-1][0]

    # Calculate the percentage error
    percent_error = ((latest_prediction - latest_actual_price) / latest_actual_price) * 100

    # Print the predicted and actual prices, and the percentage error
    print(f"Predicted Price: {latest_prediction}")
    print(f"Actual Price: {latest_actual_price}")
    print(f"Percentage Error: {percent_error:.2f}%")

    # Open the file in append mode
    with open("results.txt", "a") as file:
        # Write to a new line
        file.write(f"error: {percent_error:.2f}% prediction: {latest_prediction:.2f} actual_price: {latest_actual_price:.2f} function:{activation_function}\n")
        numbers_for_avg.append(percent_error)  # Add the percent error to the list
        if (i + 1) % 10 == 0:
            avg_error = sum(numbers_for_avg) / len(numbers_for_avg)
            file.write(f"average percent error: {avg_error:.2f}%\n")
            numbers_for_avg = []  # Clear the list for the next set of 10 iterations
