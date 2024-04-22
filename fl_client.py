import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load and preprocess the dataset
def load_data(partition_index, num_partitions):
    # Load dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    col_names = ['num_pregnant', 'glucose_concentration', 'blood_pressure', 'skin_thickness', 'serum_insulin',
                 'BMI', 'pedigree_function', 'age', 'class']
    diabetes = pd.read_csv(url, names=col_names)

    # Clean data by replacing 0 with NaN and imputing with mean
    diabetes[['glucose_concentration', 'blood_pressure', 'skin_thickness', 'serum_insulin', 'BMI']] = \
        diabetes[['glucose_concentration', 'blood_pressure', 'skin_thickness', 'serum_insulin', 'BMI']].replace(0, np.nan)
    diabetes.fillna(diabetes.mean(), inplace=True)

    # Features and Targets
    X = diabetes.drop('class', axis=1).values
    y = diabetes['class'].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into partitions for federated learning
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
    data_size = len(X_train)
    partition_size = data_size // num_partitions
    start_index = partition_index * partition_size
    end_index = start_index + partition_size if partition_index != num_partitions - 1 else data_size

    return X_train[start_index:end_index], X_test, y_train[start_index:end_index], y_test

# Parse command line arguments
partition_index = int(sys.argv[1]) - 1  # Client index (1-based index to 0-based index conversion)
num_partitions = int(sys.argv[2])       # Total number of clients

# Load partitioned data
x_train, x_test, y_train, y_test = load_data(partition_index, num_partitions)

# Define and compile Keras model
model = keras.Sequential([
    keras.layers.Dense(12, input_dim=8, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define Flower client
class DiabetesClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        history = model.fit(x_train, y_train, epochs=1, batch_size=10, verbose=0)
        print("Training loss:", history.history['loss'][0])
        print("Training accuracy:", history.history['accuracy'][0])
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        return loss, len(x_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_client(
    server_address="localhost:8080",  # Server's IP address and port (replace 'localhost' if needed)
    client=DiabetesClient().to_client(),
    grpc_max_message_length=1024 * 1024 * 1024
)