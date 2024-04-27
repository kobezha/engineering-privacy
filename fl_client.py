import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def split_dataset(X, y):
    X_train, X_test, y_train , y_test = train_test_split(X,y, test_size=0.2, random_state=0)
    scaler =StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# Load and preprocess the dataset
def load_data(partition_index, num_partitions):
    # Load dataset
    #URL to get the dataset from online 
    #url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    
    #file path for our own K-anonymized data
    file_path = "preprocessed_data.csv"

    col_names = ['num_pregnant', 'glucose_concentration', 'blood_pressure', 'skin_thickness', 'serum_insulin',
                 'BMI', 'pedigree_function', 'age', 'class']
    cat_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age','Outcomes']

    diabetes = pd.read_csv(file_path)

    #reconstruct X and y from the preprocessed_data 
    X = diabetes.iloc[:,:-1]
    y = diabetes.iloc[:,-1]


    #split dataset into testing and training 
    X_train, X_test, y_train, y_test = split_dataset(X,y)
    y_train = y_train.values
    y_test = y_test.values


    #further split dataset into partitions for each clinic 
    data_size = len(X_train)
    partition_size = data_size // num_partitions
    start_index = partition_index * partition_size
    end_index = start_index + partition_size if partition_index != num_partitions - 1 else data_size

    return X_train[start_index:end_index], X_test, y_train[start_index:end_index], y_test

# Parse command line arguments, Client id and total number of clients 
partition_index = int(sys.argv[1]) - 1  
num_partitions = int(sys.argv[2])       

# Load partitioned data
x_train, x_test, y_train, y_test = load_data(partition_index, num_partitions)

# Define and compile Keras model (local model from Zuofei)
model = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define Clinic as a Flower client
class Clinic(fl.client.NumPyClient):
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
    server_address="localhost:8080",  
    client=Clinic().to_client(),
    grpc_max_message_length=1024 * 1024 * 1024
)