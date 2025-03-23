import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import pickle  # For saving the scaler

# Load the dataset
def load_dataset(file_path):
    """
    Load the dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        X (numpy array): Features.
        y (numpy array): Labels.
    """
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values  # All columns except the last one (features)
    y = df.iloc[:, -1].values   # Last column (labels)
    return X, y

# Preprocess the data
def preprocess_data(X, y):
    """
    Preprocess the data for training an RNN model.
    
    Args:
        X (numpy array): Features.
        y (numpy array): Labels.
    
    Returns:
        X_train (numpy array): Training features.
        X_test (numpy array): Testing features.
        y_train (numpy array): Training labels.
        y_test (numpy array): Testing labels.
    """
    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Save the scaler to a file
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved to 'scaler.pkl'.")
    
    # Convert labels to one-hot encoding
    y = to_categorical(y)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Reshape the data for RNN input (samples, timesteps, features)
    # Here, we assume each window is a timestep, and features are the 40 features (5 features * 8 channels)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    return X_train, X_test, y_train, y_test

# Build the RNN model
def build_rnn_model(input_shape, num_classes):
    """
    Build an RNN model for gesture classification.
    
    Args:
        input_shape (tuple): Shape of the input data (timesteps, features).
        num_classes (int): Number of gesture classes.
    
    Returns:
        model (keras model): Compiled RNN model.
    """
    model = Sequential()
    
    # Add a SimpleRNN layer
    model.add(SimpleRNN(64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))  # Add dropout for regularization
    
    # Add a Dense output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Train the model
def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """
    Train the RNN model.
    
    Args:
        model (keras model): Compiled RNN model.
        X_train (numpy array): Training features.
        y_train (numpy array): Training labels.
        X_test (numpy array): Testing features.
        y_test (numpy array): Testing labels.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    
    Returns:
        history (keras history): Training history.
    """
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )
    
    return history

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.
    
    Args:
        model (keras model): Trained RNN model.
        X_test (numpy array): Testing features.
        y_test (numpy array): Testing labels.
    """
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

# Save the model
def save_model(model, file_path):
    """
    Save the trained model to a file.
    
    Args:
        model (keras model): Trained RNN model.
        file_path (str): Path to save the model.
    """
    model.save(file_path)
    print(f"Model saved to {file_path}")

# Main function
def main():
    # Load the dataset
    file_path = 'emg_features_all_gestures.csv'
    X, y = load_dataset(file_path)
    
    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Build the RNN model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
    num_classes = y_train.shape[1]  # Number of gesture classes
    model = build_rnn_model(input_shape, num_classes)
    
    # Train the model
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # Save the model
    save_model(model, 'rnn_gesture_classifier.h5')

if __name__ == "__main__":
    main()