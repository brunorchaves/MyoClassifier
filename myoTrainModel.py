import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib  # For saving the model

# Function to load CSV data
def load_csv_data(file_path):
    """
    Load features and labels from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        X (numpy array): Features.
        y (numpy array): Labels.
    """
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values  # All columns except the last one are features
    y = df.iloc[:, -1].values   # Last column is the label
    return X, y

# Function to train SKNN model
def train_sknn(X_train, y_train):
    """
    Train a Subspace K-Nearest Neighbors (SKNN) model.
    
    Args:
        X_train (numpy array): Training features.
        y_train (numpy array): Training labels.
    
    Returns:
        model: Trained SKNN model.
    """
    # Create a pipeline with PCA for dimensionality reduction and SKNN
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('pca', PCA(n_components=0.95)),  # Reduce dimensions while retaining 95% variance
        ('sknn', KNeighborsClassifier(n_neighbors=5, algorithm='auto'))  # SKNN classifier
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    return pipeline

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.
    
    Args:
        model: Trained SKNN model.
        X_test (numpy array): Test features.
        y_test (numpy array): Test labels.
    
    Returns:
        accuracy (float): Accuracy of the model on the test set.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Function to save the model locally
def save_model(model, file_path):
    """
    Save the trained model to a file.
    
    Args:
        model: Trained SKNN model.
        file_path (str): Path to save the model.
    """
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

# Main function to train and evaluate SKNN
def main():
    # Load CSV data (replace with the path to your CSV file)
    file_path = 'emg_features_all_gestures.csv'  # Replace with the actual file path
    X, y = load_csv_data(file_path)
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the SKNN model
    print("Training SKNN model...")
    model = train_sknn(X_train, y_train)
    
    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Save the model locally
    model_file_path = 'sknn_model_4_classes.pkl'  # Path to save the model
    save_model(model, model_file_path)

# Run the main function
if __name__ == "__main__":
    main()