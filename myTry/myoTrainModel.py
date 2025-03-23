import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump

# Carregar o dataset
def load_dataset(file_path):
    """
    Carrega o dataset de características de EMG a partir de um arquivo CSV.
    
    Args:
        file_path (str): Caminho para o arquivo CSV.
    
    Returns:
        X (numpy array): Features (características).
        y (numpy array): Labels (rótulos).
    """
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values  # Todas as colunas exceto a última (features)
    y = df.iloc[:, -1].values   # Última coluna (labels)
    return X, y

# Pré-processar os dados
def preprocess_data(X, y):
    """
    Pré-processa os dados para treinamento do modelo.
    
    Args:
        X (numpy array): Features.
        y (numpy array): Labels.
    
    Returns:
        X_train (numpy array): Dados de treinamento (features).
        X_test (numpy array): Dados de teste (features).
        y_train (numpy array): Rótulos de treinamento.
        y_test (numpy array): Rótulos de teste.
    """
    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalizar as features (média = 0, desvio padrão = 1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Treinar o modelo kNN
def train_knn(X_train, y_train, n_neighbors=5):
    """
    Treina um modelo k-Nearest Neighbors (kNN).
    
    Args:
        X_train (numpy array): Dados de treinamento (features).
        y_train (numpy array): Rótulos de treinamento.
        n_neighbors (int): Número de vizinhos (k).
    
    Returns:
        model (KNeighborsClassifier): Modelo kNN treinado.
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

# Avaliar o modelo
def evaluate_model(model, X_test, y_test):
    """
    Avalia o modelo kNN no conjunto de teste.
    
    Args:
        model (KNeighborsClassifier): Modelo kNN treinado.
        X_test (numpy array): Dados de teste (features).
        y_test (numpy array): Rótulos de teste.
    """
    # Fazer previsões
    y_pred = model.predict(X_test)
    
    # Calcular a acurácia
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo: {accuracy:.4f}")
    
    # Mostrar o relatório de classificação
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    # Mostrar a matriz de confusão
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carregar o dataset
def load_dataset(file_path):
    """
    Carrega o dataset de características de EMG a partir de um arquivo CSV.
    
    Args:
        file_path (str): Caminho para o arquivo CSV.
    
    Returns:
        X (numpy array): Features (características).
        y (numpy array): Labels (rótulos).
    """
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values  # Todas as colunas exceto a última (features)
    y = df.iloc[:, -1].values   # Última coluna (labels)
    return X, y

# Pré-processar os dados
def preprocess_data(X, y):
    """
    Pré-processa os dados para treinamento do modelo.
    
    Args:
        X (numpy array): Features.
        y (numpy array): Labels.
    
    Returns:
        X_train (numpy array): Dados de treinamento (features).
        X_test (numpy array): Dados de teste (features).
        y_train (numpy array): Rótulos de treinamento.
        y_test (numpy array): Rótulos de teste.
    """
    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalizar as features (média = 0, desvio padrão = 1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Treinar o modelo kNN
def train_knn(X_train, y_train, n_neighbors=5):
    """
    Treina um modelo k-Nearest Neighbors (kNN).
    
    Args:
        X_train (numpy array): Dados de treinamento (features).
        y_train (numpy array): Rótulos de treinamento.
        n_neighbors (int): Número de vizinhos (k).
    
    Returns:
        model (KNeighborsClassifier): Modelo kNN treinado.
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

# Avaliar o modelo
def evaluate_model(model, X_test, y_test):
    """
    Avalia o modelo kNN no conjunto de teste.
    
    Args:
        model (KNeighborsClassifier): Modelo kNN treinado.
        X_test (numpy array): Dados de teste (features).
        y_test (numpy array): Rótulos de teste.
    """
    # Fazer previsões
    y_pred = model.predict(X_test)
    
    # Calcular a acurácia
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo: {accuracy:.4f}")
    
    # Mostrar o relatório de classificação
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    # Mostrar a matriz de confusão
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

# Função principal
def main():
    # Carregar o dataset
    file_path = 'emg_features_all_gestures.csv'
    X, y = load_dataset(file_path)
    
    # Pré-processar os dados
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Treinar o modelo kNN
    print("Treinando o modelo kNN...")
    model = train_knn(X_train, y_train, n_neighbors=5)
    
    # Avaliar o modelo
    print("\nAvaliando o modelo...")
    evaluate_model(model, X_test, y_test)
    
    # Salvar o modelo treinado
    model_path = 'knn_gesture_classifier.joblib'
    dump(model, model_path)
    print(f"\nModelo salvo em {model_path}")

if __name__ == "__main__":
    main()