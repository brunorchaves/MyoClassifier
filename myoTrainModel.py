import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import joblib  # Para salvar o modelo e o scaler

# Carrega o dataset
df = pd.read_csv('emg_dataset.csv')

# Separa as features (X) e os rótulos (y)
X = df.drop(columns=['label'])  # Todas as colunas, exceto 'label'
y = df['label']  # Apenas a coluna 'label'

# Divide o dataset em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliza os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reduz a dimensionalidade com PCA
pca = PCA(n_components=0.95)  # Mantém 95% da variância
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Cria a rede neural
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),  # Camada de entrada
    BatchNormalization(),  # Normalização
    Dropout(0.5),  # Dropout para evitar overfitting
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),  # Camada oculta
    BatchNormalization(),  # Normalização
    Dropout(0.5),  # Dropout
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),  # Camada oculta
    BatchNormalization(),  # Normalização
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),  # Camada oculta
    Dense(3, activation='softmax')  # Camada de saída (3 classes)
])

# Compila o modelo
optimizer = Nadam(learning_rate=0.0001)  # Otimizador com taxa de aprendizado ajustada
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define early stopping para evitar overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Treina o modelo
history = model.fit(X_train, y_train,
                    epochs=200,  # Aumentar o número de épocas
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping])

# Avalia o modelo no conjunto de teste
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Acurácia no conjunto de teste: {test_accuracy:.4f}")

# Salva o modelo treinado
model.save('emg_neural_network.h5')

# Salva o scaler e o PCA
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')
print("Modelo, scaler e PCA salvos.")