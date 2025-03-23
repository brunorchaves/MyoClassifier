import numpy as np
from joblib import load
from pyomyo import Myo, emg_mode
import multiprocessing

# Carregar o modelo treinado
model_path = 'knn_gesture_classifier.joblib'
model = load(model_path)

# Função para normalizar os dados
def normalize_emg_data(emg_data):
    """
    Normaliza os dados de EMG para o intervalo [-1, 1].
    """
    return [value / 128.0 for value in emg_data]

# Função para extrair características de uma janela de dados
def extract_features(window):
    """
    Extrai características de uma janela de dados de EMG.
    """
    features = []
    for channel in range(8):  # Loop através de todos os 8 canais
        channel_data = window[:, channel]
        features.extend([
            np.sum(np.abs(np.diff(channel_data))),  # Enhanced Wavelength (EWL)
            np.sqrt(np.mean(np.square(channel_data))),  # Root Mean Square (RMS)
            np.mean(np.abs(channel_data)),  # Modified Mean Absolute Value (MMAV)
            np.sqrt(np.mean(np.square(np.diff(channel_data)))),  # Difference Absolute Standard Deviation Value (DASDV)
            np.log10(np.sum(np.square(np.diff(channel_data))))  # Maximum Fractal Length (MFL)
        ])
    return features

# Função para classificar gestos em tempo real
def classify_gestures(q):
    """
    Classifica gestos em tempo real usando o modelo carregado.
    """
    emg_data = []
    window_size = 100  # Tamanho da janela (100 amostras)
    overlap = 50       # Sobreposição (50 amostras)
    step = window_size - overlap

    while True:
        if not q.empty():
            emg = list(q.get())
            emg_data.append(emg)

            # Quando tivermos dados suficientes para uma janela
            if len(emg_data) >= window_size:
                # Extrair a janela atual
                window = np.array(emg_data[-window_size:])
                
                # Normalizar os dados
                normalized_window = normalize_emg_data(window)
                
                # Extrair características
                features = extract_features(np.array(normalized_window))
                
                # Fazer a previsão
                prediction = model.predict([features])
                print(f"Gesto previsto: {prediction[0]}")

                # Remover dados antigos para a próxima janela
                emg_data = emg_data[step:]

# Função worker para coletar dados do Myo
def myo_worker(q):
    """
    Worker function para coletar dados do Myo armband.
    """
    m = Myo(mode=emg_mode.RAW)
    m.connect()
    
    def add_to_queue(emg, movement):
        q.put(emg)

    m.add_emg_handler(add_to_queue)
    
    def print_battery(bat):
        print("Battery level:", bat)

    m.add_battery_handler(print_battery)

    # Configurações do Myo
    m.set_leds([128, 0, 0], [128, 0, 0])
    m.vibrate(1)  # Vibração para indicar conexão
    
    while True:
        m.run()

# Função principal
def main():
    q = multiprocessing.Queue()

    # Iniciar o worker do Myo em um processo separado
    myo_process = multiprocessing.Process(target=myo_worker, args=(q,))
    myo_process.start()

    try:
        # Classificar gestos em tempo real
        classify_gestures(q)
    except KeyboardInterrupt:
        print("Parando a classificação...")
    finally:
        myo_process.terminate()
        myo_process.join()

if __name__ == "__main__":
    main()