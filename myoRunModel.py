import numpy as np
import pandas as pd
from pyomyo import Myo, emg_mode
import multiprocessing
import time
import sys
import threading
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib  # Para carregar o modelo salvo

# ------------ Myo Setup ---------------
q = multiprocessing.Queue()

def worker(q):
    m = Myo(mode=emg_mode.RAW)
    m.connect()
    
    def add_to_queue(emg, movement):
        q.put(emg)

    m.add_emg_handler(add_to_queue)
    
    def print_battery(bat):
        print("Battery level:", bat)

    m.add_battery_handler(print_battery)

    # Orange logo and bar LEDs
    m.set_leds([128, 0, 0], [128, 0, 0])
    # Vibrate to know we connected okay
    m.vibrate(1)
    
    """worker function"""
    while True:
        m.run()
    print("Worker Stopped")

# Função para extrair características de uma janela de dados
def extract_features(window):
    features = {}
    
    # Características no domínio do tempo
    features['mean'] = np.mean(window, axis=0)  # 8 características (1 por canal)
    features['var'] = np.var(window, axis=0)  # 8 características (1 por canal)
    features['rms'] = np.sqrt(np.mean(window**2, axis=0))  # 8 características (1 por canal)
    features['zcr'] = np.array([np.sum(np.abs(np.diff(np.sign(window[:, i])))) / len(window) for i in range(window.shape[1])])  # 8 características (1 por canal)
    features['wl'] = np.array([np.sum(np.abs(np.diff(window[:, i]))) for i in range(window.shape[1])])  # 8 características (1 por canal)
    
    # Transformada de Fourier (FFT)
    fft_vals = np.fft.fft(window, axis=0)
    power_spectrum = np.abs(fft_vals)**2
    freqs = np.fft.fftfreq(len(window))
    
    # Calcula a frequência média para cada canal
    mean_freq = np.zeros(window.shape[1])  # Array para armazenar a frequência média de cada canal
    for i in range(window.shape[1]):  # Itera sobre cada canal
        mean_freq[i] = np.sum(freqs * power_spectrum[:, i]) / np.sum(power_spectrum[:, i])
    features['mean_freq'] = mean_freq  # 8 características (1 por canal)
    
    # Retorna as características como um array 1D
    return np.concatenate([features['mean'], features['var'], features['rms'], features['zcr'], features['wl'], features['mean_freq']])

# Função para verificar a entrada do usuário
def check_user_input(stop_event):
    while not stop_event.is_set():
        user_input = input()  # Aguarda o usuário pressionar Enter
        if user_input.strip().lower() == 's':
            stop_event.set()  # Sinaliza para parar a coleta

# -------- Main Program Loop -----------
if __name__ == "__main__":
    # Carrega o modelo treinado
    pipeline = joblib.load('emg_model.pkl')  # Substitua pelo caminho do seu modelo salvo

    p = multiprocessing.Process(target=worker, args=(q,))
    p.start()

    # Parâmetros da janela
    window_size = 40  # 200 ms a 200 Hz
    emg_buffer = []

    # Dicionário de gestos (apenas 3 gestos)
    gestures = {
        0: "Abrir",
        1: "Fechar",
        2: "Relaxar",
    }

    try:
        print("Bem-vindo ao classificador de gestos em tempo real!")
        print("Pressione 's' e Enter para parar.")

        # Evento para sinalizar a parada da coleta
        stop_event = threading.Event()

        # Thread para verificar a entrada do usuário
        input_thread = threading.Thread(target=check_user_input, args=(stop_event,))
        input_thread.start()

        # Loop de coleta de dados e classificação
        while not stop_event.is_set():
            # Obtém os dados EMG e processa
            while not q.empty():
                emg = list(q.get())
                emg_buffer.append(emg)

                # Quando a janela atingir o tamanho desejado (200 ms)
                if len(emg_buffer) >= window_size:
                    # Converte para um array numpy
                    window = np.array(emg_buffer[-window_size:])

                    # Extrai características
                    features = extract_features(window)

                    # Faz a previsão usando o modelo
                    gesture_id = pipeline.predict([features])[0]
                    gesture_name = gestures.get(gesture_id, "Desconhecido")

                    # Exibe o gesto classificado
                    print(f"Gesto classificado: {gesture_name}")

                    # Limpa o buffer para a próxima janela
                    emg_buffer.clear()

    except KeyboardInterrupt:
        print("Classificação concluída.")

    finally:
        p.terminate()  # Encerra o processo do Myo
        p.join()  # Aguarda o processo terminar
        print("Programa encerrado.")