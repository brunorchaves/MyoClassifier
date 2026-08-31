"""Esquema de rotulos unificado entre o EMG-EPN612 (pretrain) e os dados
proprios do projeto (fine-tune), alinhado com o que 'src/emgGestureTrainer.py'
+ 'src/data/vals*.dat' ja usam hoje (ver hand3d/README.md).

O myTry/ usa uma numeracao DIFERENTE e incompativel para os mesmos gestos
(1=Close,2=Open,3=Pointing, sem spock) -- nao ha mapeamento documentado entre
os dois esquemas do projeto, entao este modulo deliberadamente segue apenas o
esquema de src/data/vals*.dat, que e o que o sistema ao vivo (hand3d) usa.
"""

REST = 0
OPEN = 1       # "Relaxed" / mao aberta
FIST = 2       # mao fechada
SPOCK = 3      # saudacao vulcana
POINTING = 4   # sem amostras proprias gravadas ainda (ver hand3d/README.md)
WAVE_IN = 5
WAVE_OUT = 6
PINCH = 7

CLASS_NAMES = {
    REST: "rest",
    OPEN: "open",
    FIST: "fist",
    SPOCK: "spock",
    POINTING: "pointing",
    WAVE_IN: "waveIn",
    WAVE_OUT: "waveOut",
    PINCH: "pinch",
}

# Gestos que o projeto de fato usa (hand3d/gestos.json, src/data/vals*.dat).
# waveIn/waveOut/pinch só existem no EMG-EPN612 e ficam como classes extras
# só durante o pretrain -- ver docs/plans/emg-nn-pretrain-finetune/PLAN.md.
PROJECT_CLASSES = [REST, OPEN, FIST, SPOCK, POINTING]

# Tamanho fixo da cabeça da rede (0..7), igual no pretrain e no fine-tune.
# Os ids acima NAO sao contiguos (SPOCK/POINTING nao existem no EPN612;
# WAVE_OUT/PINCH nao existem no projeto) -- por isso o cabecote precisa
# cobrir o maior id usado em qualquer um dos dois estagios, e nao apenas
# len(PROJECT_CLASSES) ou len(EPN612_GESTURE_TO_LABEL). Alguns neuronios de
# saida ficam sem gradiente em cada estagio; e um custo irrelevante numa
# rede desse tamanho e mantem os ids de gesto estaveis entre os dois.
NUM_CLASSES = PINCH + 1

EPN612_GESTURE_TO_LABEL = {
    "noGesture": REST,
    "open": OPEN,
    "fist": FIST,
    "waveIn": WAVE_IN,
    "waveOut": WAVE_OUT,
    "pinch": PINCH,
}
