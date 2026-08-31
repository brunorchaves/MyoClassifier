# Classificador de gestos com rede neural (pretrain no EMG-EPN612 + fine-tuning)

## Contexto

O projeto hoje classifica gestos EMG de duas formas paralelas:
- **Pipeline principal** (`src/emgGestureTrainer.py` + `src/data/vals*.dat`): 1-NN por amostra bruta (8 canais), classes 1=Relaxed(mão aberta), 2=fist, 3=spock, 4=Pointing — mas a classe 4 (Pointing) **não tem nenhuma amostra gravada**, então só funciona via tecla manual.
- **Pipeline experimental** (`myTry/`): extrai 5 features por canal (EWL, RMS, MMAV2, DASDV, MFL) em janelas de 100 amostras/50% overlap e treina um kNN. O CSV atual (`myTry/emg_features_all_gestures.csv`) tem só **113 linhas para 4 classes desbalanceadas** (75/20/3/14).

Avaliamos que uma rede neural treinada só nesses ~113 exemplos overfitaria e teria pior desempenho que o kNN atual. A alternativa combinada — pré-treinar em um dataset público gravado com o mesmo hardware (Myo, 8 canais, 200Hz) e depois fazer fine-tuning nos dados próprios — pode superar o kNN sem exigir que o usuário grave milhares de amostras.

Pesquisamos datasets compatíveis e confirmamos:
- **EMG-EPN612** (Zenodo DOI [10.5281/zenodo.4023305](https://doi.org/10.5281/zenodo.4023305), CC-BY 4.0, acesso aberto, ZIP de 5.5GB): 612 sujeitos, 1 Myo, 8 canais, 200Hz, gestos **rest, wave-in, wave-out, pinch, open, fist** — `open` e `fist` mapeiam diretamente para as classes 1 e 2 do projeto. Não cobre `spock` nem `pointing`.
- **NinaPro DB5** ([10.5281/zenodo.1000116](https://doi.org/10.5281/zenodo.1000116), CC-BY-ND 4.0, 201.7MB, MATLAB): só 10 sujeitos, mas 53 movimentos finos com **2 Myos (16 canais)** — usaríamos só os primeiros 8 canais (1º armband) para bater com o hardware do projeto. Fica como fonte secundária opcional para tentar achar um movimento próximo de "pointing", não é o foco desta primeira entrega.

Decisões já validadas com o usuário: framework **PyTorch**, **GPU disponível** para treino, e escopo é o **pipeline completo integrado** — não só um spike de avaliação, mas um classificador plugável nas interfaces já existentes (`Live_Classifier` em `src/emgGestureTrainer.py`, e o loop de inferência em `myTry/myoRunModel.py`).

## Abordagem

Criar um novo módulo `nn_classifier/` (paralelo a `src/` e `myTry/`, não misturado com eles) com um pipeline de duas etapas — pretrain no EPN-612, fine-tune nos dados do próprio usuário — mais um adaptador que expõe o modelo treinado através dos dois contratos de classificador que já existem no projeto.

### 1. Download e parsing do EMG-EPN612
- `nn_classifier/data/download_epn612.py`: baixa o ZIP do Zenodo (URL do arquivo, não só o DOI), confere o MD5 (`98bd3c315efab607cc54b2ed2f8f3ada`) e extrai para `nn_classifier/data/raw/epn612/`.
- `nn_classifier/data/epn612_dataset.py`: parseia o JSON por sujeito/repetição. Cada gesto dinâmico vem com índices de onset/offset dentro de uma janela de 5s a 200Hz — usar esses índices para fatiar em janelas de **100 amostras / 50% overlap** (mesma convenção de `WINDOW_SIZE`/`OVERLAP` já usada em `myTry/myoFeatures.py` e `myTry/myoRunModel.py`, para manter tudo no mesmo formato de entrada). Janelas fora do intervalo onset-offset viram classe `rest`(0).
- `nn_classifier/data/label_map.py`: mapeia os 6 rótulos do EPN612 para o vocabulário do projeto — `rest`→0, `open`→1 (Relaxed), `fist`→2. `wave-in`, `wave-out`, `pinch` ficam como classes extras próprias (não existem hoje no projeto) — decisão de manter ou descartar essas 3 fica documentada como TODO explícito no código, já que não há equivalente atual em `hand3d/gestos.json`.
- Sem extração de features: o modelo consome a janela bruta (8 canais × 100 amostras), igual ao que chega em `emg_mode.RAW`/`FILTERED`.

### 2. Modelo
- `nn_classifier/model.py`: CNN 1D pequena — 3–4 blocos conv (kernel ~5, canais 8→32→64→128) + pooling global + cabeça linear para N classes. Simples e rápida de treinar em GPU nos 612 sujeitos; documentar como alternativa um encoder LSTM, mas CNN é a escolha default (é o que a maior parte da literatura sobre EPN612/NinaPro usa e não exige sequência variável).

### 3. Treino em duas etapas
- `nn_classifier/train.py`:
  - **Pretrain**: treina no split oficial de treino do EPN612 (306 sujeitos), valida no split de teste oficial (306 sujeitos) — dá uma métrica de sanity check comparável à literatura antes de tocar nos dados do usuário. Salva `nn_classifier/checkpoints/epn612_pretrained.pt`.
  - **Fine-tune**: carrega o checkpoint pretreinado, congela as camadas conv (ou usa LR baixo em todas), reinicializa/ajusta a cabeça linear para as classes do projeto, e treina nos dados próprios — tanto `src/data/vals1..3.dat` (reamostrados em janelas de 100, já que hoje são por-amostra) quanto `myTry/emg_features_all_gestures.csv`-equivalente em janelas brutas (temos que gerar as janelas brutas correspondentes, não as features, então provavelmente será preciso regravar/reagrupar os `.dat` existentes). Como o dataset próprio é minúsculo, usar k-fold ou leave-one-out em vez de um split fixo. Salva `nn_classifier/checkpoints/finetuned.pt`.

### 4. Adaptador para as interfaces existentes
- `nn_classifier/classifier_adapter.py`:
  - `NNGestureClassifier(Live_Classifier)` — subclasse de `Live_Classifier` (definida em [src/emgGestureTrainer.py:183](../../../src/emgGestureTrainer.py#L183)) que mantém um buffer circular das últimas 100 amostras brutas recebidas por `classify(emg)`; só roda o forward pass da CNN quando o buffer está cheio (senão retorna 0, igual ao comportamento atual quando não há dados suficientes). Assim ela entra de graça no `MyoClassifier`/`EMGHandler` já existentes (votação por histórico de 25 amostras em [src/emgGestureTrainer.py:88-109](../../../src/emgGestureTrainer.py#L88-L109) continua funcionando sem mudança).
  - `predict_window(window: np.ndarray[100,8]) -> int` — função simples que envolve o mesmo modelo para o loop de janelas já usado em `myTry/myoRunModel.py:35-65`, só troca a chamada `model.predict([features])` por essa função (sem extração de features).

### 5. Avaliação comparativa
- `nn_classifier/evaluate.py`: compara, no mesmo holdout dos dados próprios do usuário, a acurácia/F1/matriz de confusão do modelo NN (só pretreinado vs. fine-tuned) contra o `knn_gesture_classifier.joblib` atual. Esse script é o que responde diretamente a pergunta original — "a rede neural traz resultados melhores?" — com números, antes de trocar o classificador em produção.

### Dependências novas
- `torch` (build com CUDA, já que há GPU disponível) em `requirements.txt`.
- Nenhuma outra dependência nova é estritamente necessária (download via `urllib`/`zipfile` da stdlib).

### Riscos e limitações a documentar no README do novo módulo
- `spock` e `pointing` continuam sem dados públicos equivalentes — essas duas classes só melhoram se o usuário gravar mais amostras próprias; o EPN612 não resolve isso.
- Domain shift: EPN612 foi gravado em outros braços/posicionamentos de eletrodo. O pretrain ajuda a aprender representações gerais de ativação muscular, mas o fine-tuning nos dados do próprio usuário é essencial — é por isso que `evaluate.py` mede isso explicitamente em vez de assumir que pretrain=melhoria automática.
- Download de 5.5GB e treino em GPU são um investimento de tempo/disco bem maior que o pipeline atual — vale deixar isso explícito para o usuário antes de rodar.

## Verificação
1. `python nn_classifier/data/download_epn612.py` — confirma MD5 e contagem de arquivos extraídos.
2. `python -c "from nn_classifier.data.epn612_dataset import load_epn612; ..."` — checar shape e distribuição de classes das janelas geradas.
3. `python nn_classifier/train.py --stage pretrain` — acompanhar acurácia de validação no split oficial de teste do EPN612 (esperado ficar na faixa alta, comparável à literatura, como sanity check).
4. `python nn_classifier/train.py --stage finetune` — treinar nos dados próprios.
5. `python nn_classifier/evaluate.py` — comparar NN (pretrain-only e fine-tuned) vs. kNN atual lado a lado.
6. Trocar a chamada em `myTry/myoRunModel.py` por `predict_window` e testar em tempo real com o Myo físico, comparando taxa/latência de classificação com o comportamento atual.
