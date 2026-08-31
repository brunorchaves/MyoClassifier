# nn_classifier — CNN pretreinada no EMG-EPN612 + fine-tuning nos seus dados

Módulo experimental (paralelo a [`src/`](../src/) e [`myTry/`](../myTry/)) que treina
um classificador de gestos em PyTorch em duas etapas: pré-treino no dataset público
[EMG-EPN612](https://doi.org/10.5281/zenodo.4023305) (612 sujeitos, gravado com o
mesmo hardware deste projeto — 1 Myo armband, 8 canais, 200Hz) e depois fine-tuning
nos seus próprios dados. O raciocínio completo e as decisões de design estão em
[`docs/plans/emg-nn-pretrain-finetune/PLAN.md`](../docs/plans/emg-nn-pretrain-finetune/PLAN.md).

**Por quê**: com só as ~113 amostras que o projeto tinha, uma rede neural do zero
overfita e perde pro kNN atual. Pré-treinar num dataset gravado com o mesmo Myo dá à
rede uma noção geral de "como é ativação muscular via EMG de superfície" antes de
especializar no seu braço.

## Passo a passo do pipeline

```bash
# 1. Baixar o EMG-EPN612 (~5.5GB, CC-BY 4.0, acesso aberto)
python -m nn_classifier.data.download_epn612

# 2. Conferir quantas amostras/janelas voce ja tem gravadas por classe
python -m nn_classifier.data.check_own_data

# 3. Se faltar alguma classe (hoje falta "rest" e "pointing"), grave --
#    veja o passo a passo de coleta abaixo -- e confira de novo com o passo 2

# 4. Pretrain no EPN612 (precisa de GPU pra ser rapido; ~612 sujeitos)
python -m nn_classifier.train --stage pretrain

# 5. Fine-tune nos seus dados, a partir do checkpoint do pretrain
python -m nn_classifier.train --stage finetune \
    --pretrained nn_classifier/checkpoints/epn612_pretrained.pt

# 6. Comparar contra o kNN atual, numeros lado a lado
python -m nn_classifier.evaluate \
    --pretrained nn_classifier/checkpoints/epn612_pretrained.pt \
    --finetuned nn_classifier/checkpoints/finetuned.pt

# 7. Testar ao vivo com o Myo fisico
python -m nn_classifier.live_infer --checkpoint nn_classifier/checkpoints/finetuned.pt
```

`torch` não está no `requirements.txt` raiz por padrão — instale a build certa pra sua
GPU pelo [seletor oficial do PyTorch](https://pytorch.org/get-started/locally/) antes
do passo 4 (uma build CPU também funciona, só que bem mais lenta pra treinar nos 612
sujeitos).

## Passo a passo de coleta (dados próprios)

A coleta usa a ferramenta que já existe no projeto — [`src/emgGestureTrainer.py`](../src/emgGestureTrainer.py) —
não é preciso nenhum script novo. Ela grava em `src/data/vals{classe}.dat`, o mesmo
lugar que o `hand3d`/sistema ao vivo já lê.

1. **Conecte o Myo** via Bluetooth, do jeito de sempre (ver [`hand3d/README.md`](../hand3d/README.md) se tiver dúvida de pareamento).

2. **Rode o script de dentro de `src/`, não da raiz do repo:**
   ```bash
   cd src
   python emgGestureTrainer.py
   ```
   Isso importa porque o script grava em `data/vals{classe}.dat` — um caminho
   relativo ao diretório onde ele roda. Se rodar da raiz (`python src/emgGestureTrainer.py`,
   como o README principal sugere), ele vai procurar/criar uma pasta `data/` na
   raiz do repo em vez de usar `src/data/`, e o resto do pipeline (`check_own_data.py`,
   `train.py --stage finetune`) não vai achar essas amostras.

3. Uma janela abre com um placar por classe (0–9) e o LED do Myo fica colorido.
   **Vista o armband sempre na mesma posição/orientação** (mesma altura no
   antebraço, mesma rotação do logo) em toda sessão de gravação — isso importa mais
   do que parece: o classificador aprende o padrão espacial entre os 8 canais, e se
   o armband girar entre sessões os canais deixam de corresponder aos mesmos músculos.

4. Para cada gesto, **pressione e mantenha pressionada** a tecla numérica
   correspondente enquanto faz e sustenta o gesto; solte pra descansar entre
   repetições. Prefira várias repetições curtas (3–5s, com pausa entre elas) a uma
   tecla pressionada continuamente por 30s — a variação natural entre repetições
   (leve fadiga, reposicionamento) ajuda o modelo a generalizar.

   | Tecla | Classe | Gesto | Amostras hoje |
   |:-:|:-:|---|--:|
   | `0` | rest | mão relaxada, parada | **0** — prioridade |
   | `1` | open | mão aberta ("Relaxed") | 259 |
   | `2` | fist | mão fechada | 315 |
   | `3` | spock | saudação vulcana | 402 |
   | `4` | pointing | apontando | **0** — prioridade |

5. **Priorize `rest` (0) e `pointing` (4)** — hoje têm zero amostras. O 1-NN atual
   contorna isso com um artifício (só "aceita" alguma classe depois de um número
   mínimo de amostras totais, senão sempre devolve 0); uma rede neural não tem esse
   comportamento de fallback — se `rest` nunca foi visto, ela vai forçar a
   classificação em open/fist/spock mesmo com o braço parado. Grave essas duas até
   chegar numa ordem de grandeza parecida com as outras (algumas centenas de
   amostras / algumas dezenas de repetições curtas).

6. Depois de gravar, confirme as contagens:
   ```bash
   python -m nn_classifier.data.check_own_data
   ```

7. Cuidado com duas teclas do próprio `emgGestureTrainer.py`: `e` **apaga os dados
   de TODAS as classes** (não só da que você está gravando); `r` recarrega do disco.
   `store_data` só *acrescenta* ao arquivo (`ab`), então rodar o script de novo em
   dias diferentes vai somando às gravações anteriores sem apagar nada.

## Limitações conhecidas (ver PLAN.md para detalhes)

- O EMG-EPN612 cobre `rest/open/fist/waveIn/waveOut/pinch` — **não** tem `spock` nem
  `pointing`. Essas duas classes só melhoram com dados próprios; pré-treino não ajuda.
- **Domain shift**: o EPN612 foi gravado em outros braços/posicionamentos de
  eletrodo. `evaluate.py` mede isso explicitamente (zero-shot vs. fine-tuned) em vez
  de assumir que pré-treino = melhoria automática.
- `myTry/` usa uma numeração de gestos **diferente e incompatível** com
  `src/data/vals*.dat` (ex.: lá `1`=Close, aqui `1`=open) — não há mapeamento
  documentado entre as duas. Por isso este módulo segue só o esquema de
  `src/data/vals*.dat` (o mesmo que o sistema ao vivo usa), e `evaluate.py` treina o
  kNN de comparação do zero no mesmo split, em vez de reusar
  `myTry/knn_gesture_classifier.joblib`.
