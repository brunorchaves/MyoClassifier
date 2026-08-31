# Resumos dos artigos

Confirmação: os 7 arquivos baixados em `papers/` cobrem exatamente os 7
artigos listados no [README.md](README.md) — nada faltando, nada extra.

| Arquivo | Artigo |
|---|---|
| `3742471.pdf` | Myoelectric Prosthetic Hands (review, ACM CSUR) |
| `s41586-025-09255-w.pdf` | Meta Neural Band (Nature) |
| `2410.23986v1.pdf` | HD-EMG + Deep Learning (kinematics + força) |
| `s12984-021-00832-4.pdf` | MRL — controle simultâneo/proporcional |
| `Chen_2021_J._Neural_Eng._18_056010.pdf` | Decodificação de unidades motoras em tempo real |
| `file.pdf` | CNN vs SVM em controle mioelétrico |
| `aer.pdf` | Swarm-Contrastive Decomposition (J. Physiol.) |

---

## 1. Myoelectric Prosthetic Hands: A Review of Muscle Synergy, Machine Learning and Edge Computing

**Farag, Gaber, Awad, Elhady — ACM Computing Surveys, 2025.**
DOI: [10.1145/3742471](https://doi.org/10.1145/3742471)

Review guarda-chuva sobre a última década de controle de próteses mioelétricas.
Não traz experimento novo, mapeia o campo em três eixos: (1) extração de
sinergias musculares para reduzir a dimensionalidade do sEMG antes de
alimentar o classificador/regressor, (2) evolução dos controladores de EMG
em "gerações" — da simples classificação discreta até controladores de
4ª geração com controle simultâneo e proporcional multi-DoF via deep
learning, e (3) o gargalo prático: esses modelos de deep learning
raramente cabem num microcontrolador embarcado de prótese real, exigindo
técnicas de compressão (pruning, quantização, hyperdimensional computing).

**Achado central para sua pergunta:** os autores apontam HD-EMG combinado
com fusão de sensores (IMU, FMG/força-miografia) como a rota mais
promissora para controle robusto e generalizado de próteses — mas
constatam que a maioria dos estudos não reporta especificações de
hardware suficientes para saber se o modelo proposto rodaria em tempo
real num dispositivo real. Ou seja: a distância entre "acurácia em
paper" e "controle usável no dia a dia" ainda é, em boa parte, uma
lacuna de engenharia de sistemas embarcados, não só de algoritmo.

---

## 2. A generic non-invasive neuromotor interface for human-computer interaction

**Kaifosh, Reardon & CTRL-labs at Reality Labs (Meta) — Nature, 2025.**
DOI: [10.1038/s41586-025-09255-w](https://doi.org/10.1038/s41586-025-09255-w)

Base científica do **Meta Neural Band**. Um bracelete de eletrodos secos no
pulso (dry-electrode sEMG-RD, 2kHz, 2.46 µVrms de ruído) treinado com dados
de **6.627 participantes** para produzir modelos *genéricos* (sem
calibração por pessoa) capazes de decodificar: controle de cursor 1D
contínuo pela postura do pulso, gestos discretos (toques/swipes de
polegar, pinças de dedo) e escrita à mão imaginária.

**Números concretos:**
- Classificação offline de gestos e handwriting: **>90%** de acurácia
  held-out (sem calibração pessoal).
- Erro angular de velocidade do pulso: **<13°/s**.
- Em uso fechado (closed-loop) real: 0,66 aquisições de alvo/s em
  navegação contínua por pulso (vs. 0,96–1,01 de trackpad/motion-capture
  como "gold standard"), 0,88 detecções de gesto/s em navegação discreta,
  e **20,9 palavras/minuto** de escrita à mão decodificada (vs. 25,1 WPM
  escrevendo normalmente no papel).
- Personalizar o modelo genérico com apenas **20 minutos** de dados do
  usuário reduz o erro de caractere em ~16% adicional.

**Relevância:** é hoje o sistema sEMG de superfície não-invasivo com
melhor generalização *entre pessoas* sem calibração — mas atua sobre
punho/gestos de mão inteira, não sobre controle independente de cada
dedo individualmente com a fidelidade de uma mão biológica.

---

## 3. Simultaneous Control of Human Hand Joint Positions and Grip Force via HD-EMG and Deep Learning

**Rahimi, Badamchizadeh, Sîmpetru, Ghaemi, Eskofier, Del Vecchio — arXiv 2410.23986, 2024.**

Usa 5 grids HD-sEMG (320 canais) no antebraço + um modelo 3DCNN-MLP para
estimar **simultaneamente 20 posições de junta da mão (3D) e a força de
pinça** — 21 graus de liberdade ao mesmo tempo — durante 2pinch, 3pinch e
fechamento de punho, em 9 sujeitos.

**Números concretos:**
- Cinemática: distância euclidiana média de **11,01 ± 2,22 mm** por
  junta; correlação (PCC) de **0,98**.
- Força: erro médio absoluto de **0,8 ± 0,33 N** offline e correlação de
  **0,97**; em tempo real o erro sobe para **2,09 ± 0,9 N** (PCC 0,92) —
  a própria equipe atribui a queda a deslocamento do eletrodo entre
  sessões, não a limite do modelo.
- Supera trabalhos anteriores comparáveis (Li et al. e Mao et al.), que
  só tratavam ~2–4 DoF, alcançando PCC de 97,8% (cinemática) e 97,2%
  (força) com 21 DoF.

**Relevância:** é o melhor exemplo concreto de controle *contínuo e
simultâneo de força + postura de todos os dedos ao mesmo tempo* — ainda
em laboratório, com grids fixados manualmente e sem amputados testados.

---

## 4. Learning regularized representations of categorically labelled surface EMG enables simultaneous and proportional myoelectric control

**Olsson, Malešević, Björkman, Antfolk — J. NeuroEngineering and Rehabilitation, 2021.**
DOI: [10.1186/s12984-021-00832-4](https://doi.org/10.1186/s12984-021-00832-4)

Propõe o método **MRL** (Myoelectric Representation Learning): uma rede
neural multitarefa que aprende, a partir de apenas 8 canais de sEMG do
Myo armband e rótulos categóricos de calibração, um mapeamento *contínuo*
para 2 DoF (flexão/extensão de punho e abertura/fechamento da mão),
permitindo controle simultâneo e proporcional — não apenas classificação
de poses discretas.

**Números concretos:**
- Comparado ao padrão comercial (LDA/pattern recognition), o MRL venceu
  em **5 de 5** métricas de controle em tempo real (teste estilo Fitts'
  law), com effect size de d=0,62 a 1,13.
- Nenhuma deterioração significativa após **7 dias** sem recalibração —
  ou seja, robustez temporal razoável nesse intervalo curto.
- Acurácia de classificação de validação (referência): 92,71% ± 4,05%.

**Relevância direta ao seu projeto:** usa exatamente o **Myo armband
(8 canais)** — o mesmo hardware do MyoClassifier — mas troca o
classificador discreto por um regressor contínuo multitarefa, mostrando
que dá para extrair controle proporcional/simultâneo do Myo sem precisar
de HD-sEMG, desde que o classificador seja substituído por uma rede que
aprenda representações contínuas.

---

## 5. Simultaneous and proportional control of wrist and hand movements by decoding motor unit discharges in real time

**Chen, Yu, Sheng, Farina, Zhu — Journal of Neural Engineering, 2021.**
DOI: [10.1088/1741-2552/abf186](https://doi.org/10.1088/1741-2552/abf186)

Generaliza a decomposição de sEMG em unidades motoras (MUs) — até então só
offline — para **tempo real**, usando 128 canais HD-sEMG do antebraço.
Agrupa os disparos de MUs por tarefa motora (extensão, flexão, pronação,
supinação, grasp) e usa a taxa de disparo cumulativa de cada grupo como
sinal de controle contínuo para mover um cursor 2D, comparando contra os
métodos convencionais NMF (sinergias) e MLR (regressão linear).

**Números concretos:**
- Decompôs em média **>20 MUs em tempo real** com acurácia estimada
  **>85%** (PNR médio >29 dB).
- Correlação entre disparo cumulativo e ativação do movimento: **R = 0,93
  ± 0,05**.
- No teste online multi-DoF (Exp2), o método baseado em MUs superou NMF e
  MLR em **todas** as métricas (taxa de conclusão, tempo, eficiência de
  trajetória, p<0,05); testado também em 2 pacientes com deficiência de
  membro, com taxa de conclusão consistentemente maior que os outros
  métodos.

**Relevância:** é a prova de que decodificar o "comando neural" quase na
fonte (disparo de motoneurônios) dá controle simultâneo mais fiel do que
os métodos clássicos baseados em envelope de EMG — inclusive em
pacientes reais, não só sujeitos sãos.

---

## 6. Real-time, simultaneous myoelectric control using a convolutional neural network

**Ameri, Akhaee, Scheme, Englehart — PLOS ONE, 2018.**
DOI: [10.1371/journal.pone.0203835](https://doi.org/10.1371/journal.pone.0203835)

Testa se uma CNN alimentada com sEMG **cru** (sem engenharia de features)
consegue igualar um SVM clássico (alimentado com features de domínio de
tempo + frequência) em controle mioelétrico simultâneo de punho, em teste
Fitts' law em tempo real.

**Números concretos:**
- Nenhuma diferença estatística (p>0,05) entre CNN e SVM em nenhuma das 4
  métricas de controle (throughput 0,36 vs 0,35 bits/s; completion rate
  100% ambos; overshoot 0,98 vs 1,00; path efficiency 91,7% vs 91,0%).
- Relação tempo-de-movimento vs. índice-de-dificuldade obedece a Lei de
  Fitts com R² > 0,98 para ambos os métodos.

**Relevância:** é um dos primeiros a validar CNN *end-to-end* em EMG bruto
num teste **online fechado** (não só offline) — mostra que deep learning
consegue extrair sozinho a informação que antes exigia features
projetadas manualmente, sem ganho nem perda de desempenho nesse caso
específico (2 DoF, sujeitos sãos, Myo/eletrodos convencionais).

---

## 7. Unlocking the full potential of high-density surface EMG: novel non-invasive high-yield motor unit decomposition

**Grison, Mendez Guerra, Clarke, Muceli, Ibáñez, Farina — Journal of Physiology, 2025.**
DOI: [10.1113/JP287913](https://doi.org/10.1113/JP287913)

Propõe o **SCD** (Swarm-Contrastive Decomposition): um algoritmo que ajusta
dinamicamente a função de contraste usada para separar disparos de
unidades motoras individuais a partir de HD-sEMG, em vez de usar um
expoente fixo como os métodos anteriores (ex. cBSS). É o paper de método
mais recente (2025) por trás da linha de pesquisa de decomposição de
Farina citada no artigo #5.

**Números concretos:**
- Em diferentes níveis de excitação simulados: SCD detectou em média
  **25,9 ± 5,8** MUs vs. **13,9 ± 2,7** do método de referência
  (praticamente o dobro).
- Em condições de alta sincronização (contrações balísticas): **31,2 ±
  4,3** MUs (SCD) vs. **10,5 ± 1,7** (baseline) — quase o triplo.
- Validado experimentalmente contra EMG intramuscular (padrão-ouro):
  taxa de acordo (RoA) do SCD consistentemente ≥ a do método anterior
  (cBSS), incluindo em sinais de mulheres, historicamente mais difíceis
  de decompor (~50% mais MUs decompostas nesses casos).

**Relevância:** representa o estado da arte atual em **quantos**
comandos motores individuais conseguimos "ouvir" de fora da pele — quanto
mais MUs decompostas e com mais precisão, mais fina pode ser, em teoria,
a decodificação de intenção por dedo. É método/validação de decomposição,
não um sistema de controle em si — é a base que alimenta trabalhos como o
#5.

---

## Síntese

Do mais simples (compatível com o que o MyoClassifier já usa) ao mais
avançado:

1. **#6** mostra que trocar features manuais por CNN não muda o resultado
   com poucos canais — não é ali que está o ganho.
2. **#4** é o upgrade mais direto e barato para o projeto: mesmo hardware
   (Myo, 8 canais), troca só o classificador discreto por um regressor
   contínuo (MRL) e já ganha controle proporcional/simultâneo.
3. **#1** dá o panorama de para onde a área está indo (sinergias, edge
   computing, fusão de sensores) e por que HD-sEMG é considerado o
   caminho mais promissor.
4. **#3** e **#5** mostram o que HD-sEMG com deep learning ou decomposição
   de unidades motoras já conseguem hoje: 21 DoF simultâneos com erro de
   ~11mm por junta, ou controle multi-DoF com R>0,93 decodificando o
   disparo neural direto.
5. **#7** é o limite metodológico atual de quantos comandos motores
   individuais se consegue extrair do sEMG de superfície.
6. **#2** é o único dos sete que já é produto real, validado em milhares
   de pessoas sem calibração — mas ainda em nível de gesto/pulso, não de
   dedo-a-dedo independente.
