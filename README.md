# machine_learn_0526

Projeto dedicado ao estudo e implementação do **MLP (Multi-Layer Perceptron)**, um tipo de rede neural artificial com múltiplas camadas ocultas capaz de resolver problemas de classificação não linearmente separáveis. O projeto implementa um classificador MLP completo "from scratch" utilizando **PyTorch**, explorando arquitetura de rede, funções de ativação, regularização e early stopping.

## Conteúdo

- **mlp.ipynb**: Notebook principal contendo a implementação completa da classe `MLPClassifier`. O projeto abrange:

  - **Construtor da Classe**: Configuração de hiperparâmetros como número de camadas ocultas (`hidden_layers`), função de ativação (`relu`/`tanh`), taxa de aprendizado, épocas, patience para early stopping, batch size, otimizador (`sgd`/`adam`), regularização (`dropout`/`l2`) e semente aleatória para reprodutibilidade. Detecção automática de GPU (CUDA) com fallback para CPU.
  
  - **Arquitetura do Modelo** (`_build_model`): Construção dinâmica das camadas usando `nn.Sequential`, intercalando camadas `nn.Linear` com funções de ativação (`nn.ReLU` / `nn.Tanh`) e camadas de `nn.Dropout` quando a regularização é ativada. Suporte a dispositivos GPU/CPU.
  
  - **Treinamento** (`fit`): Implementação completa do treinamento com mini-batches, incluindo:
    - Conversão dos dados para tensores PyTorch
    - Seleção automática de `CrossEntropyLoss` (multiclasse) ou `BCEWithLogitsLoss` (binário)
    - Suporte a SGD e Adam (com weight decay para regularização L2)
    - Cálculo de loss e acurácia por época
    - Validação com early stopping: interrompe o treinamento se a loss de validação não melhorar após `patience` épocas
    - Logs de progresso a cada 10 épocas com métricas de treino e validação
  
  - **Predição** (`predict` / `predict_proba`): Métodos para classificação e probabilidades das classes.
  
  - **Score**: Cálculo de acurácia do modelo.
  
  - **Testes com diferentes cenários**: Experimentos variando arquiteturas, otimizadores e regularizações para análise de convergência.

- **train_dataset.csv** (769 linhas): Dataset principal de treinamento com 50 features numéricas e label de classe (0, 1, 2, 3 — classificação multiclasse).

- **test_dataset.csv** (193 linhas): Dataset de teste para avaliação da generalização do modelo.

- **train_dataset[1-3].csv** / **test_dataset[1-3].csv**: Datasets menores para testes rápidos de convergência em diferentes cenários de complexidade.

## Tecnologias e Bibliotecas

As ferramentas essenciais deste projeto são:

- **Python 3**: Linguagem de programação.
- **PyTorch** (`torch`, `torch.nn`, `torch.optim`): Framework principal para construção e treinamento da rede neural. Utiliza tensores, autograd para backpropagation, e suporte nativo a GPU CUDA.
- **NumPy**: Processamento e manipulação dos datasets antes da conversão para tensores.
- **Matplotlib**: Plotagem de gráficos de evolução da loss e acurácia ao longo das épocas.
- **CSV**: Leitura dos datasets de treino, teste e validação.
- **Jupyter Notebook**: Ambiente interativo para desenvolvimento, experimentação e documentação.

## Como começar

1. Clone este repositório:
   ```bash
   git clone https://github.com/smokeeaasd/mlp.git
   ```

2. Instale as bibliotecas necessárias:
   ```bash
   pip install numpy matplotlib torch
   ```

3. Execute o notebook:
   - Abra o `mlp.ipynb`.
   - Experimente diferentes arquiteturas alterando o parâmetro `hidden_layers` (ex: `[64, 32]`, `[128, 64, 32]`).
   - Compare o desempenho entre os otimizadores SGD e Adam.
   - Ative a regularização Dropout ou L2 e observe o impacto na generalização.
   - Analise os gráficos de loss e acurácia para identificar overfitting e convergência.


