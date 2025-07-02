# 🏈 NFL Yard Prediction

Este projeto tem como objetivo prever a quantidade de jardas que uma jogada da NFL resultará, utilizando dados de tracking (movimentação dos jogadores em campo) e aprendizado de máquina.

## 📁 Estrutura do Projeto

```
nfl-prediction/
├── app/                      # Scripts de animação e execução
├── data/                     # Dados (tracking, plays, etc.)
├── models/                   # Modelos treinados (.pkl)
├── src/                      # Módulos do pipeline (data_loader, preprocess, model)
├── train.py                 # Script principal de treino
├── requirements.txt         # Dependências
└── README.md                # Este arquivo
```

## 🚀 Como Rodar

### 1. Clone o repositório

```bash
git clone https://github.com/brendongabriel/nfl-prediction.git
cd nfl-prediction
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Adicione os dados

Crie a pasta `data/` e adicione os arquivos:

- `tracking_week_1.csv`
- `plays.csv`

> Os dados podem ser obtidos através do [NFL Big Data Bowl (Kaggle)](https://www.kaggle.com/competitions/nfl-big-data-bowl-2021).

### 4. Treine o modelo

```bash
python train.py
```

O modelo será salvo em `models/trained_model.pkl`.

### 5. Visualize a animação de uma jogada

```bash
python app/animate_play.py
```

O script seleciona uma jogada aleatória e exibe os jogadores se movendo, com a predição de jardas destacada.

## 🧠 Modelo

- Utiliza um modelo XGBoost regressivo
- Features incluem: posição média dos jogadores, velocidade, aceleração, orientação, direção e distância ao QB no snap
- O modelo espera 22 jogadores × 7 features por jogada

## 🎨 Animação

- Visualização do campo em tempo real
- Cores diferentes para cada time
- Números das camisas sobre os jogadores
- A bola aparece como um ponto **preto**
- Predição de jardas exibida no topo

## 🧾 Licença

Este projeto é de uso educacional e de pesquisa. Os dados utilizados são de domínio público fornecidos pela NFL via Kaggle.

---

📬 Dúvidas ou sugestões? Fique à vontade para abrir uma issue ou contribuir!
