# Modelo de Previsão de Preços de Casas "HOUSE SCANNER"

## Passo a Passo

### 1. Definição do Problema
- **Objetivo**: Prever o preço de uma casa com base em suas características.
- **Dados**: Utilizamos um dataset fictício com as seguintes features:
  - `tamanho`: Tamanho da casa em metros quadrados.
  - `quartos`: Número de quartos.
  - `localizacao`: Proximidade de pontos de interesse (escala de 1 a 10).

### 2. Escolha do Modelo
- **Algoritmo**: Regressão Linear, adequado para problemas de previsão numérica.
- **Biblioteca**: `scikit-learn` para o  treinamento e avaliação do modelo.

### 3. Treinamento do Modelo
- Dividir os dados em conjuntos de treino e teste.
- Treinar o modelo com os dados de treino.
- Avaliar o desempenho usando métricas como RMSE (Root Mean Squared Error).

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# Dados fictícios
dados = {
    'tamanho': [70, 80, 90, 100, 120],
    'quartos': [2, 2, 3, 3, 4],
    'localizacao': [5, 6, 7, 8, 9],
    'preco': [300000, 350000, 400000, 450000, 500000]
}
df = pd.DataFrame(dados)

# Separar features e target
X = df[['tamanho', 'quartos', 'localizacao']]
y = df['preco']

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Avaliar modelo
y_pred = modelo.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'RMSE: {rmse}')
