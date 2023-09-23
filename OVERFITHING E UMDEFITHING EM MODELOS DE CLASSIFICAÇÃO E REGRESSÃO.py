#!/usr/bin/env python
# coding: utf-8

# # CASE DE ESTUDO EXEPLIFICANDO MODELOS OVERFITANDO E UNDERFITANDO EM CLASSIFICAÇÃO E REGRESSÃO

# ## Objetivo: O objetivo deste trabalho é demonstrar um exemplo pratico de como identificar através de métricas e gráficos situações de overtfithing e umderfithing
# 

# In[1]:


# importando bibliotecas

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Dados de exemplo
np.random.seed(0)
X = np.random.rand(10, 1)
y = 2 * X.squeeze() + np.random.rand(10)

# Criar características polinomiais para gerar overfitting
poly_features = PolynomialFeatures(degree=15)
X_poly = poly_features.fit_transform(X)

# Treinar um modelo de regressão linear
model = LinearRegression()
model.fit(X_poly, y)

# Plotar os dados reais e a previsão do modelo
X_test = np.linspace(0, 1, 100)[:, np.newaxis]
X_test_poly = poly_features.transform(X_test)
y_pred = model.predict(X_test_poly)

plt.scatter(X, y, label='Dados reais')
plt.plot(X_test, y_pred, color='red', label='Modelo Overfit')
plt.legend()
plt.title('Overfitting')
plt.show()


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dados de exemplo
X = np.random.rand(10, 1)
y = 2 * X.squeeze() + np.random.rand(10)

# Treinar um modelo de regressão linear simples
model = LinearRegression()
model.fit(X, y)

# Plotar os dados reais e a previsão do modelo
X_test = np.linspace(0, 1, 100)[:, np.newaxis]
y_pred = model.predict(X_test)

plt.scatter(X, y, label='Dados reais')
plt.plot(X_test, y_pred, color='red', label='Modelo Underfit')
plt.legend()
plt.title('Underfitting')
plt.show()


# ## OVERFITHING E UMDEFITHING EM MODELOS DE REGRESSÃO

# Regressão:
# 
# Para problemas de regressão, como prever valores numéricos, você pode usar métricas como o Erro Quadrático Médio (Mean Squared Error - MSE) ou o Erro Absoluto Médio (Mean Absolute Error - MAE).
# 
# Overfitting:
# 
# ### Se o erro no conjunto de treinamento (MSE ou MAE) for significativamente menor do que o erro no conjunto de teste, isso pode indicar overfitting. O modelo está se ajustando muito bem aos dados de treinamento, mas não generaliza bem para novos dados.
# Underfitting:
# 
# ### Se tanto o erro no conjunto de treinamento quanto no conjunto de teste for alto, isso pode indicar underfitting. O modelo não está sendo capaz de capturar a relação nos dados, resultando em desempenho fraco em ambos os conjuntos.

# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Gerar dados de exemplo
import numpy as np
X = np.random.rand(100, 1)
y = 2 * X.squeeze() + np.random.rand(100)

# Dividir os dados em conjunto de treinamento e conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar um modelo de regressão linear simples
model = LinearRegression()
model.fit(X_train, y_train)

# Avaliar no conjunto de treinamento
y_train_pred = model.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)

# Avaliar no conjunto de teste
y_test_pred = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print("Erro Quadrático Médio (MSE) no conjunto de treinamento:", mse_train)
print("Erro Quadrático Médio (MSE) no conjunto de teste:", mse_test)
print("Erro Absoluto Médio (MAE) no conjunto de treinamento:", mae_train)
print("Erro Absoluto Médio (MAE) no conjunto de teste:", mae_test)


# ## OVERFITHING E UMDERFITHING EM MODELO DE CLASSIFICAÇÃO

# ### Se o modelo estiver com acurácia muito alta em dados de treino e baixa no de teste, está overfitando.ou seja, 
# 
# ### Se modelo estiver com acurácia tanto nos dados de treino quanto de teste , está umdefitando, ou seja, não capturando bem os dados.

# In[9]:


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Gerar dados de exemplo com menos recursos informativos
X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_redundant=1, n_repeated=0, random_state=42)

# Dividir os dados em conjunto de treinamento e conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar um modelo de regressão logística simples
model = LogisticRegression()
model.fit(X_train, y_train)

# Avaliar no conjunto de treinamento
y_train_pred = model.predict(X_train)
accuracy_train = accuracy_score(y_train, y_train_pred)
precision_train = precision_score(y_train, y_train_pred)
recall_train = recall_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)

# Avaliar no conjunto de teste
y_test_pred = model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)

print("Acurácia no conjunto de treinamento:", accuracy_train)
print("Precisão no conjunto de treinamento:", precision_train)
print("Recall no conjunto de treinamento:", recall_train)
print("F1-Score no conjunto de treinamento:", f1_train)

print("Acurácia no conjunto de teste:", accuracy_test)
print("Precisão no conjunto de teste:", precision_test)
print("Recall no conjunto de teste:", recall_test)
print("F1-Score no conjunto de teste:", f1_test)


# In[ ]:




