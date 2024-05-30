# Importando as bibliotecas necessárias
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Passo 1: Preparação dos Dados

# Carregar o dataset MNIST
mnist = fetch_openml('mnist_784', version=1)

# Separar os dados de entrada (X) e as labels (y)
X, y = mnist.data, mnist.target

# Dividir o conjunto de dados em treinamento e teste (80% treinamento, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Passo 2: Arquitetura da Rede

# Definir a arquitetura da rede neural
mlp = MLPClassifier(hidden_layer_sizes=(20, 10), activation='relu', random_state=42)

# Passo 3: Treinamento do Modelo

# Treinar o modelo
mlp.fit(X_train, y_train)

# Passo 4: Avaliação do Modelo

# Avaliar o modelo utilizando o conjunto de teste
y_pred = mlp.predict(X_test)

# Criar um relatório de classificação
report = classification_report(y_test, y_pred)

print("Relatório de Classificação:")
print(report)

# Plotar as curvas de aprendizado
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(mlp.loss_curve_)
plt.title('Curva de Perda')
plt.xlabel('Épocas')
plt.ylabel('Perda')

plt.subplot(1, 2, 2)
plt.plot(mlp.validation_scores_)
plt.title('Curva de Precisão')
plt.xlabel('Épocas')
plt.ylabel('Precisão')

plt.tight_layout()
plt.show()
