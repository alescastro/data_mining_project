import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# carregaamento do dataset
file_path = "winequality-red.csv"  
wine_data = pd.read_csv(file_path, sep=';') 

print(wine_data.head())

# grafico de distribuicao das qualidades
sns.countplot(x='quality', data=wine_data)
plt.title('Distribuição das Qualidades no Wine Quality Dataset')
plt.show()


wine_numeric_data = wine_data.drop('quality', axis=1) # remoção da coluna quality
sns.heatmap(wine_numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlação')
plt.show()

# segmentação dos dados
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = SVC(kernel='linear', random_state=42)
clf.fit(X_train, y_train)

# bloco de previsão, acuracia e relatorio
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=1)

# printa relatório e acuracia
print(f"Acurácia: {accuracy:.2f}")
print("Relatório de Classificação:\n", report)

# printa matriz de confusão
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine_data['quality'].unique(), yticklabels=wine_data['quality'].unique())
plt.title('Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()
