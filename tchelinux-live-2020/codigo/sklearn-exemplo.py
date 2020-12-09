import pandas as pd

sim_nao = lambda x: 1 if (x == "yes") else 0
masculino_feminino = lambda x: 1 if (x == "Male") else 0

converter_dict = {'Gender':masculino_feminino, 'family_history_with_overweight':sim_nao, 'FAVC':sim_nao, 'SMOKE':sim_nao, 'SCC':sim_nao}

dados = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv", converters=converter_dict)

rows = []
for column in dados.columns:
  row = {'coluna': column, 'nans': dados[column].isnull().sum(), 'frac_nans': dados[column].isnull().sum() / dados.shape[0]}
  rows.append(row)
res = pd.DataFrame(rows)
res[res.nans>0].sort_values('nans', ascending=False)
print(res)

from sklearn.preprocessing import LabelBinarizer

def binarize_coluna(dados, nome_coluna):
        binarizador = LabelBinarizer()
        binarizador.fit(dados[nome_coluna])

        # Com o codificador "treinado", convertemos os dados na coluna desejada
        atributo_transformado = binarizador.transform(dados[nome_coluna])

        # Agora reconvertemos para um dataframe em pandas 
        # Mudamos o nome das colunas para refletir os valores originais
        onehot = pd.DataFrame(atributo_transformado)
        
        #onehot.columns = binarizador.classes_
        onehot.columns = pd.Series(binarizador.classes_).apply(lambda x: nome_coluna + '_' + x)

        # Juntamos de volta com os dados originais e removemos a coluna categórica
        dados_preparados = pd.concat([dados, onehot], axis=1).drop([nome_coluna], axis=1)
        return dados_preparados

colunas_a_serem_binarizadas=['CAEC', 'CALC', 'MTRANS']
for col in colunas_a_serem_binarizadas:
    dados = binarize_coluna(dados, col)

# O método train_test_split será utilizado para separação dos conjuntos
from sklearn.model_selection import train_test_split

treino, teste = train_test_split(dados, test_size=0.2)
print("Tamanhos dos conjuntos (treino, teste): ", len(treino), len(teste))

"""Definindo a coluna HOSPITALIZADO como o atributo alvo."""

alvo ='NObeyesdad'
X_treino = treino.drop(alvo, axis='columns')
y_treino = treino[alvo]

X_teste = teste.drop(alvo, axis='columns')
y_teste = teste[alvo]

from sklearn.tree import DecisionTreeClassifier

modelo = DecisionTreeClassifier(random_state=0, criterion="entropy")

# A função fit recebe como primeiro parâmetro uma matriz CxN, com C colunas e N linhas, onde cada linha especifica um exemplo
# O segundo parâmetro é um vetor com N posições, indicando os rótulos das linhas da matriz no primeiro parâmetro

print("Treinando o modelo")
modelo_treinado = modelo.fit(X_treino, y_treino)

#Acurácia
from sklearn.metrics import accuracy_score
#Relatório de classificação
from sklearn.metrics import classification_report
#Matriz de confusão
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

predicoes = modelo_treinado.predict(X_teste)
acuracia = accuracy_score(y_teste, predicoes)
print("Acurácia do modelo: ", acuracia)

relatorio = classification_report(y_teste, predicoes)
print(relatorio)

plot_confusion_matrix(modelo_treinado, X_teste, y_teste)

plt.show()

