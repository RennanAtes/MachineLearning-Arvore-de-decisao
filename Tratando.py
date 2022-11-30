import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

AbrirGrafico = True
VizualizarGrafico1 =True
VizualizarGrafico2 =True
VizualizarGrafico3 =True
VizualizarGrafico4 =True
base_credit = pd.read_csv("D:\Trabalho\credit_data.csv") #Importando a base
#Gerando o gráfico para vizualizar os dados.
if AbrirGrafico:
    grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color = 'default')
    grafico.show()

#Possui valores negativos, então precisamos localizar.
AInegativa = base_credit.loc[base_credit['age'] < 0]
Inegativo = base_credit.loc[base_credit['income'] < 0]
Lnegativo = base_credit.loc[base_credit['loan'] < 0]

print ('=============================')
print (f'Idade negativa:{AInegativa}')
print ('=============================')
print (f'Economia negativa:{Inegativo}')
print ('=============================')
print (f'Divida negativa:{Lnegativo}')

#Aqui eu vou optar por calculiar o valor médo, e substituir os dados negativo pelo valor médio.
#Calculando o valor medio e colocando em uma variavel
basecredit3 = base_credit.drop(base_credit[base_credit['age'] < 0 ].index)
valormedio = basecredit3['age'].mean()
#Aplicando o valor medio nas variavel negativas
base_credit.loc[base_credit['age'] < 0, 'age'] = valormedio
#Vizualizando
print (base_credit.loc[base_credit['clientid'].isin([16,22,27])])

if AbrirGrafico:
    grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color = 'default')
    grafico.show()

#Verificando se existe algum numero faltante

print (base_credit.isnull().sum())
# Três idades faltantes
#Verificando as idades faltantes
print (base_credit[pd.isnull(base_credit['age'])])
#Definindo as idades faltantes para a idade media.
base_credit['age'].fillna(base_credit['age'].mean(), inplace = True)
#Vizualizando.
print (base_credit.loc[base_credit['clientid'].isin([29,31,32])])


for n in range (0,3):
    print ('Calculando a chance da pessoa pagar a divida')

X_credit = base_credit.iloc[:, 1:4].values #Separando

Y_credit = base_credit.iloc[:, 4].values #Separando

print (X_credit) #Verificando
print (Y_credit) #Verificando
print ("============")
print ("Renda Anual:", X_credit[:,0])
print ("Idade:", X_credit[:,1] )
print ("Divida:", X_credit[:,2])
print ("============")

#Padronizando os dados.
#Padronização para OutLiers, muito fora do padrão.
#Normalização

from sklearn.preprocessing import StandardScaler

scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit) #Transformando de -1 até 1

print ("Renda Anual:", X_credit[:,0])
print ("Idade:", X_credit[:,1] )
print ("Divida:", X_credit[:,2])

print (X_credit)

#Arvore de decisão.
#Criando a base de testes
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(X_credit, Y_credit, test_size= 0.25, random_state=0) #Random = Valores fixos sempre
#test_size = Tamanho da arvore de teste = 0.25% ou 75% De arvore real e 25 de teste

print (X_credit_treinamento.shape)

print (y_credit_treinamento.shape)

print (X_credit_teste.shape, y_credit_teste.shape)

import pickle 

with open('credit.pkl', mode='wb') as f: #Salvando a base de treinamento com 500 registros.
    pickle.dump([X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste], f)

if VizualizarGrafico1: #Apenas vizualizando os graficos
    sns.countplot(x = base_credit['default']) # a contagem de quantos registros que tem por 0 - 1
    plt.show()

if VizualizarGrafico2:
    plt.hist(x = base_credit['age']) #olhando a quantidade de pessoas em idade media
    plt.show()
    print ('==============================')

if VizualizarGrafico3:
    plt.hist(x = base_credit['income']) #pessoas e renda minima por ano.
    plt.show()

if VizualizarGrafico4:
    plt.hist(x = base_credit['loan'])
    plt.show()

grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color = 'default')
grafico.show()
