
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import tree
basePrevendo= pd.read_csv("NovosClientes.csv") #Importando a base


Separando = basePrevendo.iloc[:, 1:4].values #Separando
scaler_credit = StandardScaler()
NovosClientes = scaler_credit.fit_transform(Separando) #Transformando de -1 até 1


from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pickle
with open('credit.pkl', 'rb') as f:  
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)
arvore_credit = DecisionTreeClassifier(criterion='entropy', random_state = 0)
arvore_credit.fit(X_credit_treinamento, y_credit_treinamento)
PagaOuNao = arvore_credit.predict(NovosClientes)
print (PagaOuNao)
previsores = ['income', 'age', 'loan']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (20,20))
tree.plot_tree(arvore_credit, feature_names=previsores, class_names=['0','1'], filled=True)
fig.savefig('arvore_credit.png')