
import pickle
from sklearn.tree import DecisionTreeClassifier

with open('D:/Nova pasta/credit.pkl', 'rb') as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)
print (X_credit_treinamento.shape, y_credit_treinamento.shape) #Vizualizando
print (X_credit_teste.shape, y_credit_teste.shape) #Vizualizando
arvore_credit = DecisionTreeClassifier(criterion='entropy', random_state=0)
arvore_credit.fit(X_credit_treinamento, y_credit_treinamento)
print (arvore_credit.fit(X_credit_treinamento, y_credit_treinamento))
previsoes = arvore_credit.predict(X_credit_teste) #Definindo a arvore de teste.
print ("Teste")
print (previsoes) #dados de teste
print ("Real")
print (y_credit_teste) #dados real

from sklearn.metrics import accuracy_score, classification_report

print (accuracy_score(y_credit_teste, previsoes))
print (classification_report(y_credit_teste, previsoes))


from sklearn import tree #Chamando a biblioteca que gera a arvore
import matplotlib.pyplot as plt
previsores = ['income', 'age', 'loan'] #Definindo as classes 'previsores' oq define
fig,axes = plt.subplots(nrows = 1, ncols = 1, figsize =  (20,20))
tree.plot_tree(arvore_credit, feature_names=previsores, class_names=str(arvore_credit.classes_),filled=True)
fig.savefig('arvore_de_credito.png')

print (len(previsoes))
print (len(y_credit_teste))