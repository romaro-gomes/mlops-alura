import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Treinando o modelo
df=pd.read_csv('casas.csv')

colunas =['tamanho','ano','garagem']

X=df.drop('preco',axis=1)
y=df['preco']

X_train, X_test, y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42
)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

pickle.dump(modelo,open('modelo.sav', 'wb'))