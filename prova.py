import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# Prova1:2h
# 1) Lettura dataset
# 2) EDA - plot 
# 3) Correlation matrix
# 4) fit modello -  valutazione errori
# 5) salvataggio modello

# Prova2:2h
# 1) creazione repository github
# 2) creazione frontend streamlit 
# 3) visualizzazione di correlation matrix su streamlit
# 4) Inference del modello salvato su file
# 5) deployment CI/CD streamlit.io

def main():
        

        st.title("Prova 03.05")
        path = "https://frenzy86.s3.eu-west-2.amazonaws.com/python/data/formart_house.csv"
        dataf = pd.read_csv(path)
        dataf = dataf.iloc[:-1].astype(float) 

    


        fig = plt.figure(figsize = (10, 8))
        plt.title("prova")
        sns.heatmap(dataf.corr(), annot=True, cmap="Greens")
        st.pyplot(fig)

        
        crim=st.slider('inserisci crim',1,10000,500)
        zn=st.slider('inserisci zn',1,10000,500)
        indus=st.number_input('inserisci indus',1,10000,500)
        chas=st.slider('inserisci indus',0,10,0)
        nox=st.number_input('inserisci nox',0.0,1.0,0.5)
        rm=st.slider('inserisci rm',1,10,5)
        age=st.number_input('inserisci age',18,100,30)
        dis=st.number_input('inserisci dis',1,10,2)
        rad=st.number_input('inserisci rad',1,3,1)
        tax=st.number_input('inserisci tax',1,500,100)
        ptratio=st.slider('inserisci ptratio',1,50,10)
        b=st.slider('inserisci b',1,300,100)
        lstat=st.number_input('inserisci istat',1,10,5)

        newmodel = joblib.load('reg_prova.pkl')

        pred = newmodel.predict([[crim,	zn,indus,chas,nox,rm,age,dis,rad,tax,ptratio,b,lstat]])
        st.write(f"risultato: ${round(pred[0],2)}") 


#     administration = st.slider('inserisci spesa adm',0, 160000, 50000) 

#     marketing= st.slider('inserisci spesa mrktng',0, 160000, 50000)



if __name__ == "__main__":
    main()