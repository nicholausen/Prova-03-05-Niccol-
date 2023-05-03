import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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

        st.df(dataf)
        st.df(dataf.corr())

        fig = plt.figure(figsize=(10,5))
        sns.heatmap(dataf.corr(), annot=True, cmap="Greens")
        st.pyplot(fig)


if __name__ == "__main__":
    main()