
import streamlit as st
import pandas as pd

def app():
    st.title('Data')

    st.write("This is the `Data` page of the multi-page app.")

    st.write("The following is the DataFrame of the `iris` dataset.")

    new_movieDF = pd.read_csv("./FYP_file.csv")


    st.write(new_movieDF)