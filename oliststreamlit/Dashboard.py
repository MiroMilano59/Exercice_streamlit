import streamlit as st
import pandas as pd
import sqlite3

connection = sqlite3.connect("olist.db")

df_reviews = pd.read_sql_query("SELECT * FROM CleanDataset",connection)

connection.close()

st.set_page_config(page_title='Olist Dataset')
st.header('Olist Machine learning project')
st.markdown('Deployment of the Olist dataset machine learning model.')
st.markdown('use this dashboard to understand the data and to make predictions')
st.markdown('')
st.image('Schema_Olist.png')