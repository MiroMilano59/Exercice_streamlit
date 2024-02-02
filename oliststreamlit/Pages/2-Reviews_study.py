import plotly.express as px
import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from outils import import_clean_dataset

connection = sqlite3.connect("olist.db")

df_reviews = pd.read_sql_query("SELECT * FROM CleanDataset",connection)

connection.close()

st.header('Analyse des notes de reviews')

fig = px.histogram(df_reviews, x='review_score', title='Distribution des notes attribuées aux commandes de 1 à 5',
                   labels={'review_score': 'Note obtenue', 'count': 'Fréquence'})

st.plotly_chart(fig)

st.markdown('')
st.markdown('Si on passe toutes les notes de 1 à 4 en 0 et celle de 5 en 1')
st.markdown('On obtient ce nouveau graphique:')

df_reviews['score'] = df_reviews['review_score'].apply(lambda x: 1 if x == 5 else 0)

color_discrete_map = {0: 'blue', 1: 'orange'}
fig2 = px.histogram(df_reviews, x='score', title='Distribution des notes attribuées aux commandes regroupant les notes de 1 à 4 en une seule bar',
                    labels={'score': 'Note obtenue'},
                    hover_data=df_reviews.columns,
                    color='score',
                    color_discrete_map=color_discrete_map)

st.plotly_chart(fig2)

st.header('Analyse des variables de la table Orders')
st.markdown('Analyse des temps de livraison')

df_orders = import_clean_dataset()
df_orders['score'] = df_orders['review_score'].apply(lambda x : 1 if x == 5 else 0)
df_orders["temps_livraison"] = (df_orders.order_delivered_customer_date - df_orders.order_purchase_timestamp).dt.days

fig6, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
# Plot the histogram
ax1.hist(df_orders.temps_livraison, bins=100, color='skyblue', edgecolor='black')
# Add labels and title
ax1.set_xlabel('Values')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution de temps de livraison')

# Plot the histogram
ax2 = sns.boxplot(x=df_orders.temps_livraison, color='skyblue')
# Add labels and title
ax2.set_xlabel('Values')
ax2.set_ylabel('Quantiles')
ax2.set_title('Distribution de temps de livraison')

st.pyplot(fig6)

fig7, ax7 = plt.subplots(figsize=(8, 4))

# Plot the boxplot
sns.boxplot(x=df_orders.temps_livraison, hue=df_orders.score, showfliers=False, ax=ax7)

# Add labels and title
ax7.set_xlabel('Temps de livraison')
ax7.set_ylabel('Score')
ax7.set_title('Distribution de temps de livraison')

# Show the plot
st.pyplot(fig7)


st.markdown('Conclusion de l analyse de temps de livraison:')
st.markdown('- il y a des valeurs extremes pour temps de livraison, un travail est potentiellement nécessaire:')
st.markdown('-- Soit binariser la variable (créer des catégories)')
st.markdown('-- Soit plafonner la variable (remplace les valeurs extreme par un maximum qu on défini)')
st.markdown('-- Soit laisser telle quelle la variable')
st.markdown('- Les écarts de moyene et de médiane semblent nous indiquer un fort pouvori explicatif de la variable temps de livraison sur la satisfaction.')

