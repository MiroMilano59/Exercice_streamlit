# Exercice_streamlit

Pour cet exercice j'ai créer un visualisation des données étudiées du projet Olist avec Streamlit.
## Dashboard
Ci-dessous une image du Dashboard dans lequel il est possible d'apercevoir les 4 onglets d'étude du projet.
- Dashboard
- Geolocation
- Reviews study
- Trainning and prediction
![Dashboard](https://github.com/MiroMilano59/Exercice_streamlit/assets/153615242/f2993e7a-2e9e-4a01-b478-2036c9129948)
## Geolocation
Dans l'onglet "Geolocation" il est possible voir en détail la position des acheteurs et vendeurs.
![Geolocation1](https://github.com/MiroMilano59/Exercice_streamlit/assets/153615242/1f0c0825-422c-4db3-bb5f-de0d34f3d596)

En effet, la carte intéractive est réalisée uniquement avec plotly.express, Pandas et Streamlit comme vous pouvez le voir sur le code ci-dessous:
Etant donné le nombre important d'acheteur, j'ai retiré tout doublon à l'aide de la ligne du df_unique_customers.
```
if options == 'Customers':
    df_mergedd = pd.merge(olist_customers_dataset, olist_geolocation_dataset, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='inner')
    df_unique_customers = df_mergedd.drop_duplicates(subset='customer_id', keep='first')
    fig = px.scatter_geo(df_unique_customers,
                        lat='geolocation_lat',
                        lon='geolocation_lng',
                        hover_name='customer_city',
                        projection='natural earth',
                        title='Customers map',
                        color_discrete_sequence=['red'],
                        height=800,
                        width=800)
    
    st.plotly_chart(fig)
```
![Geolocation2](https://github.com/MiroMilano59/Exercice_streamlit/assets/153615242/ed792714-588f-4916-ad76-0c9feeaedba7)

## Reviews study

Dans l'onglet Reviews study j'ai simplement essayé de reproduire un exercice que j'avais fais lors de mes études. 
![Reviews_study](https://github.com/MiroMilano59/Exercice_streamlit/assets/153615242/49a717db-c728-4172-b04e-6aab2ab7d29b)
Cependant je voulais pouvoir voir plus en détail les valeurs étudiées, j'ai donc adapté mes graphiques à Streamlit. A présent les graphique réagissent au passage de la souris, qui nous indique par exemple la quantitée d'une barre.
![Reviews_study2](https://github.com/MiroMilano59/Exercice_streamlit/assets/153615242/d7c3a8c0-c176-4ecf-b6d1-1deb467f80b2)

## Trainning and prediction

Pour finir dans l'onglet Trainning and prediction, 

![Classificateur](https://github.com/MiroMilano59/Exercice_streamlit/assets/153615242/1ed234d0-ebd6-4e4e-b697-796bc752b939)

![Matrice_de_confusion](https://github.com/MiroMilano59/Exercice_streamlit/assets/153615242/95d4cff8-0599-482c-ab2c-b8cc38d515fa)
![Courbe_ROC](https://github.com/MiroMilano59/Exercice_streamlit/assets/153615242/6efa34bd-2ff5-448c-91ff-78d6c63391b9)
![Precision_recall_curve](https://github.com/MiroMilano59/Exercice_streamlit/assets/153615242/63ca5255-cd70-4e8f-baec-b8e8cbb4b121)
