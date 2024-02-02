import plotly.express as px
import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from geopy.geocoders import Nominatim

parent_directory = os.path.abspath(os.path.join(os.getcwd()))

olist_customers_dataset = pd.read_csv(os.path.join(parent_directory,'data\customers_dataset.csv'), sep = ',')
olist_geolocation_dataset = pd.read_csv(os.path.join(parent_directory,'data\geolocation_dataset.csv'), sep = ',')
olist_order_items_dataset = pd.read_csv(os.path.join(parent_directory,'data\order_items_dataset.csv'), sep = ',')
olist_order_payments_dataset = pd.read_csv(os.path.join(parent_directory,'data\order_payments_dataset.csv'), sep = ',')
# olist_order_reviews_dataset = pd.read_csv(os.path.join(parent_directory,'data\order_reviews_dataset.csv'), sep = ',')
olist_orders_dataset = pd.read_csv(os.path.join(parent_directory,'data\orders_dataset.csv'), sep = ',')
olist_products_dataset = pd.read_csv(os.path.join(parent_directory,'data\products_dataset.csv'), sep = ',')
olist_sellers_dataset = pd.read_csv(os.path.join(parent_directory,'data\sellers_dataset.csv'), sep = ',')
olist_product_category_name_translation = pd.read_csv(os.path.join(parent_directory,'data\product_category_name_translation.csv'), sep = ',')

st.set_page_config(page_title='Olist Dataset')
st.header('Geolocation - olist dataset')
st.markdown('Explore the different geolocation of all ')
st.sidebar.header('Variable comparaison')

options = st.sidebar.radio('Select category',
                          options=['Customers',
                                   'Sellers',
                                   'Customers and sellers',
                                   'Personnes par region'])

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

elif options == 'Sellers':
    df_mergeddd = pd.merge(olist_sellers_dataset, olist_geolocation_dataset, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='inner')
    df_unique_sellers= df_mergeddd.drop_duplicates(subset='seller_id', keep='first')

    fig2 = px.scatter_geo(df_unique_sellers,
                        lat='geolocation_lat',
                        lon='geolocation_lng',
                        hover_name='seller_city',
                        projection='natural earth',
                        title='Seller map',
                        color_discrete_sequence=['blue'],
                        height=800,
                        width=800)

    st.plotly_chart(fig2)

elif options == 'Customers and sellers':
    df_mergedd = pd.merge(olist_customers_dataset, olist_geolocation_dataset, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='inner')
    df_unique_customers = df_mergedd.drop_duplicates(subset='customer_id', keep='first')

    fig3 = px.scatter_geo(df_unique_customers,
                        lat='geolocation_lat',
                        lon='geolocation_lng',
                        hover_name='customer_city',
                        projection='natural earth',
                        color_discrete_sequence=['red'])

    df_mergeddd = pd.merge(olist_sellers_dataset, olist_geolocation_dataset, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='inner')
    df_unique_sellers= df_mergeddd.drop_duplicates(subset='seller_id', keep='first')

    fig4 = px.scatter_geo(df_unique_sellers,
                        lat='geolocation_lat',
                        lon='geolocation_lng',
                        hover_name='seller_city',
                        projection='natural earth',
                        color_discrete_sequence=['blue'])

    fig5 = fig3.add_trace(fig4.data[0])

    st.plotly_chart(fig5)

