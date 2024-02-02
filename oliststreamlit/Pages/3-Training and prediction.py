import plotly.express as px
import streamlit as st
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve

connection = sqlite3.connect("olist.db")

df = pd.read_sql_query("SELECT * FROM TrainingDataset", connection)

connection.close()

st.set_page_config(page_title='Olist Dataset')
st.header('Classificateur - olist dataset')
st.sidebar.header('Classificateur')

options = st.sidebar.selectbox('Choisir un classificateur',
                               options=['RandomForest', 'OneHotEncoder'])

st.sidebar.header('Hyperparamètre du modèle')

nb_arbre = st.sidebar.slider('Nombre d\'arbre dans la foret', min_value=50, max_value=250, value=50, step=25)

profondeur_arbre = st.sidebar.slider('Profondeur maximale d\'un arbre', min_value=1, max_value=10, value=1)

st.sidebar.markdown('Choisir une métrique d\'évaluation')
colonne_selected = st.sidebar.multiselect('Choisir une ou des colonnes',
                                          options=['produit_recu',
                                                   'temps_livraison',
                                                #    'order_status',
                                                   'retard_livraison'])

# options_fit = st.sidebar.radio('Souhaitez-vous fit le modèle ?',
#                                options=['Oui', 'Non'])

# if options_fit == 'Oui':
#     show_model_details = st.sidebar.checkbox("Afficher les détails du modèle", value=True)
show_model_details = st.sidebar.checkbox("Afficher les détails du modèle", value=True)

if st.sidebar.button("Exécuter le modèle"):
    if not colonne_selected:
        st.markdown('Erreur, vous devez choisir une ou des colonne(s).')
    else:
        if options == 'RandomForest':
            y = df['score']
            X = df[colonne_selected]
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
            model = RandomForestClassifier(max_depth=profondeur_arbre, n_estimators=nb_arbre)

            progress_bar = st.progress(0)

            # if options_fit == 'Oui':
            #     with st.spinner("Entraînement du modèle en cours..."):
            #         model.fit(X_train, y_train)
            #     st.success("Modèle entraîné avec succès!")
            # elif options_fit == 'Non':
            #     st.text('Vous avez choisi de ne pas fit le modèle')
            with st.spinner("Entraînement du modèle en cours..."):
                model.fit(X_train, y_train)
            st.success("Modèle entraîné avec succès!")

            recall_train = round(recall_score(y_train, model.predict(X_train)), 4)
            acc_train = round(accuracy_score(y_train, model.predict(X_train)), 4)
            f1_train = round(f1_score(y_train, model.predict(X_train)), 4)
            st.text(f"Pour le jeu d'entrainement: \n le recall est de {recall_train}, \n l'accuracy de {acc_train} \n le f1 score de {f1_train}")

            recall_test = round(recall_score(y_test, model.predict(X_test)), 4)
            acc_test = round(accuracy_score(y_test, model.predict(X_test)), 4)
            f1_test = round(f1_score(y_test, model.predict(X_test)), 4)
            st.text(f"Pour le jeu de test: \n le recall est de {recall_test}, \n l'accuracy de {acc_test} \n le f1 score de {f1_test}")

            if show_model_details:
                st.text(f"Détails du modèle : {model.get_params()}")

            st.text('Matrice de confusion')
            fig_confusion_matrix = px.imshow(confusion_matrix(y_test, model.predict(X_test)),
                                     labels=dict(x="Predicted", y="True"),
                                     x=['0', '1'], y=['0', '1'],
                                     color_continuous_scale='Viridis')
            st.plotly_chart(fig_confusion_matrix)

            y_probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            auc_score = auc(fpr, tpr)

            fig_roc_curve = go.Figure()
            fig_roc_curve.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='RandomForest'))
            fig_roc_curve.update_layout(title='Courbe ROC (AUC={:.2f})'.format(auc_score),
                                        xaxis=dict(title='Taux de faux positifs'),
                                        yaxis=dict(title='Taux de vrais positifs'),
                                        )
            st.plotly_chart(fig_roc_curve)

            precision, recall, _ = precision_recall_curve(y_test, y_probs)

            fig_precision_recall = go.Figure()
            fig_precision_recall.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='RandomForest'))
            fig_precision_recall.update_layout(title='Precision-Recall Curve',
                                            xaxis=dict(title='Recall'),
                                            yaxis=dict(title='Precision'),
                                            )
            st.plotly_chart(fig_precision_recall)

            progress_bar.progress(1.0)

        elif options == 'OneHotEncoder':
            y = df['score']
            X = df[colonne_selected]
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

            categorical_cols = X.select_dtypes(include=['object']).columns
            numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

            preprocessor = ColumnTransformer(
                transformers=[('num', 'passthrough', numeric_cols),
                            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])

            model = RandomForestClassifier(max_depth=profondeur_arbre, n_estimators=nb_arbre)

            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('imputer', SimpleImputer(strategy='mean')),
                ('robust', RobustScaler()),
                ('poly', PolynomialFeatures(2)),
                ('select_feature', SelectKBest(f_classif, k='all')),
                ('random_forest', model),])

            progress_bar = st.progress(0)

            with st.spinner("Fitting the ColumnTransformer..."):
                X_train_transformed = preprocessor.fit_transform(X_train)

            with st.spinner("Entraînement du modèle en cours..."):
                pipe.fit(X_train, y_train)

            st.success("Modèle entraîné avec succès!")

            recall_train = round(recall_score(y_train, pipe.predict(X_train)), 4)
            acc_train = round(accuracy_score(y_train, pipe.predict(X_train)), 4)
            f1_train = round(f1_score(y_train, pipe.predict(X_train)), 4)
            st.text(f"Pour le jeu d'entraînement: \n le recall est de {recall_train}, \n l'accuracy de {acc_train} \n le f1 score de {f1_train}")

            with st.spinner("Prédiction sur le jeu de test..."):
                X_test_transformed = preprocessor.transform(X_test)
                y_probs = pipe.predict_proba(X_test)[:, 1]

            recall_test = round(recall_score(y_test, pipe.predict(X_test)), 4)
            acc_test = round(accuracy_score(y_test, pipe.predict(X_test)), 4)
            f1_test = round(f1_score(y_test, pipe.predict(X_test)), 4)
            st.text(f"Pour le jeu de test: \n le recall est de {recall_test}, \n l'accuracy de {acc_test} \n le f1 score de {f1_test}")

            if show_model_details:
                st.text(f"Détails du modèle : {model.get_params()}")
            
            # fig_confusion_matrix = px.imshow(confusion_matrix(y_test, model.predict(X_test)),
            #                          labels=dict(x="Predicted", y="True"),
            #                          x=['0', '1'], y=['0', '1'],
            #                          color_continuous_scale='Viridis')
            # st.plotly_chart(fig_confusion_matrix)

            # y_probs = model.predict_proba(X_test)[:, 1]
            # fpr, tpr, _ = roc_curve(y_test, y_probs)
            # auc_score = auc(fpr, tpr)

            # fig_roc_curve = go.Figure()
            # fig_roc_curve.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='RandomForest'))
            # fig_roc_curve.update_layout(title='Courbe ROC (AUC={:.2f})'.format(auc_score),
            #                             xaxis=dict(title='Taux de faux positifs'),
            #                             yaxis=dict(title='Taux de vrais positifs'),
            #                             )
            # st.plotly_chart(fig_roc_curve)

            # X_test_transformed = preprocessor.transform(X_test)
            # y_probs = model.predict_proba(X_test_transformed)[:, 1]

            # # y_probs = model.predict_proba(X_test)[:, 1]
            # precision, recall, _ = precision_recall_curve(y_test, y_probs)

            # fig_precision_recall = go.Figure()
            # fig_precision_recall.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='RandomForest'))
            # fig_precision_recall.update_layout(title='Precision-Recall Curve',
            #                                 xaxis=dict(title='Recall'),
            #                                 yaxis=dict(title='Precision'),
            #                                 )
            # st.plotly_chart(fig_precision_recall)

            progress_bar.progress(1.0)

