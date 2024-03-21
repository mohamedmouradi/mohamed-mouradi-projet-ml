# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import pandas as pd
import joblib
import mlflow
import os

# Initialiser le suivi de MLflow
mlflow.start_run(run_id='modest_stem_3016ct2x49')

# Charger le modèle .pkl
model = joblib.load('model.pkl')

# Définir une fonction pour effectuer des prédictions
def predict(data):
    prediction = model.predict(data)
    return prediction

# Définir le titre et l'icône de la page
st.set_page_config(
    page_title="Prédiction du diabète",
    page_icon="💉",
)

# Titre de l'application
st.title('Prédiction du diabète')

# Description de l'application
st.write("Prédire le début du diabète en fonction des mesures de diagnostic")

# Saisie des données utilisateur
st.sidebar.header('Saisir les caractéristiques')
pregnancies = st.sidebar.number_input('Grossesses', min_value=0, max_value=20)
glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=300)
# Ajouter d'autres caractéristiques ici...

# Bouton pour effectuer la prédiction
if st.sidebar.button('Prédire'):
    # Rassembler les données saisies par l'utilisateur dans un DataFrame
    input_data = pd.DataFrame({'Grossesses': [pregnancies],
                               'Glucose': [glucose]})
    # Effectuer la prédiction
    prediction = predict(input_data)
    # Afficher la prédiction
    st.write('La prédiction est :', prediction)

# Liens et ressources supplémentaires
st.markdown(
    """
    ### Pour en savoir plus :
    - Consultez la [documentation de Streamlit](https://docs.streamlit.io)
    - Posez vos questions sur le [forum de la communauté Streamlit](https://discuss.streamlit.io)
    """
)

# Terminer l'exécution MLflow actuelle s'il y en a une
if mlflow.active_run():
    mlflow.end_run()


