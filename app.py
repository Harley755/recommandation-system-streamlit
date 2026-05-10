"""
Interface Streamlit - Système de Recommandation Item-Item
Application interactive pour démontrer le fonctionnement du système de recommandation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from recommender_system import ItemItemRecommender, create_sample_dataset
from data_loader import DataLoader
import time
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Système de Recommandation Item-Item",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un meilleur design
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #00b4d8;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_dataset(dataset_type: str, n_users: int = 100, n_items: int = 50, n_ratings: int = 2000):
    """
    Charge ou crée le dataset avec mise en cache
    """
    loader = DataLoader()
    
    if dataset_type == "Synthétique":
        ratings_df, movies_df = loader.load_data("synthetic", n_users=n_users, n_items=n_items, n_ratings=n_ratings)
        return ratings_df, movies_df, "Dataset synthétique créé"
    elif dataset_type == "MovieLens 100K":
        ratings_df, movies_df = loader.load_data("movielens_100k")
        return ratings_df, movies_df, "Dataset MovieLens 100K chargé"
    elif dataset_type == "MovieLens Latest":
        ratings_df, movies_df = loader.load_data("movielens_latest")
        return ratings_df, movies_df, "Dataset MovieLens Latest Small chargé"
    else:
        # Fallback vers synthétique
        ratings_df, movies_df = loader.load_data("synthetic", n_users=200, n_items=100, n_ratings=5000)
        return ratings_df, movies_df, "Dataset de démonstration (similaire à MovieLens)"

@st.cache_resource
def train_recommender(df: pd.DataFrame, min_ratings_per_item: int, min_ratings_per_user: int):
    """
    Entraîne le modèle de recommandation avec mise en cache
    """
    recommender = ItemItemRecommender(
        min_ratings_per_item=min_ratings_per_item,
        min_ratings_per_user=min_ratings_per_user
    )
    recommender.fit(df)
    return recommender

def display_user_ratings(recommender, user_id: int):
    """
    Affiche les notes de l'utilisateur sélectionné
    """
    if user_id not in recommender.user_mapping:
        st.warning("Cet utilisateur n'a pas assez de notes pour être inclus dans le modèle.")
        return pd.DataFrame()
    
    user_idx = recommender.user_mapping[user_id]
    rated_items, ratings = recommender._get_user_rated_items(user_idx)
    
    if len(rated_items) == 0:
        st.info("Cet utilisateur n'a aucune note enregistrée.")
        return pd.DataFrame()
    
    # Créer un DataFrame avec les notes
    item_ids = [recommender.reverse_item_mapping[idx] for idx in rated_items]
    user_ratings_df = pd.DataFrame({
        'Item ID': item_ids,
        'Note': ratings,
        'Note Normalisée': ratings - recommender.user_means[user_idx]
    })
    
    user_ratings_df = user_ratings_df.sort_values('Note', ascending=False)
    
    return user_ratings_df

def display_user_ratings_with_metadata(recommender, user_id: int, movies_df: pd.DataFrame):
    """
    Affiche les notes de l'utilisateur avec métadonnées des films
    """
    if user_id not in recommender.user_mapping:
        st.warning("Cet utilisateur n'a pas assez de notes pour être inclus dans le modèle.")
        return pd.DataFrame()
    
    user_idx = recommender.user_mapping[user_id]
    rated_items, ratings = recommender._get_user_rated_items(user_idx)
    
    if len(rated_items) == 0:
        st.info("Cet utilisateur n'a aucune note enregistrée.")
        return pd.DataFrame()
    
    # Créer un DataFrame avec les notes et métadonnées
    item_ids = [recommender.reverse_item_mapping[idx] for idx in rated_items]
    user_ratings_data = []
    
    for item_id, rating in zip(item_ids, ratings):
        movie_info = movies_df[movies_df['item_id'] == item_id]
        
        if not movie_info.empty:
            title = movie_info['title'].iloc[0]
            genres = movie_info.get('genres', pd.Series(['N/A'])).iloc[0]
            year = movie_info.get('year', pd.Series(['N/A'])).iloc[0]
        else:
            title = f"Item {item_id}"
            genres = "N/A"
            year = "N/A"
        
        user_ratings_data.append({
            'Titre': title,
            'Genre': genres,
            'Année': year,
            'Note': rating,
            'Note Normalisée': rating - recommender.user_means[user_idx]
        })
    
    user_ratings_df = pd.DataFrame(user_ratings_data)
    user_ratings_df = user_ratings_df.sort_values('Note', ascending=False)
    
    return user_ratings_df

def display_recommendations_with_metadata(recommendations: list, movies_df: pd.DataFrame):
    """
    Affiche les recommandations avec métadonnées des films
    """
    if not recommendations:
        st.warning("Aucune recommandation générée.")
        return
    
    st.markdown("### Recommandations Top-N")
    
    for i, (item_id, score) in enumerate(recommendations, 1):
        # Récupérer les métadonnées du film
        movie_info = movies_df[movies_df['item_id'] == item_id]
        
        if not movie_info.empty:
            title = movie_info['title'].iloc[0]
            genres = movie_info.get('genres', pd.Series(['N/A'])).iloc[0]
            year = movie_info.get('year', pd.Series(['N/A'])).iloc[0]
            
            # Affichage enrichi
            st.markdown(f"""
            <div class="recommendation-card">
                <strong>#{i} - {title}</strong><br>
                <small>Genre: {genres} | Année: {year}</small><br>
                Score de recommandation: <code>{score:.3f}</code>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Fallback si pas de métadonnées
            st.markdown(f"""
            <div class="recommendation-card">
                <strong>#{i} - Item {item_id}</strong><br>
                Score de recommandation: <code>{score:.3f}</code>
            </div>
            """, unsafe_allow_html=True)

def create_similarity_visualization(recommender, item_ids: list):
    """
    Crée une visualisation interactive de la matrice de similarité
    """
    if len(item_ids) < 2:
        st.warning("Il faut au moins 2 items pour visualiser la similarité.")
        return
    
    # Vérifier que les items existent
    valid_items = [item_id for item_id in item_ids if item_id in recommender.item_mapping]
    
    if len(valid_items) < 2:
        st.warning("Pas assez d'items valides pour la visualisation.")
        return
    
    # Extraire la sous-matrice de similarité
    item_indices = [recommender.item_mapping[item_id] for item_id in valid_items]
    similarity_submatrix = recommender.item_similarity_matrix[np.ix_(item_indices, item_indices)]
    
    # Créer le heatmap avec Plotly
    fig = px.imshow(
        similarity_submatrix,
        x=valid_items,
        y=valid_items,
        color_continuous_scale='RdBu',
        aspect='auto',
        title='Matrice de Similarité Item-Item'
    )
    
    fig.update_layout(
        xaxis_title='Items',
        yaxis_title='Items',
        coloraxis_colorbar_title='Similarité'
    )
    
    st.plotly_chart(fig, width='stretch')

def main():
    """
    Fonction principale de l'application Streamlit
    """
    # Header principal
    st.markdown('<h1 class="main-header">Système de Recommandation Item-Item</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Cette application démontre un système de recommandation basé sur le **filtrage collaboratif item-item**.
    
    **Principe de fonctionnement :**
    - Pour chaque utilisateur, on identifie les items déjà notés
    - On calcule la similarité entre ces items et tous les autres items
    - On recommande les items les plus similaires avec un score pondéré
    """)
    
    # Sidebar pour les paramètres
    st.sidebar.markdown("## Paramètres")
    
    # Choix du dataset
    dataset_type = st.sidebar.selectbox(
        "Type de dataset",
        ["Synthétique", "MovieLens 100K", "MovieLens Latest"],
        help="Choisissez le type de données à utiliser"
    )
    
    # Paramètres du dataset synthétique
    if dataset_type == "Synthétique":
        n_users = st.sidebar.slider("Nombre d'utilisateurs", 50, 500, 100)
        n_items = st.sidebar.slider("Nombre d'items", 20, 200, 50)
        n_ratings = st.sidebar.slider("Nombre d'évaluations", 500, 5000, 2000)
    else:
        n_users = 200
        n_items = 100
        n_ratings = 5000
    
    # Paramètres du modèle
    st.sidebar.markdown("### Paramètres du modèle")
    min_ratings_item = st.sidebar.slider(
        "Notes minimum par item", 
        1, 20, 5,
        help="Nombre minimum de notes qu'un item doit avoir pour être inclus"
    )
    min_ratings_user = st.sidebar.slider(
        "Notes minimum par utilisateur", 
        1, 20, 5,
        help="Nombre minimum de notes qu'un utilisateur doit avoir pour être inclus"
    )
    
    # Bouton pour charger et entraîner
    if st.sidebar.button("Charger et entraîner", type="primary"):
        with st.spinner("Chargement des données..."):
            ratings_df, movies_df, message = load_dataset(dataset_type, n_users, n_items, n_ratings)
            time.sleep(0.5)  # Pour l'effet visuel
        
        st.success(message)
        
        with st.spinner("Entraînement du modèle..."):
            recommender = train_recommender(ratings_df, min_ratings_item, min_ratings_user)
            time.sleep(1)  # Pour l'effet visuel
        
        st.success(" Modèle entraîné avec succès!")
        
        # Stocker dans session_state
        st.session_state.ratings_df = ratings_df
        st.session_state.movies_df = movies_df
        st.session_state.recommender = recommender
        st.session_state.model_trained = True
    
    # Vérifier si le modèle est entraîné
    if 'model_trained' not in st.session_state or not st.session_state.model_trained:
        st.info("Utilisez la barre latérale pour charger les données et entraîner le modèle.")
        return
    
    # Récupérer le modèle et les données
    recommender = st.session_state.recommender
    ratings_df = st.session_state.ratings_df
    movies_df = st.session_state.movies_df
    
    # Section des statistiques
    st.markdown('<h2 class="section-header">Statistiques du modèle</h2>', 
                unsafe_allow_html=True)
    
    stats = recommender.get_model_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <strong>Utilisateurs</strong><br>
            <span style="font-size: 1.5rem; color: #1f77b4;">{stats['n_users']}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <strong>Items</strong><br>
            <span style="font-size: 1.5rem; color: #1f77b4;">{stats['n_items']}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <strong>Évaluations</strong><br>
            <span style="font-size: 1.5rem; color: #1f77b4;">{stats['n_ratings']}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        sparsity_pct = stats['sparsity'] * 100
        st.markdown(f"""
        <div class="metric-card">
            <strong>Sparsité</strong><br>
            <span style="font-size: 1.5rem; color: #e74c3c;">{sparsity_pct:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Section de recommandation
    st.markdown('<h2 class="section-header"> Génération de recommandations</h2>', 
                unsafe_allow_html=True)
    
    # Sélection de l'utilisateur
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Liste des utilisateurs valides
        valid_users = [recommender.reverse_user_mapping[idx] for idx in range(stats['n_users'])]
        selected_user = st.selectbox(
            "Sélectionnez un utilisateur",
            valid_users,
            help="Choisissez un utilisateur pour lequel générer des recommandations"
        )
    
    with col2:
        n_recommendations = st.slider(
            "Nombre de recommandations",
            1, 20, 10,
            help="Combien d'items recommander?"
        )
    
    # Options avancées
    with st.expander("Options avancées"):
        use_mean_centering = st.checkbox(
            "Utiliser la normalisation par moyenne (Mean-Centering)",
            value=True,
            help="Normalise les notes en soustrayant la moyenne de l'utilisateur"
        )
        
        show_user_ratings = st.checkbox(
            "Afficher les notes de l'utilisateur",
            value=True,
            help="Montre les items déjà notés par l'utilisateur"
        )
    
    # Bouton pour générer les recommandations
    if st.button("Générer les recommandations", type="primary"):
        with st.spinner("Génération des recommandations..."):
            recommendations = recommender.recommend_items(
                selected_user, 
                n_recommendations=n_recommendations,
                use_mean_centering=use_mean_centering
            )
            time.sleep(0.5)
        
        # Afficher les recommandations avec métadonnées
        display_recommendations_with_metadata(recommendations, movies_df)
        
        # Afficher les notes de l'utilisateur si demandé
        if show_user_ratings:
            st.markdown("### Notes de l'utilisateur")
            user_ratings_df = display_user_ratings_with_metadata(recommender, selected_user, movies_df)
            
            if not user_ratings_df.empty:
                st.dataframe(user_ratings_df, width='stretch')
                
                # Statistiques sur l'utilisateur
                user_idx = recommender.user_mapping[selected_user]
                user_mean = recommender.user_means[user_idx]
                st.metric("Moyenne des notes de l'utilisateur", f"{user_mean:.2f}")
    
    # Section de visualisation
    st.markdown('<h2 class="section-header">Visualisation des similarités</h2>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Visualisation des items similaires
        st.markdown("#### Items similaires")
        selected_item_for_sim = st.number_input(
            "ID de l'item de référence",
            min_value=1,
            max_value=stats['n_items'],
            value=1,
            help="Entrez l'ID d'un item pour voir ses items similaires"
        )
        
        n_similar = st.slider("Nombre d'items similaires", 1, 10, 5)
        
        if st.button("Afficher les items similaires"):
            try:
                similar_items = recommender.get_similar_items(selected_item_for_sim, n_similar)
                
                if similar_items:
                    similar_df = pd.DataFrame(similar_items, columns=['Item ID', 'Similarité'])
                    st.dataframe(similar_df, width='stretch')
                else:
                    st.warning("Aucun item similaire trouvé.")
            except ValueError as e:
                st.error(str(e))
    
    with col2:
        # Visualisation de la matrice de similarité
        st.markdown("#### Matrice de similarité")
        st.info("Sélectionnez jusqu'à 10 items pour visualiser leur matrice de similarité.")
        
        selected_items_for_heatmap = st.multiselect(
            "Items à visualiser",
            options=list(range(1, min(stats['n_items'] + 1, 51))),  # Limiter à 50 pour la performance
            default=[1, 2, 3, 4, 5],
            max_selections=10
        )
        
        if selected_items_for_heatmap:
            create_similarity_visualization(recommender, selected_items_for_heatmap)
    
    # Section d'explication
    with st.expander("Comment fonctionne l'algorithme?"):
        st.markdown("""
        ### Formule de calcul des recommandations
        
        Pour un utilisateur *u* et un item candidat *i*, le score de recommandation est calculé comme :
        
        ```
        score(u,i) = Σ [sim(i,j) × (rating(u,j) - avg_rating(u))] / Σ |sim(i,j)|
        ```
        
        Où :
        - *sim(i,j)* est la similarité cosine entre les items *i* et *j*
        - *rating(u,j)* est la note de l'utilisateur *u* pour l'item *j*
        - *avg_rating(u)* est la moyenne des notes de l'utilisateur *u*
        
        ### Similarité Cosine
        
        La similarité cosine entre deux items *i* et *j* est calculée comme :
        
        ```
        sim(i,j) = (v_i · v_j) / (||v_i|| × ||v_j||)
        ```
        
        Où *v_i* et *v_j* sont les vecteurs des notes pour les items *i* et *j*.
        
        ### Processus de recommandation
        
        1. **Identification** : On trouve les items déjà notés par l'utilisateur
        2. **Calcul** : On calcule la similarité entre ces items et tous les autres items
        3. **Scoring** : On pondère les similarités par les notes normalisées
        4. **Classement** : On trie les items par score décroissant
        5. **Sélection** : On retourne le Top-N des meilleurs scores
        """)

if __name__ == "__main__":
    main()
