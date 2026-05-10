"""
Système de Recommandation - Filtrage Collaboratif Item-Item
Implémentation complète d'un système de recommandation Top-N basé sur la similarité cosine
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class ItemItemRecommender:
    """
    Système de recommandation basé sur le filtrage collaboratif item-item
    
    Principe : Pour recommander des items à un utilisateur, on calcule la similarité
    entre les items déjà notés par l'utilisateur et les autres items, puis on
    propose les items les plus similaires.
    """
    
    def __init__(self, min_ratings_per_item: int = 5, min_ratings_per_user: int = 5):
        """
        Initialisation du système de recommandation
        
        Args:
            min_ratings_per_item: Nombre minimum de notes par item pour être considéré
            min_ratings_per_user: Nombre minimum de notes par utilisateur pour être considéré
        """
        self.min_ratings_per_item = min_ratings_per_item
        self.min_ratings_per_user = min_ratings_per_user
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.user_means = None
        self.item_means = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        
    def fit(self, ratings_df: pd.DataFrame) -> None:
        """
        Entraînement du modèle : construction de la matrice utilisateur-item et 
        calcul de la matrice de similarité item-item
        
        Args:
            ratings_df: DataFrame avec colonnes [user_id, item_id, rating]
        """
        print("Préparation des données...")
        
        # Filtrage des items et utilisateurs avec suffisamment de notes
        item_counts = ratings_df['item_id'].value_counts()
        user_counts = ratings_df['user_id'].value_counts()
        
        valid_items = item_counts[item_counts >= self.min_ratings_per_item].index
        valid_users = user_counts[user_counts >= self.min_ratings_per_user].index
        
        filtered_df = ratings_df[
            (ratings_df['item_id'].isin(valid_items)) & 
            (ratings_df['user_id'].isin(valid_users))
        ].copy()
        
        # Création des mappings pour les indices continus
        unique_users = sorted(filtered_df['user_id'].unique())
        unique_items = sorted(filtered_df['item_id'].unique())
        
        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
        
        # Construction de la matrice utilisateur-item
        self.user_item_matrix = np.zeros((len(unique_users), len(unique_items)))
        
        for _, row in filtered_df.iterrows():
            user_idx = self.user_mapping[row['user_id']]
            item_idx = self.item_mapping[row['item_id']]
            self.user_item_matrix[user_idx, item_idx] = row['rating']
        
        # Calcul des moyennes pour normalisation
        self.user_means = np.array([
            np.mean(self.user_item_matrix[user_idx, self.user_item_matrix[user_idx, :] > 0])
            if np.any(self.user_item_matrix[user_idx, :] > 0) else 0
            for user_idx in range(len(unique_users))
        ])
        
        self.item_means = np.array([
            np.mean(self.user_item_matrix[self.user_item_matrix[:, item_idx] > 0, item_idx])
            if np.any(self.user_item_matrix[:, item_idx] > 0) else 0
            for item_idx in range(len(unique_items))
        ])
        
        print("Calcul de la matrice de similarité item-item...")
        
        # Calcul de la similarité cosine entre items
        # On utilise la matrice transposée pour comparer les items (colonnes)
        self.item_similarity_matrix = cosine_similarity(self.user_item_matrix.T)
        
        # Mettre la diagonale à 0 pour éviter de recommander les mêmes items
        np.fill_diagonal(self.item_similarity_matrix, 0)
        
        print("Entraînement terminé!")
    
    def _get_user_rated_items(self, user_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Récupère les items déjà notés par un utilisateur
        
        Args:
            user_idx: Index de l'utilisateur
            
        Returns:
            Tuple: (indices des items notés, notes correspondantes)
        """
        rated_mask = self.user_item_matrix[user_idx, :] > 0
        rated_items = np.where(rated_mask)[0]
        ratings = self.user_item_matrix[user_idx, rated_items]
        return rated_items, ratings
    
    def recommend_items(self, user_id: int, n_recommendations: int = 10, 
                       use_mean_centering: bool = True) -> List[Tuple[int, float]]:
        """
        Génère des recommandations Top-N pour un utilisateur
        
        Args:
            user_id: ID de l'utilisateur
            n_recommendations: Nombre de recommandations à générer
            use_mean_centering: Utiliser la normalisation par moyenne
            
        Returns:
            Liste de tuples (item_id, score) triée par score décroissant
        """
        if user_id not in self.user_mapping:
            raise ValueError(f"Utilisateur {user_id} non trouvé dans les données d'entraînement")
        
        user_idx = self.user_mapping[user_id]
        
        # Récupérer les items déjà notés par l'utilisateur
        rated_items, ratings = self._get_user_rated_items(user_idx)
        
        if len(rated_items) == 0:
            print("L'utilisateur n'a aucune note. Impossible de générer des recommandations.")
            return []
        
        # Calcul des scores pour tous les items non notés
        candidate_items = np.setdiff1d(np.arange(self.user_item_matrix.shape[1]), rated_items)
        scores = np.zeros(len(candidate_items))
        
        for i, candidate_item in enumerate(candidate_items):
            score = 0.0
            similarity_sum = 0.0
            
            # Pour chaque item déjà noté par l'utilisateur
            for rated_item, rating in zip(rated_items, ratings):
                similarity = self.item_similarity_matrix[candidate_item, rated_item]
                
                if similarity > 0:
                    if use_mean_centering:
                        # Normalisation par moyenne (mean-centering)
                        normalized_rating = rating - self.user_means[user_idx]
                        score += similarity * normalized_rating
                    else:
                        score += similarity * rating
                    
                    similarity_sum += similarity
            
            # Normalisation du score
            if similarity_sum > 0:
                scores[i] = score / similarity_sum
                if use_mean_centering:
                    scores[i] += self.user_means[user_idx]  # Ajouter la moyenne de l'utilisateur
        
        # Tri des items par score décroissant
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        
        recommendations = []
        for idx in top_indices:
            item_id = self.reverse_item_mapping[candidate_items[idx]]
            recommendations.append((item_id, scores[idx]))
        
        return recommendations
    
    def get_similar_items(self, item_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """
        Retourne les items les plus similaires à un item donné
        
        Args:
            item_id: ID de l'item de référence
            n_similar: Nombre d'items similaires à retourner
            
        Returns:
            Liste de tuples (item_id, similarité) triée par similarité décroissante
        """
        if item_id not in self.item_mapping:
            raise ValueError(f"Item {item_id} non trouvé dans les données d'entraînement")
        
        item_idx = self.item_mapping[item_id]
        similarities = self.item_similarity_matrix[item_idx, :]
        
        # Tri par similarité décroissante
        top_indices = np.argsort(similarities)[::-1][:n_similar]
        
        similar_items = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Ne garder que les similarités positives
                similar_item_id = self.reverse_item_mapping[idx]
                similar_items.append((similar_item_id, similarities[idx]))
        
        return similar_items
    
    def visualize_similarity_heatmap(self, item_ids: List[int], figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Visualise la matrice de similarité pour un sous-ensemble d'items
        
        Args:
            item_ids: Liste des IDs d'items à visualiser
            figsize: Taille de la figure
        """
        # Vérifier que tous les items existent
        valid_items = [item_id for item_id in item_ids if item_id in self.item_mapping]
        
        if len(valid_items) < 2:
            print("Il faut au moins 2 items valides pour visualiser la similarité")
            return
        
        # Extraire la sous-matrice de similarité
        item_indices = [self.item_mapping[item_id] for item_id in valid_items]
        similarity_submatrix = self.item_similarity_matrix[np.ix_(item_indices, item_indices)]
        
        # Création du heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(similarity_submatrix, 
                   xticklabels=valid_items, 
                   yticklabels=valid_items,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt='.3f')
        plt.title('Matrice de Similarité Item-Item')
        plt.xlabel('Items')
        plt.ylabel('Items')
        plt.tight_layout()
        plt.show()
    
    def get_model_stats(self) -> Dict:
        """
        Retourne des statistiques sur le modèle entraîné
        
        Returns:
            Dictionnaire avec les statistiques du modèle
        """
        if self.user_item_matrix is None:
            return {"error": "Modèle non entraîné"}
        
        sparsity = 1 - (np.count_nonzero(self.user_item_matrix) / self.user_item_matrix.size)
        
        return {
            "n_users": self.user_item_matrix.shape[0],
            "n_items": self.user_item_matrix.shape[1],
            "n_ratings": np.count_nonzero(self.user_item_matrix),
            "sparsity": sparsity,
            "avg_ratings_per_user": np.count_nonzero(self.user_item_matrix) / self.user_item_matrix.shape[0],
            "avg_ratings_per_item": np.count_nonzero(self.user_item_matrix) / self.user_item_matrix.shape[1],
            "global_avg_rating": np.mean(self.user_item_matrix[self.user_item_matrix > 0])
        }


def create_sample_dataset(n_users: int = 100, n_items: int = 50, 
                         n_ratings: int = 2000, rating_range: Tuple[int, int] = (1, 5),
                         random_state: int = 42) -> pd.DataFrame:
    """
    Crée un dataset synthétique pour tester le système de recommandation
    
    Args:
        n_users: Nombre d'utilisateurs
        n_items: Nombre d'items
        n_ratings: Nombre total d'évaluations
        rating_range: Plage des notes (min, max)
        random_state: Graine aléatoire
        
    Returns:
        DataFrame avec les évaluations synthétiques
    """
    np.random.seed(random_state)
    
    # Génération des évaluations aléatoires
    user_ids = np.random.randint(1, n_users + 1, n_ratings)
    item_ids = np.random.randint(1, n_items + 1, n_ratings)
    ratings = np.random.randint(rating_range[0], rating_range[1] + 1, n_ratings)
    
    # Création du DataFrame
    df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings
    })
    
    # Suppression des doublons
    df = df.drop_duplicates(subset=['user_id', 'item_id'])
    
    # Ajout d'un peu de structure : certains utilisateurs préfèrent certains types d'items
    for user_id in range(1, min(11, n_users + 1)):
        # Chaque utilisateur a une préférence pour un certain groupe d'items
        preferred_items = range((user_id - 1) * 5 + 1, min(user_id * 5 + 1, n_items + 1))
        for item_id in preferred_items:
            if np.random.random() < 0.3:  # 30% de chance d'avoir noté cet item
                if not ((df['user_id'] == user_id) & (df['item_id'] == item_id)).any():
                    new_rating = np.random.randint(4, 6)  # Notes plus élevées pour les items préférés
                    df = pd.concat([df, pd.DataFrame([{
                        'user_id': user_id, 
                        'item_id': item_id, 
                        'rating': min(new_rating, 5)
                    }])], ignore_index=True)
    
    return df.reset_index(drop=True)


if __name__ == "__main__":
    # Test du système avec un dataset synthétique
    print("Création du dataset de test...")
    df = create_sample_dataset(n_users=50, n_items=30, n_ratings=800)
    
    print("\nAperçu des données:")
    print(df.head())
    print(f"\nStatistiques: {len(df)} évaluations, {df['user_id'].nunique()} utilisateurs, {df['item_id'].nunique()} items")
    
    # Entraînement du modèle
    recommender = ItemItemRecommender(min_ratings_per_item=3, min_ratings_per_user=3)
    recommender.fit(df)
    
    # Statistiques du modèle
    print("\nStatistiques du modèle:")
    stats = recommender.get_model_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test de recommandation
    test_user = df['user_id'].iloc[0]
    print(f"\nRecommandations pour l'utilisateur {test_user}:")
    recommendations = recommender.recommend_items(test_user, n_recommendations=5)
    for item_id, score in recommendations:
        print(f"  Item {item_id}: score = {score:.3f}")
    
    # Test d'items similaires
    test_item = df['item_id'].iloc[0]
    print(f"\nItems similaires à l'item {test_item}:")
    similar_items = recommender.get_similar_items(test_item, n_similar=5)
    for item_id, similarity in similar_items:
        print(f"  Item {item_id}: similarité = {similarity:.3f}")
