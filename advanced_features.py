"""
Fonctionnalités avancées pour le système de recommandation
- Comparaison de métriques de similarité
- Gestion du cold start
- Optimisation des performances
- Évaluations et métriques
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import time
from typing import List, Tuple, Dict, Optional
from recommender_system import ItemItemRecommender

class AdvancedRecommender(ItemItemRecommender):
    """
    Version étendue du système de recommandation avec fonctionnalités avancées
    """
    
    def __init__(self, min_ratings_per_item: int = 5, min_ratings_per_user: int = 5):
        super().__init__(min_ratings_per_item, min_ratings_per_user)
        self.pearson_similarity_matrix = None
        self.euclidean_similarity_matrix = None
        self.performance_metrics = {}
        
    def compute_alternative_similarities(self):
        """
        Calcule des matrices de similarité alternatives pour comparaison
        """
        if self.user_item_matrix is None:
            raise ValueError("Le modèle doit être entraîné avant de calculer les similarités")
        
        print("Calcul des similarités alternatives...")
        
        # Similarité de Pearson (corrélation)
        print("  - Similarité de Pearson...")
        # Centrer les données par colonne (item)
        centered_matrix = self.user_item_matrix - np.nanmean(self.user_item_matrix, axis=0)
        # Remplacer NaN par 0 pour le calcul
        centered_matrix = np.nan_to_num(centered_matrix)
        
        # Calcul de la corrélation de Pearson
        self.pearson_similarity_matrix = np.corrcoef(centered_matrix.T)
        np.fill_diagonal(self.pearson_similarity_matrix, 0)
        
        # Similarité Euclidienne (convertie en similarité)
        print("  - Similarité Euclidienne...")
        euclidean_dist = euclidean_distances(self.user_item_matrix.T)
        # Conversion distance -> similarité (plus la distance est petite, plus la similarité est grande)
        max_dist = np.max(euclidean_dist)
        self.euclidean_similarity_matrix = 1 - (euclidean_dist / max_dist)
        np.fill_diagonal(self.euclidean_similarity_matrix, 0)
        
        print("Similarités alternatives calculées!")
    
    def recommend_items_with_similarity(self, user_id: int, n_recommendations: int = 10, 
                                      similarity_type: str = 'cosine',
                                      use_mean_centering: bool = True) -> List[Tuple[int, float]]:
        """
        Génère des recommandations en utilisant un type de similarité spécifique
        
        Args:
            user_id: ID de l'utilisateur
            n_recommendations: Nombre de recommandations
            similarity_type: 'cosine', 'pearson', ou 'euclidean'
            use_mean_centering: Utiliser la normalisation par moyenne
            
        Returns:
            Liste de tuples (item_id, score)
        """
        # Sélectionner la matrice de similarité appropriée
        if similarity_type == 'cosine':
            similarity_matrix = self.item_similarity_matrix
        elif similarity_type == 'pearson':
            if self.pearson_similarity_matrix is None:
                self.compute_alternative_similarities()
            similarity_matrix = self.pearson_similarity_matrix
        elif similarity_type == 'euclidean':
            if self.euclidean_similarity_matrix is None:
                self.compute_alternative_similarities()
            similarity_matrix = self.euclidean_similarity_matrix
        else:
            raise ValueError(f"Type de similarité non supporté: {similarity_type}")
        
        if user_id not in self.user_mapping:
            raise ValueError(f"Utilisateur {user_id} non trouvé")
        
        user_idx = self.user_mapping[user_id]
        rated_items, ratings = self._get_user_rated_items(user_idx)
        
        if len(rated_items) == 0:
            return []
        
        candidate_items = np.setdiff1d(np.arange(self.user_item_matrix.shape[1]), rated_items)
        scores = np.zeros(len(candidate_items))
        
        for i, candidate_item in enumerate(candidate_items):
            score = 0.0
            similarity_sum = 0.0
            
            for rated_item, rating in zip(rated_items, ratings):
                similarity = similarity_matrix[candidate_item, rated_item]
                
                if similarity > 0:
                    if use_mean_centering:
                        normalized_rating = rating - self.user_means[user_idx]
                        score += similarity * normalized_rating
                    else:
                        score += similarity * rating
                    
                    similarity_sum += similarity
            
            if similarity_sum > 0:
                scores[i] = score / similarity_sum
                if use_mean_centering:
                    scores[i] += self.user_means[user_idx]
        
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        recommendations = []
        
        for idx in top_indices:
            item_id = self.reverse_item_mapping[candidate_items[idx]]
            recommendations.append((item_id, scores[idx]))
        
        return recommendations
    
    def compare_similarity_methods(self, user_id: int, n_recommendations: int = 10) -> pd.DataFrame:
        """
        Compare les recommandations générées par différentes méthodes de similarité
        
        Args:
            user_id: ID de l'utilisateur à tester
            n_recommendations: Nombre de recommandations
            
        Returns:
            DataFrame comparatif des résultats
        """
        methods = ['cosine', 'pearson', 'euclidean']
        results = {}
        
        for method in methods:
            try:
                recommendations = self.recommend_items_with_similarity(
                    user_id, n_recommendations, method
                )
                results[method] = [item_id for item_id, _ in recommendations]
            except Exception as e:
                results[method] = []
                print(f"Erreur avec {method}: {e}")
        
        # Créer un DataFrame comparatif
        comparison_df = pd.DataFrame(results)
        comparison_df.index = range(1, n_recommendations + 1)
        comparison_df.index.name = 'Rang'
        
        return comparison_df
    
    def handle_cold_start(self, n_recommendations: int = 10, strategy: str = 'popular') -> List[Tuple[int, float]]:
        """
        Gère le problème du cold start pour les nouveaux utilisateurs
        
        Args:
            n_recommendations: Nombre de recommandations
            strategy: 'popular' (items les plus notés) ou 'highly_rated' (moyenne la plus élevée)
            
        Returns:
            Liste de tuples (item_id, score)
        """
        if self.user_item_matrix is None:
            raise ValueError("Le modèle doit être entraîné")
        
        recommendations = []
        
        if strategy == 'popular':
            # Items avec le plus de notes
            item_counts = np.sum(self.user_item_matrix > 0, axis=0)
            top_items = np.argsort(item_counts)[::-1][:n_recommendations]
            
            for item_idx in top_items:
                item_id = self.reverse_item_mapping[item_idx]
                score = item_counts[item_idx] / self.user_item_matrix.shape[0]  # Normalisation
                recommendations.append((item_id, score))
                
        elif strategy == 'highly_rated':
            # Items avec la meilleure moyenne
            item_means = []
            for item_idx in range(self.user_item_matrix.shape[1]):
                ratings = self.user_item_matrix[:, item_idx]
                ratings = ratings[ratings > 0]
                if len(ratings) > 0:
                    item_means.append((item_idx, np.mean(ratings)))
                else:
                    item_means.append((item_idx, 0))
            
            # Trier par moyenne décroissante
            item_means.sort(key=lambda x: x[1], reverse=True)
            top_items = item_means[:n_recommendations]
            
            for item_idx, mean_rating in top_items:
                item_id = self.reverse_item_mapping[item_idx]
                recommendations.append((item_id, mean_rating))
        
        return recommendations
    
    def evaluate_performance(self, test_df: pd.DataFrame, similarity_type: str = 'cosine') -> Dict:
        """
        Évalue les performances du modèle sur un jeu de test
        
        Args:
            test_df: DataFrame de test avec colonnes [user_id, item_id, rating]
            similarity_type: Type de similarité à utiliser
            
        Returns:
            Dictionnaire avec les métriques de performance
        """
        if self.user_item_matrix is None:
            raise ValueError("Le modèle doit être entraîné")
        
        predictions = []
        actuals = []
        
        print("Evaluation des performances...")
        
        for _, row in test_df.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            actual_rating = row['rating']
            
            # Vérifier si l'utilisateur et l'item existent dans le modèle
            if user_id not in self.user_mapping or item_id not in self.item_mapping:
                continue
            
            # Obtenir les recommandations pour cet utilisateur
            try:
                recommendations = self.recommend_items_with_similarity(
                    user_id, n_recommendations=self.user_item_matrix.shape[1],
                    similarity_type=similarity_type
                )
                
                # Trouver le score prédit pour cet item
                predicted_score = 0
                for rec_item_id, score in recommendations:
                    if rec_item_id == item_id:
                        predicted_score = score
                        break
                
                if predicted_score > 0:
                    predictions.append(predicted_score)
                    actuals.append(actual_rating)
                    
            except Exception:
                continue
        
        if len(predictions) == 0:
            return {"error": "Aucune prédiction valide générée"}
        
        # Calcul des métriques
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        
        # Calculer la corrélation
        if len(predictions) > 1:
            correlation, _ = pearsonr(predictions, actuals)
        else:
            correlation = 0
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'correlation': correlation,
            'n_predictions': len(predictions)
        }
        
        self.performance_metrics[similarity_type] = metrics
        return metrics
    
    def benchmark_similarities(self, test_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Compare les performances des différentes méthodes de similarité
        
        Args:
            test_df: DataFrame de test (optionnel)
            
        Returns:
            DataFrame comparatif des performances
        """
        methods = ['cosine', 'pearson', 'euclidean']
        results = []
        
        for method in methods:
            print(f"Test de la méthode: {method}")
            
            if test_df is not None:
                # Évaluation sur jeu de test
                start_time = time.time()
                metrics = self.evaluate_performance(test_df, method)
                end_time = time.time()
                
                result = {
                    'method': method,
                    'rmse': metrics.get('rmse', 0),
                    'mae': metrics.get('mae', 0),
                    'correlation': metrics.get('correlation', 0),
                    'n_predictions': metrics.get('n_predictions', 0),
                    'time_seconds': end_time - start_time
                }
            else:
                # Benchmark de performance (temps de calcul)
                start_time = time.time()
                # Test avec un utilisateur aléatoire
                if len(self.user_mapping) > 0:
                    test_user = list(self.user_mapping.keys())[0]
                    _ = self.recommend_items_with_similarity(test_user, 10, method)
                end_time = time.time()
                
                result = {
                    'method': method,
                    'time_seconds': end_time - start_time
                }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def optimize_memory_usage(self):
        """
        Optimise l'utilisation mémoire en convertissant les matrices en formats plus compacts
        """
        if self.user_item_matrix is not None:
            # Conversion en float32 pour économiser de la mémoire
            self.user_item_matrix = self.user_item_matrix.astype(np.float32)
            
        if self.item_similarity_matrix is not None:
            self.item_similarity_matrix = self.item_similarity_matrix.astype(np.float32)
            
        if self.pearson_similarity_matrix is not None:
            self.pearson_similarity_matrix = self.pearson_similarity_matrix.astype(np.float32)
            
        if self.euclidean_similarity_matrix is not None:
            self.euclidean_similarity_matrix = self.euclidean_similarity_matrix.astype(np.float32)
        
        print("Optimisation mémoire terminée")

def create_train_test_split(df: pd.DataFrame, test_ratio: float = 0.2, 
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Crée un split train/test pour l'évaluation
    
    Args:
        df: DataFrame complet
        test_ratio: Proportion de données pour le test
        random_state: Graine aléatoire
        
    Returns:
        Tuple (train_df, test_df)
    """
    np.random.seed(random_state)
    
    # Mélanger aléatoirement les données
    shuffled_df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calculer le point de split
    split_idx = int(len(shuffled_df) * (1 - test_ratio))
    
    train_df = shuffled_df.iloc[:split_idx].copy()
    test_df = shuffled_df.iloc[split_idx:].copy()
    
    return train_df, test_df

def analyze_cold_start_scenario(recommender: AdvancedRecommender, 
                               new_user_ratings: Dict[int, float]) -> Dict:
    """
    Analyse le comportement du système dans un scénario de cold start
    
    Args:
        recommender: Le système de recommandation entraîné
        new_user_ratings: Dictionnaire {item_id: rating} pour le nouvel utilisateur
        
    Returns:
        Analyse du cold start
    """
    analysis = {
        'n_ratings_provided': len(new_user_ratings),
        'recommendations_with_data': [],
        'recommendations_cold_start': [],
        'coverage_improvement': 0
    }
    
    # Simuler un utilisateur avec quelques notes
    if len(new_user_ratings) > 0:
        # Ajouter temporairement l'utilisateur au modèle
        temp_user_id = max(recommender.user_mapping.keys()) + 1
        
        # Créer un vecteur pour le nouvel utilisateur
        new_user_vector = np.zeros(recommender.user_item_matrix.shape[1])
        for item_id, rating in new_user_ratings.items():
            if item_id in recommender.item_mapping:
                item_idx = recommender.item_mapping[item_id]
                new_user_vector[item_idx] = rating
        
        # Ajouter l'utilisateur temporaire
        temp_matrix = np.vstack([recommender.user_item_matrix, new_user_vector])
        
        # Calculer les similarités avec cet utilisateur
        user_similarities = cosine_similarity([new_user_vector], recommender.user_item_matrix)[0]
        
        # Trouver les utilisateurs les plus similaires
        similar_users = np.argsort(user_similarities)[::-1][:5]
        
        analysis['similar_users'] = similar_users.tolist()
        analysis['max_similarity'] = float(np.max(user_similarities))
    
    # Obtenir les recommandations cold start
    cold_start_recs = recommender.handle_cold_start(n_recommendations=10)
    analysis['recommendations_cold_start'] = cold_start_recs
    
    return analysis

if __name__ == "__main__":
    # Test des fonctionnalités avancées
    print("Test des fonctionnalités avancées...")
    
    # Créer un dataset plus grand pour les tests
    from recommender_system import create_sample_dataset
    df = create_sample_dataset(n_users=100, n_items=50, n_ratings=3000)
    
    # Split train/test
    train_df, test_df = create_train_test_split(df, test_ratio=0.2)
    print(f"Train: {len(train_df)} évaluations, Test: {len(test_df)} évaluations")
    
    # Entraîner le modèle avancé
    advanced_recommender = AdvancedRecommender(min_ratings_per_item=3, min_ratings_per_user=3)
    advanced_recommender.fit(train_df)
    
    # Calculer les similarités alternatives
    advanced_recommender.compute_alternative_similarities()
    
    # Comparer les méthodes de similarité
    test_user = train_df['user_id'].iloc[0]
    print(f"\nComparaison des méthodes pour l'utilisateur {test_user}:")
    comparison = advanced_recommender.compare_similarity_methods(test_user, n_recommendations=10)
    print(comparison)
    
    # Benchmark des performances
    print(f"\nBenchmark des performances:")
    benchmark_results = advanced_recommender.benchmark_similarities(test_df)
    print(benchmark_results)
    
    # Test du cold start
    print(f"\nTest du scénario de cold start:")
    cold_start_recs = advanced_recommender.handle_cold_start(n_recommendations=5, strategy='popular')
    print("Recommandations cold start (popular):", cold_start_recs)
    
    cold_start_recs_high = advanced_recommender.handle_cold_start(n_recommendations=5, strategy='highly_rated')
    print("Recommandations cold start (highly_rated):", cold_start_recs_high)
    
    print("\nTests des fonctionnalités avancées terminés!")
