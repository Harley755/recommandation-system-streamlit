"""
Module de chargement des datasets - Support pour MovieLens et autres sources
Gestion des métadonnées et prétraitement des données
"""

import pandas as pd
import numpy as np
import requests
import os
from typing import Optional, Tuple, Dict
import zipfile
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """
    Classe utilitaire pour charger différents types de datasets
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialise le chargeur de données
        
        Args:
            data_dir: Répertoire pour stocker les datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_movielens_small(self, force_download: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Télécharge et charge le dataset MovieLens Small (100k ratings)
        
        Args:
            force_download: Force le re-téléchargement même si les données existent
            
        Returns:
            Tuple (ratings_df, movies_df)
        """
        ratings_file = self.data_dir / "ml-100k" / "u.data"
        movies_file = self.data_dir / "ml-100k" / "u.item"
        zip_file = self.data_dir / "ml-100k.zip"
        
        # Télécharger si nécessaire
        if not ratings_file.exists() or force_download:
            print("Téléchargement du dataset MovieLens 100K...")
            url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(zip_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extraire le zip
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            print("Dataset téléchargé et extrait!")
        
        # Charger les données
        print("Chargement des données MovieLens...")
        
        # Charger les ratings
        ratings_df = pd.read_csv(
            ratings_file,
            sep='\t',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            engine='python'
        )
        
        # Charger les films
        movies_df = pd.read_csv(
            movies_file,
            sep='|',
            names=['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + [f'genre_{i}' for i in range(19)],
            engine='python',
            encoding='latin-1'
        )
        
        # Nettoyage des métadonnées
        movies_df['year'] = pd.to_datetime(movies_df['release_date'], errors='coerce').dt.year
        movies_df['genres'] = movies_df[[f'genre_{i}' for i in range(19)]].apply(
            lambda row: '|'.join([f"Genre_{i}" for i, val in enumerate(row) if val == 1]), 
            axis=1
        )
        
        print(f"Dataset chargé: {len(ratings_df)} évaluations, {ratings_df['user_id'].nunique()} utilisateurs, {ratings_df['item_id'].nunique()} films")
        
        return ratings_df, movies_df
    
    def download_movielens_latest(self, force_download: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Télécharge et charge le dataset MovieLens Latest Small (25k ratings)
        
        Args:
            force_download: Force le re-téléchargement
            
        Returns:
            Tuple (ratings_df, movies_df)
        """
        ratings_file = self.data_dir / "ml-latest-small" / "ratings.csv"
        movies_file = self.data_dir / "ml-latest-small" / "movies.csv"
        zip_file = self.data_dir / "ml-latest-small.zip"
        
        if not ratings_file.exists() or force_download:
            print("Téléchargement du dataset MovieLens Latest Small...")
            url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(zip_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            print("Dataset téléchargé et extrait!")
        
        # Charger les données
        print("Chargement des données MovieLens Latest...")
        
        ratings_df = pd.read_csv(ratings_file)
        movies_df = pd.read_csv(movies_file)
        
        # Extraire l'année du titre
        movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)
        
        print(f"Dataset chargé: {len(ratings_df)} évaluations, {ratings_df['userId'].nunique()} utilisateurs, {ratings_df['movieId'].nunique()} films")
        
        # Renommer les colonnes pour la cohérence
        ratings_df = ratings_df.rename(columns={'userId': 'user_id', 'movieId': 'item_id'})
        movies_df = movies_df.rename(columns={'movieId': 'item_id'})
        
        return ratings_df, movies_df
    
    def create_enhanced_sample_dataset(self, n_users: int = 200, n_items: int = 100, 
                                     n_ratings: int = 5000, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Crée un dataset synthétique amélioré avec des métadonnées réalistes
        
        Args:
            n_users: Nombre d'utilisateurs
            n_items: Nombre d'items
            n_ratings: Nombre total d'évaluations
            random_state: Graine aléatoire
            
        Returns:
            Tuple (ratings_df, movies_df)
        """
        np.random.seed(random_state)
        
        # Générer les évaluations
        user_ids = np.random.randint(1, n_users + 1, n_ratings)
        item_ids = np.random.randint(1, n_items + 1, n_ratings)
        ratings = np.random.randint(1, 6, n_ratings)
        
        ratings_df = pd.DataFrame({
            'user_id': user_ids,
            'item_id': item_ids,
            'rating': ratings,
            'timestamp': np.random.randint(1000000000, 1600000000, n_ratings)
        })
        
        # Supprimer les doublons
        ratings_df = ratings_df.drop_duplicates(subset=['user_id', 'item_id'])
        
        # Ajouter de la structure (préférences utilisateur)
        for user_id in range(1, min(21, n_users + 1)):
            # Chaque utilisateur a une préférence pour certains genres
            preferred_items = range((user_id - 1) * 5 + 1, min(user_id * 5 + 1, n_items + 1))
            for item_id in preferred_items:
                if np.random.random() < 0.4:  # 40% de chance
                    if not ((ratings_df['user_id'] == user_id) & (ratings_df['item_id'] == item_id)).any():
                        new_rating = np.random.randint(4, 6)
                        ratings_df = pd.concat([ratings_df, pd.DataFrame([{
                            'user_id': user_id,
                            'item_id': item_id,
                            'rating': min(new_rating, 5),
                            'timestamp': np.random.randint(1000000000, 1600000000)
                        }])], ignore_index=True)
        
        # Créer les métadonnées des films
        genres = ['Action', 'Comedy', 'Drama', 'Thriller', 'Romance', 'Sci-Fi', 'Horror', 'Documentary']
        years = np.random.randint(1990, 2024, n_items)
        
        movies_data = []
        for item_id in range(1, n_items + 1):
            # Générer un titre réaliste
            year = years[item_id - 1]
            title_adjectives = ['Amazing', 'Incredible', 'Fantastic', 'Wonderful', 'Spectacular', 'Epic']
            title_nouns = ['Adventure', 'Journey', 'Story', 'Tale', 'Saga', 'Legend']
            title = f"{np.random.choice(title_adjectives)} {np.random.choice(title_nouns)} ({year})"
            
            # Assigner des genres
            n_genres = np.random.randint(1, 4)
            item_genres = '|'.join(np.random.choice(genres, n_genres, replace=False))
            
            movies_data.append({
                'item_id': item_id,
                'title': title,
                'genres': item_genres,
                'year': year
            })
        
        movies_df = pd.DataFrame(movies_data)
        
        print(f"Dataset synthétique créé: {len(ratings_df)} évaluations, {ratings_df['user_id'].nunique()} utilisateurs, {ratings_df['item_id'].nunique()} films")
        
        return ratings_df, movies_df
    
    def load_data(self, dataset_type: str = "synthetic", **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Charge un dataset selon le type spécifié
        
        Args:
            dataset_type: Type de dataset ('synthetic', 'movielens_100k', 'movielens_latest')
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Tuple (ratings_df, movies_df)
        """
        if dataset_type == "synthetic":
            return self.create_enhanced_sample_dataset(**kwargs)
        elif dataset_type == "movielens_100k":
            return self.download_movielens_small(kwargs.get('force_download', False))
        elif dataset_type == "movielens_latest":
            return self.download_movielens_latest(kwargs.get('force_download', False))
        else:
            raise ValueError(f"Type de dataset non supporté: {dataset_type}")
    
    def get_dataset_info(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> Dict:
        """
        Retourne des informations détaillées sur le dataset
        
        Args:
            ratings_df: DataFrame des évaluations
            movies_df: DataFrame des films
            
        Returns:
            Dictionnaire avec les statistiques
        """
        info = {
            'n_ratings': len(ratings_df),
            'n_users': ratings_df['user_id'].nunique(),
            'n_items': ratings_df['item_id'].nunique(),
            'rating_range': f"{ratings_df['rating'].min()}-{ratings_df['rating'].max()}",
            'avg_rating': ratings_df['rating'].mean(),
            'sparsity': 1 - (len(ratings_df) / (ratings_df['user_id'].nunique() * ratings_df['item_id'].nunique())),
            'ratings_per_user': len(ratings_df) / ratings_df['user_id'].nunique(),
            'ratings_per_item': len(ratings_df) / ratings_df['item_id'].nunique()
        }
        
        if 'year' in movies_df.columns:
            info['year_range'] = f"{int(movies_df['year'].min())}-{int(movies_df['year'].max())}"
            info['avg_year'] = movies_df['year'].mean()
        
        if 'genres' in movies_df.columns:
            all_genres = []
            for genres_str in movies_df['genres'].dropna():
                all_genres.extend(genres_str.split('|'))
            info['n_genres'] = len(set(all_genres))
            info['most_common_genre'] = max(set(all_genres), key=all_genres.count) if all_genres else None
        
        return info

if __name__ == "__main__":
    # Test du chargeur de données
    loader = DataLoader()
    
    print("Test du chargeur de données...")
    
    # Test dataset synthétique
    print("\n1. Dataset synthétique:")
    ratings, movies = loader.load_data("synthetic", n_users=50, n_items=30, n_ratings=1000)
    info = loader.get_dataset_info(ratings, movies)
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test MovieLens Latest Small
    print("\n2. MovieLens Latest Small:")
    try:
        ratings_ml, movies_ml = loader.load_data("movielens_latest")
        info_ml = loader.get_dataset_info(ratings_ml, movies_ml)
        for key, value in info_ml.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"  Erreur: {e}")
    
    print("\nTests terminés!")
