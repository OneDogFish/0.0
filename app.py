import streamlit as st
import kagglehub
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

MAX_GAMES = 100000 
MAX_FEATURES = 100000 
N_RECOMMENDATIONS = 10 
MIN_POSITIVE_REVIEWS = 50

# 加载和准备数据
def load_and_prepare_data(kaggle_path):
    csv_files = [f for f in os.listdir(kaggle_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in path: {kaggle_path}")

    df = pd.read_csv(os.path.join(kaggle_path, csv_files[0]))
    
    if 'total_positive' in df.columns:
        df_filtered = df[df['total_positive'] > MIN_POSITIVE_REVIEWS].copy()
        df_filtered = df_filtered.sort_values(by='total_positive', ascending=False)
        df = df_filtered.head(MAX_GAMES)
    else:
        print("Warning: 'total_positive' column not found, skipping quality filter.")
        df = df.head(MAX_GAMES).copy()
        
    features_to_use = ['name', 'genres', 'categories', 'developers'] 
    
    for feature in features_to_use:
        if feature in df.columns:
            df[feature] = df[feature].fillna('')
        else:
            df[feature] = ''
    
    def combine_features_focused(row):
        return (row['genres'] + " " + row['genres'] + " " + row['categories'])

    df['combined_features'] = df.apply(combine_features_focused, axis=1)
    
    df = df.reset_index(drop=True)
    indices = pd.Series(df.index, index=df['name'].apply(lambda x: x.lower().strip())).drop_duplicates()
    
    return df, indices

# 训练推荐模型
def train_model(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=MAX_FEATURES)
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(tfidf_matrix)
    
    print(f"TF-IDF Matrix size: {tfidf_matrix.shape}")
    return tfidf_matrix, model_knn

# 获取推荐
def get_recommendations_knn(title, df, indices, tfidf_matrix, model_knn):
    normalized_title = title.lower().strip()
    
    if normalized_title not in indices:
        return f"Error: Game '{title}' not found."
    
    idx = indices[normalized_title]
    if isinstance(idx, pd.Series): idx = idx.iloc[0]
    
    query_vector = tfidf_matrix[idx]
    
    distances, neighbor_indices = model_knn.kneighbors(
        query_vector, 
        n_neighbors=N_RECOMMENDATIONS + 1
    )
    
    distances = distances.flatten()
    neighbor_indices = neighbor_indices.flatten()
    
    recommendation_indices = neighbor_indices[1:]
    
    similarity_scores = 1 - distances[1:]
    
    recommendations_df = df.iloc[recommendation_indices][['name', 'genres', 'categories', 'total_positive', 'developers']].copy()
    recommendations_df['similarity_score'] = similarity_scores
    
    return recommendations_df.sort_values(by='similarity_score', ascending=False)

# Streamlit Web 应用
def main():
    st.title('游戏推荐系统')
    
    try:
        path = kagglehub.dataset_download("srgiomanhes/steam-games-dataset-2025")
        st.write(f"Dataset path: {path}")
    except Exception as e:
        st.write(f"Kaggle download error: {e}")
        return
    
    df_games, game_indices = load_and_prepare_data(path)
    tfidf_mat, knn_model = train_model(df_games)

    user_input_game = st.text_input("请输入你喜欢的游戏名称", "DARK SOULS™ III")
    
    if user_input_game:
        st.write(f"\n推荐游戏：")
        recommendations = get_recommendations_knn(
            user_input_game, 
            df_games, 
            game_indices,  
            tfidf_mat,     
            knn_model
        )
        
        if isinstance(recommendations, str):
            st.write(recommendations)  # 如果没有找到推荐游戏
        else:
            st.dataframe(recommendations[['name', 'genres', 'categories', 'similarity_score']])

if __name__ == '__main__':
    main()
