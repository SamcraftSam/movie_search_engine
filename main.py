#!/usr/bin/env python3

import os
import pandas as pd
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class Neo4jConnector:
    URI = "bolt://127.0.0.1:7687"
    USER = "neo4j"
    PASSWORD = "password"
    
    def __init__(self):
        self.driver = GraphDatabase.driver(self.URI, auth=(self.USER, self.PASSWORD))
    
    def close(self):
        self.driver.close()
    
    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    def create_constraints(self):
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT movie_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.movieId IS UNIQUE")
            session.run("CREATE CONSTRAINT genre_name IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE")
    
    def ingest_movies(self, movies_df):
        with self.driver.session() as session:
            for idx, row in movies_df.iterrows():
                movie_id = int(row['movieId'])
                title = row['title']
                genres = row['genres'].split('|')
                
                session.run(
                    """
                    MERGE (m:Movie {movieId: $movieId})
                    SET m.title = $title
                    """, movieId=movie_id, title=title
                )
                
                for genre in genres:
                    if genre != "(no genres listed)":
                        session.run(
                            """
                            MERGE (g:Genre {name: $genreName})
                            """, genreName=genre
                        )
                        session.run(
                            """
                            MATCH (m:Movie {movieId: $movieId}), (g:Genre {name: $genreName})
                            MERGE (m)-[:HAS_GENRE]->(g)
                            """, movieId=movie_id, genreName=genre
                        )
    
    def get_movies_by_genre(self, genre, limit=5):
        results = []
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre {name: $genre})
                RETURN m.title AS title
                LIMIT $limit
                """, genre=genre, limit=limit
            )
            for record in result:
                results.append(record["title"])
        return results


class FaissIndexer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.titles = None
    
    def build_index(self, titles):
        self.titles = titles
        embeddings = self.model.encode(titles)
        embeddings = np.array(embeddings, dtype=np.float32)
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        
        return self.index
    
    def search(self, query_text, k=5):
        if not isinstance(query_text, list):
            query_text = [query_text]
            
        query_emb = self.model.encode(query_text)
        query_emb = np.array(query_emb, dtype=np.float32)
        
        distances, indices = self.index.search(query_emb, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append((self.titles[idx], dist))
            
        return results


class MovieRecommender:
    def __init__(self):
        self.neo4j = Neo4jConnector()
        self.faiss = FaissIndexer()
        self.movies_df = None
    
    def load_data(self, data_path="data/ml-latest-small"):
        movies_csv = os.path.join(data_path, "movies.csv")
        if not os.path.exists(movies_csv):
            raise FileNotFoundError("movies.csv not found.")
        
        print("Загружаю movies.csv...")
        self.movies_df = pd.read_csv(movies_csv)

    def ingest_into_neo4j(self):
        self.neo4j.ingest_movies(self.movies_df)

    def build_faiss_index(self):
        self.faiss.build_index(self.movies_df['title'].tolist())

    def get_movies_by_genre(self, genre, limit=5):
        print(f"\n=== Neo4j querry by genre: '{genre}' ===")
        genre_movies = self.neo4j.get_movies_by_genre(genre, limit)
        for movie in genre_movies:
            print(" -", movie)
        return genre_movies

    def get_similar_movies(self, query_text):
        print(f"\n=== Using FAISS querry to search movies, similar to {query_text}...")
        results = self.faiss.search(query_text)
        print("Top-5 similar:", query_text[0])
        for title, dist in results[:5]:
            print(f" - {title} (distance={dist:.4f})")

def main():
    recommender = MovieRecommender()
    recommender.load_data()
    recommender.ingest_into_neo4j()
    recommender.build_faiss_index()
    
    while True:
        prompt = str(input("Enter your prompt: "))
        recommender.get_movies_by_genre(prompt)
    
        recommender.get_similar_movies([prompt])

if __name__ == "__main__":
    main()
