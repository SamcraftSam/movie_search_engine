#!/usr/bin/env bash
set -e

echo "===== [1] Create Python virtual environment ====="
python -m venv venv
source venv/bin/activate

echo "===== [2] Install Python dependencies in venv ====="
pip install --upgrade pip
pip install --upgrade pip
pip install neo4j-driver faiss-cpu sentence-transformers pandas


echo "===== [3] Run Neo4j in Docker (requires Docker pre-installed) ====="
# If a container named neo4j-movies exists, remove it
if [ "$(docker ps -aq -f name=neo4j-movies)" ]; then
  docker stop neo4j-movies || true
  docker rm neo4j-movies || true
fi

sudo docker run -d \
  --name neo4j-movies \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5.6

echo "===== [4] Download MovieLens data ====="
mkdir -p data
cd data
if [ ! -f "ml-latest-small.zip" ]; then
  wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
fi
unzip -o ml-latest-small.zip
cd ..

echo "===== [5] Run main.py (loads data into Neo4j & builds FAISS index) ====="
python main.py

echo "===== Done! Check Neo4j at http://localhost:7474 (user=neo4j, pass=password) ====="
