"""
レコメンド

参考：https://github.com/openai/openai-cookbook/blob/main/examples/Recommendation_using_embeddings.ipynb

"""
import pickle
from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    tsne_components_from_embeddings,
    chart_from_components,
    indices_of_nearest_neighbors_from_distances,
)
from typing import Dict
from dotenv import load_dotenv
import openai
import os
import tiktoken
from typing import List, Dict, Tuple
from openai.datalib import numpy as np


EMBEDDING_MODEL = "text-embedding-ada-002"

load_dotenv()
openai.api_key: str = os.environ.get("OPENAI_API_KEY")
# 参考：https://zenn.dev/microsoft/articles/3438cf410cc0b5
# NOTE: tokenizeとutf-8 encodeを行っている
enc = tiktoken.get_encoding("cl100k_base")


def embedding(body: str):
    EMBED_MAX_SIZE = 8191
    body = body.replace("\n", " ")
    encoded_body = enc.encode(body)
    # 参考：https://platform.openai.com/docs/guides/embeddings/use-cases
    # NOTE: token上限は8191 tokensなので、足切りをしてあげる
    # NOTE: 1000 tokenあたりに、$0.0004かかるから叩き過ぎには注意
    if len(encoded_body) > EMBED_MAX_SIZE:
        encoded_body = encoded_body[:EMBED_MAX_SIZE]

    # NOTE: 無駄にapi叩かないように
    res = openai.Embedding.create(
        input=[encoded_body],
        model=EMBEDDING_MODEL
    )
    return res["data"][0]["embedding"]

with open("index.pickle", "rb") as f:
    index: Dict = pickle.load(f)

# 検索index
embeddings = list(index.values())

titles = [k[0] for k in list(index.keys())]

user_input = "LINEスタンプの購入方法知りたい"

# ユーザークエリのembeded
embedding_user_input = embedding(user_input)

# 類似度検索
distances: List[List] = distances_from_embeddings(embedding_user_input, embeddings, distance_metric="cosine")
indices_of_nearest_neighbors: np.ndarray = indices_of_nearest_neighbors_from_distances(distances)

print(distances)
print(indices_of_nearest_neighbors)

for i, recomend in enumerate(indices_of_nearest_neighbors):
    title = titles[recomend]
    print(f"---------レコメンド: {i+1}")
    print(title)
    print(f"類似度: {distances[recomend]}")

