import pickle
import openai
from dotenv import load_dotenv
import os
import tiktoken
import json
import re
from typing import List,Dict


INDEX_FILE = "index.pickle"
ROW_DATA = "./micce/data/data.json"
EMBEDDING_MODEL = "text-embedding-ada-002"

load_dotenv()
openai.api_key: str = os.environ.get("OPENAI_API_KEY")
# 参考：https://zenn.dev/microsoft/articles/3438cf410cc0b5
# NOTE: tokenizeとutf-8 encodeを行っている
enc = tiktoken.get_encoding("cl100k_base")


embetting_res_dummy = json.load(open("./micce/data/embetting_res_dummy.json"))

def embedding(body: str):
    EMBED_MAX_SIZE = 8191
    body = body.replace("\n", " ")
    encoded_body = enc.encode(body)
    # 参考：https://platform.openai.com/docs/guides/embeddings/use-cases
    # NOTE: token上限は8191 tokensなので、足切りをしてあげる
    # NOTE: 1000 tokenあたりに、$0.0004かかるから叩き過ぎには注意
    if len(encoded_body) > EMBED_MAX_SIZE:
        encoded_body = encoded_body[:EMBED_MAX_SIZE]

    # NOTE: 無駄にapi叩かないようにdummyデータを利用
    res = openai.Embedding.create(
        input=[encoded_body],
        model=EMBEDDING_MODEL
    )
    return res["data"][0]["embedding"]

def create_index():
    index = Index()
    with open(ROW_DATA) as f:
        data: Dict = json.load(f)
    buf: List = []
    for page in data["pages"]:
        title: str = page["title"]
        for line in data["pages"][0]["lines"]:
            line = line.strip()
            line = re.sub(r"https?://[^\s]+", "URL", line)
            line = re.sub(r"[\s]+", " ", line)
            buf.append(line)
        body: str = " ".join(buf).strip()
        index.get_or_create(body, title)
    
class Index:
    def __init__(self, name = INDEX_FILE) -> None:
        self.name = name
        try:
            self.data = pickle.load(open(self.name, "rb"))
        except:
            self.data = {}

    def get_or_create(self, body: str, title: str):
        embetting_body: List = embedding(body)
        if body not in self.data:
            self.data[(title, body)] = embetting_body
            with open(INDEX_FILE, "wb") as f:
                pickle.dump(self.data, f)
        else:
            print("already exist")
                


if __name__ == "__main__":
    create_index()
