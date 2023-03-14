import pickle
import openai
from dotenv import load_dotenv
import os
import tiktoken
import json
import re


INDEX_FILE = "index.pickle"
ROW_DATA = "./micce/data/data.json"

load_dotenv()
openai.api_key: str = os.environ.get("OPENAI_API_KEY")
# 参考：https://zenn.dev/microsoft/articles/3438cf410cc0b5
# NOTE: tokenizeとutf-8 encodeを行っている
enc = tiktoken.get_encoding("cl100k_base")


embetting_res_dummy = json.load(open("./micce/data/embetting_res_dummy.json"))

class Index:
    def __init__(self, name = INDEX_FILE) -> None:
        self.name = name
        try:
            self.data = pickle.load(open(self.name, "rb"))
        except:
            self.data = {}

    def print(self) -> None:
        print(self.content)

def embetting(body: str):
    EMBED_MAX_SIZE = 8191
    body = body.replace("\n", " ")
    encoded_body = enc.encode(body)
    # 参考：https://platform.openai.com/docs/guides/embeddings/use-cases
    # NOTE: token上限は8191 tokensなので、足切りをしてあげる
    # NOTE: 1000 tokenあたりに、$0.0004かかるから叩き過ぎには注意
    if len(encoded_body) > EMBED_MAX_SIZE:
        encoded_body = encoded_body[:EMBED_MAX_SIZE]

    res = embetting_res_dummy 
    # NOTE: 無駄にapi叩かないようにdummyデータを利用
    # res = openai.Embedding.create(
    #     input=[encoded_body],
    #     model="text-embedding-ada-002"
    # )
    print(res["data"][0]["embedding"])

def create_index():
    with open(ROW_DATA) as f:
        data = json.load(f)
    buf = []
    for page in data["pages"]:
        title = page["title"]
        for line in data["pages"][0]["lines"]:
            line = line.strip()
            line = re.sub(r"https?://[^\s]+", "URL", line)
            line = re.sub(r"[\s]+", " ", line)
            buf.append(line)
        body = " ".join(buf).strip()
    embetting(body)
    # print(body)
        

    # index: Index = Index("初めまして！！！")
    # # pickleの説明：https://qiita.com/hatt0519/items/f1f4c059c28cb1575a93
    # with open(INDEX_FILE, "wb") as f:
    #     pickle.dump(index, f)


if __name__ == "__main__":
    create_index()
