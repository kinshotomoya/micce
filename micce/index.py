import pickle
import dotenv
import openai

class Index:
    def __init__(self, content: str) -> None:
        self.content = content
    
    def print(self) -> None:
        print(self.content)

index: Index = Index("初めまして！！！")
# pickleの説明：https://qiita.com/hatt0519/items/f1f4c059c28cb1575a93
with open("index.pickle", "wb") as f:
    pickle.dump(index, f)


print(__name__)