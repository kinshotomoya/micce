import pickle

class Index:
    def __init__(self, content: str) -> None:
        self.content = content
    
    def print(self) -> None:
        print(self.content)

index: Index = Index("初めまして！！！")
with open("index.pickle", "wb") as f:
    pickle.dump(index, f)

