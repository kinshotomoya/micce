import pickle
from index import Index

with open("index.pickle", "rb") as f:
    index: Index = pickle.load(f)

index.print()

