from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = 'all-MiniLM-L6-v2'

def load_model():
    return SentenceTransformer(MODEL_NAME)

def generate_embeddings(texts, model):
    embeddings = model.encode(texts)
    return np.array(embeddings).astype('float32')

if __name__ == '__main__':
    model = load_model()
    sample_texts = ["This is the first sentence.", "Here is the second sentence."]
    embeddings = generate_embeddings(sample_texts, model)
    print(f"Shape of embeddings: {embeddings.shape}")
    print(f"First embedding: {embeddings[0][:10]}")