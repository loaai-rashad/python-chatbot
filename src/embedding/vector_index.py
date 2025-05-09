import faiss
import numpy as np

def build_faiss_index(embeddings, dimension):
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity
    index.add(embeddings)
    return index

def search_faiss_index(index, query_embedding, top_k=3):
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), top_k)
    return distances[0], indices[0]

if __name__ == '__main__':
    # Example usage
    embedding_dimension = 384  # For all-MiniLM-L6-v2
    sample_embeddings = np.random.rand(10, embedding_dimension).astype('float32')

    index = build_faiss_index(sample_embeddings, embedding_dimension)
    print("FAISS index built.")

    query = np.random.rand(embedding_dimension).astype('float32')
    distances, indices = search_faiss_index(index, query)
    print("Search results:")
    print(f"Distances: {distances}")
    print(f"Indices: {indices}")