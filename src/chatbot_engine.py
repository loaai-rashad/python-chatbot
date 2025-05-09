import os
from src.data_processing.pdf_extractor import extract_text_from_pdf
from src.data_processing.text_chunker import chunk_text
from src.embedding.embedding_generator import load_model, generate_embeddings
from src.embedding.vector_index import build_faiss_index, search_faiss_index

PDF_PATH = 'data/college_rules.pdf'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIMENSION = 384  # For all-MiniLM-L6-v2

def setup_chatbot():
    print("Loading PDF...")
    pdf_text = extract_text_from_pdf(PDF_PATH)
    if isinstance(pdf_text, str) and pdf_text.startswith("Error"):
        print(pdf_text)
        return None, None, None

    print("Chunking text...")
    text_chunks = chunk_text(pdf_text)

    print("Loading embedding model...")
    embedding_model = load_model()

    print("Generating embeddings...")
    embeddings = generate_embeddings(text_chunks, embedding_model)

    print("Building FAISS index...")
    vector_index = build_faiss_index(embeddings, EMBEDDING_DIMENSION)

    return embedding_model, vector_index, text_chunks

def query_chatbot(question, embedding_model, vector_index, text_chunks, top_k=3):
    question_embedding = generate_embeddings([question], embedding_model)[0]
    distances, indices = search_faiss_index(vector_index, question_embedding, top_k)

    relevant_chunks = [text_chunks[i] for i in indices]

    print("\nRelevant Chunks:")
    for i, chunk in enumerate(relevant_chunks):
        print(f"Chunk {i+1}: '{chunk}' (Distance: {distances[i]:.4f})")

    return relevant_chunks

if __name__ == '__main__':
    embedding_model, vector_index, text_chunks = setup_chatbot()

    if embedding_model and vector_index and text_chunks:
        while True:
            user_question = input("\nAsk your question (or type 'exit' to quit): ")
            if user_question.lower() == 'exit':
                break

            relevant_chunks = query_chatbot(user_question, embedding_model, vector_index, text_chunks)

            print("\nMost Relevant Information:")
            for i, chunk in enumerate(relevant_chunks):
                print(f"--- Chunk {i+1} ---")
                print(chunk)