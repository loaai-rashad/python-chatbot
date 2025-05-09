import os
from data_processing.pdf_extractor import extract_text_from_pdf
from data_processing.text_chunker import chunk_text
from embedding.embedding_generator import load_model, generate_embeddings
from embedding.vector_index import build_faiss_index, search_faiss_index
import google.generativeai as genai

# Replace with your actual Gemini API key
GENAI_API_KEY = "AIzaSyDzTehB-COpWdpfg0aNQZPT6koHysFGSCc"

# Configure the Gemini API
genai.configure(api_key=GENAI_API_KEY)

# Select the Gemini model
MODEL_NAME = "gemini-2.0-flash"
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]

model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    generation_config=generation_config,
    safety_settings=safety_settings,
)


def generate_answer_with_gemini(question, context):
    """
    Uses the Gemini API to generate an answer based on the question and the provided context.
    """
    prompt_parts = [
        f"""You are a helpful chatbot answering questions based on the following context:
    {context}

    Question: {question}
    Answer:
    """
    ]

    try:
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        return f"Error generating answer with Gemini: {e}"


PDF_PATH = "data/college_rules.pdf"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
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
        print(f"Chunk {i + 1}: '{chunk}' (Distance: {distances[i]:.4f})")

    # Use Gemini to generate the answer
    answer = generate_answer_with_gemini(question, " ".join(relevant_chunks))

    return answer


if __name__ == "__main__":
    embedding_model, vector_index, text_chunks = setup_chatbot()

    if embedding_model and vector_index and text_chunks:
        while True:
            user_question = input("\nAsk your question (or type 'exit' to quit): ")
            if user_question.lower() == "exit":
                break

            relevant_information = query_chatbot(
                user_question, embedding_model, vector_index, text_chunks
            )

            print("\nMost Relevant Information:")
            print(relevant_information)
