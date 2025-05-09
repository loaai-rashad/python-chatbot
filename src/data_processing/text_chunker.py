def chunk_text(text, chunk_size=500, chunk_overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks

if __name__ == '__main__':
    sample_text = "This is a long piece of text that needs to be chunked into smaller segments for processing. We will use a fixed chunk size and some overlap to ensure context is preserved across chunks. This is the second sentence, continuing the example. And here is a third sentence to make it a bit longer."
    text_chunks = chunk_text(sample_text)
    print(f"Number of chunks: {len(text_chunks)}")
    for i, chunk in enumerate(text_chunks):
        print(f"Chunk {i+1}: '{chunk[:50]}...'")
