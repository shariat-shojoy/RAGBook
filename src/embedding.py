from typing import List, Any
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
try:
    from src.data_loader import load_all_documents
except ImportError:  
    from data_loader import load_all_documents

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = []
        for chunk in chunks:
            metadata = getattr(chunk, "metadata", {}) or {}
            if isinstance(metadata, dict):
                meta_lines = []
                if metadata.get("title"):
                    meta_lines.append(f"Title: {metadata['title']}")
                if metadata.get("source"):
                    meta_lines.append(f"Source: {metadata['source']}")
                if metadata.get("author"):
                    meta_lines.append(f"Author: {metadata['author']}")
                if metadata.get("page") is not None:
                    meta_lines.append(f"Page: {metadata['page']}")
                if meta_lines:
                    texts.append("\n".join(meta_lines) + "\n\n" + chunk.page_content)
                    continue
            texts.append(chunk.page_content)

        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)
