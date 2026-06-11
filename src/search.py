import os
from dotenv import load_dotenv
try:
    from src.vectorstore import FaissVectorStore
except ImportError:
    from vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "openai/gpt-oss-120b"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your environment or .env file, for example:\n"
                "GROQ_API_KEY=your_api_key_here"
            )
        llm_model = os.getenv("GROQ_MODEL", llm_model)
        try:
            self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize Groq LLM with model '{llm_model}'. "
                "Check GROQ_MODEL, your Groq account access, and whether that model exists. "
                "A common valid option is GROQ_MODEL=gemma2:8b."
            ) from exc
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def _truncate_text(self, text: str, max_chars: int = 300) -> str:
        text = text.strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "..."

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        context_lines = []
        for idx, r in enumerate(results, start=1):
            meta = r["metadata"] or {}
            title = meta.get("title") or "unknown title"
            source = meta.get("source") or meta.get("file_path") or "unknown source"
            page = meta.get("page") if meta.get("page") is not None else "unknown page"
            text = self._truncate_text(meta.get("text", ""))
            context_lines.append(
                f"Chunk {idx}:\n"
                f"Title: {title}\n"
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Text: {text}"
            )

        context = "\n\n---\n\n".join(context_lines)
        max_context_length = 3200
        if len(context) > max_context_length:
            context = context[:max_context_length].rstrip() + "\n\n...[truncated additional context]"

        if not context.strip():
            return "No relevant documents found."
        prompt = (
            "You are a helpful Examinar. Your main intesion is to create 5 MCQs with 4 options each and answers based on the provided context+your knowledge. "
            "you will get the querry and the context which is retrieved from the vectorstore. The context may contain multiple chunks of text from different sources."
            "For each question, mention the source title and page number if available. "
            "You can generate question in thee structure of a json file"
            "If the source or page number is unknown, say 'source unknown' or 'page unknown'.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:" 
        )
        response = self.llm.invoke([prompt])
        return response.content

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "generate 5 MCQs on Process Scheduling"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Answer:", summary)
