import requests


class AnswerGenerator:
    def __init__(self, model: str = "llama3.2", ollama_url: str = "http://localhost:11434"):
        self.model = model
        self.ollama_url = ollama_url

    def generate(self, context: str, question: str) -> str:
        """Generates a grounded answer using retrieved context via Ollama (local Llama 3)."""
        prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context below.
If the context does not contain the answer, say "I cannot find this in the document."

Context:
{context}

Question: {question}

Answer:"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            return f"[ERROR] Failed to generate answer: {e}"