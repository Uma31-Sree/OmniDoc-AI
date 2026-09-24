import google.generativeai as genai
from .config import GEMINI_API_KEY


class Summarizer:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel("gemini-3.6-flash")

    def summarize_text(self, text_chunk: str) -> str:
        """Generates a concise semantic summary of a text chunk."""
        if not text_chunk.strip():
            return ""
        prompt = (
            "Summarize the following text in 2-3 sentences for retrieval purposes. "
            "Keep it factual and concise. Preserve any important numbers, names, and dates.\n\n"
            f"{text_chunk}"
        )
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"[WARN] Gemini summarization failed: {e}")
            return text_chunk[:200]

    def summarize_text_pair(self, text_chunk: str) -> tuple:
        """Returns BOTH the original text and its summary."""
        summary = self.summarize_text(text_chunk)
        return text_chunk, summary

    def describe_image(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        """Generates a text description of an image using Gemini."""
        try:
            response = self.model.generate_content([
                {"mime_type": mime_type, "data": image_bytes},
                "Describe this image in detail for retrieval. Include any text, charts, or diagrams."
            ])
            return response.text.strip()
        except Exception as e:
            print(f"[WARN] Gemini image description failed: {e}")
            return ""