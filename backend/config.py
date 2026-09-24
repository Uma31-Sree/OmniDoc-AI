import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MIXEDBREAD_API_KEY = os.getenv("MIXEDBREAD_API_KEY")

CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "omnidoc_collection"