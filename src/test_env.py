from dotenv import load_dotenv
import os
load_dotenv()
key = os.getenv("GROQ_API_KEY")
print("Key downloaded:" "YES" if key else "NO")

from src.config import MODEL_NAME
