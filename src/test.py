import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(".env"))

print("KEY:", os.getenv("GROQ_API_KEY"))