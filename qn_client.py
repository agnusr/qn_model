# src/gemini_client.py

import os
from dotenv import load_dotenv
import google.generativeai as genai

# ✅ Load environment variables
load_dotenv()

# ✅ Get API key from .env
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("❌ API Key not found in .env file!")

# ✅ Configure Gemini SDK
genai.configure(api_key=API_KEY)

# ✅ AUTO PICK WORKING MODEL
def get_working_model():
    models = genai.list_models()
    for m in models:
        if "generateContent" in m.supported_generation_methods:
            print(f"✅ Using QN model")
            return m.name
    raise RuntimeError("❌ No usable  model found for your API key!")

# ✅ Select best available Gemini model automatically
GEMINI_MODEL = get_working_model()

# ✅ Ask Gemini safely
def ask_gemini(prompt: str) -> str:
    model = genai.GenerativeModel(GEMINI_MODEL)

    response = model.generate_content(prompt)

    return response.text
