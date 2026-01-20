import os

from dotenv import load_dotenv
from google import genai


def load_gemini_key():
    """Load the Gemini API key from .env file,

    the name of the variable is GEMINI_API_KEY
    """
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not found in .env file.")
    return key


class QueryCorrector:
    def __init__(self):
        api_key = load_gemini_key()
        self.client = genai.Client(api_key=api_key)

    def correct_query(self, query: str):
        """Corrects spelling in search queries and returns only the corrected query"""

        prompt = f"""
               Você é um corretor ortográfico especializado em consultas curtas em português.
               Corrija somente erros ortográficos ou gramaticais leves.
               Não altere o significado.
               Retorne APENAS a consulta corrigida.

               Consulta original: "{query}"
               Consulta corrigida:
               """

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-lite", contents=prompt
            )
        except Exception as e:
            print("Gemini error:", e)
            return query

        corrected = (response.text or "").strip()

        if corrected.lower().startswith("consulta corrigida"):
            corrected = corrected.split(":", 1)[-1].strip()

        return corrected
