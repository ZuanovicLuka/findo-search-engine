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


class AnswerGenerator:
    def __init__(self):
        api_key = load_gemini_key()
        self.client = genai.Client(api_key=api_key)

    def is_question(self, query: str):
        """Check if the input string is a question in Portuguese"""
        prompt = (
            "Responda apenas com 'YES' ou 'NO'. "
            "A seguinte frase é uma pergunta em português?\n"
            f"Frase: {query}\n"
            "Resposta:"
        )

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
        except Exception as e:
            print("Gemini error:", e)
            return False

        return response.text == "YES"

    def generate_answer(self, query: str, document_content: str):
        """Generate an answer based on the document content and query"""
        prompt = f"""
            Responda à pergunta usando APENAS a informação do documento abaixo.
            Caso a informação necessária **não esteja presente** no documento,
            você pode usar o seu próprio conhecimento,
            MAS deve indicar explicitamente no final da resposta:
            "(Resposta gerada por IA, informação não encontrada no documento)".

            Pergunta:
            {query}

            Documento:
            {document_content}

            Instruções:
            1. Se a resposta estiver claramente no documento, responda normalmente.
            2. Se o documento não contiver informação suficiente, responda com seu conhecimento,
            mas obrigatoriamente inclua ao final:
            "(Resposta gerada por IA, informação não encontrada no documento)".

            Resposta:
            """
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
        except Exception as e:
            print("Gemini error:", e)
            return "Desculpe, ocorreu um erro ao gerar a resposta."

        return response.text
