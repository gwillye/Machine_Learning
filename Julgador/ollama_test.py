import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"

prompt = {
    "model": "gemma2:27b",
    "prompt": "Qual o nome do presidente do Brasil?",
    "system": "Você é um assistente útil.",
    "stream": False
}

response = httpx.post(OLLAMA_URL, json=prompt, timeout=60)
print(response.json()["response"])