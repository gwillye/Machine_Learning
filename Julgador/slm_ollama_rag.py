import pandas as pd
import httpx
import json
import re

OLLAMA_URL = "http://s0646.ms:11434/api/generate"
MODEL = "gemma2:27b"
print("Configurações lidas com sucesso.")

SYSTEM_PROMPT = """
Você classifica produtos na NCM.
Entrada: descrição, capítulo do cliente e capítulo da IA, e contexto.
Saída: JSON curto com:
{"capitulo_correto": "XX", "explicacao": "<IA | CLIENTE | INCONCLUSIVO>: <motivo com até 10 palavras>"}
"""

# ---------------- FUNÇÕES DE CARREGAMENTO ----------------

def carregar_contexto(path, caps):
    """Carrega apenas os capítulos especificados de um CSV."""
    df = pd.read_csv(path)
    caps_str = [str(c).zfill(2) for c in caps]
    filtrado = df[df['Codigo'].astype(str).str.zfill(2).isin(caps_str)]
    linhas = [f"Capítulo {row['Codigo']}: {row['Descricao']}" for _, row in filtrado.iterrows()]
    return "\n".join(linhas)

# ---------------- GERAÇÃO DE PROMPT ----------------

def gerar_user_prompt(prod_xprod, rotulo_cliente, rotulo_ia, contexto_rag):
    return f"""
Produto: {prod_xprod}
Cliente: {rotulo_cliente}
IA: {rotulo_ia}
Capítulos disponíveis:
{contexto_rag}
"""

# ---------------- CONSULTA AO MODELO ----------------

def perguntar_ao_modelo(system_prompt: str, user_prompt: str) -> dict:
    payload = {
        "model": MODEL,
        "prompt": user_prompt,
        "system": system_prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_predict": 256
        }
    }

    with httpx.Client(timeout=300.0) as client:
        response = client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        resposta_raw = response.json().get("response", "")

        json_match = re.search(r'\{.*\}', resposta_raw, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                return {"capitulo_correto": "ERRO", "explicacao": "INCONCLUSIVO: JSON inválido"}
        else:
            return {"capitulo_correto": "ERRO", "explicacao": "INCONCLUSIVO: Sem JSON detectado"}

# ---------------- FUNÇÃO PRINCIPAL ----------------

def avaliar_arquivo(input_csv="dados.csv", amostra_caio_com_tratamento="amostra_caio_com_tratamento.csv",
                    path_capitulos="ncm_capitulos.csv", path_completo="ncm_reduzido.csv"):

    # Leitura do arquivo de entrada
    try:
        df = pd.read_csv(input_csv, encoding='utf-8')
    except Exception as e:
        print(f"Erro ao ler o arquivo de entrada: {e}")
        return

    # Verifica colunas obrigatórias
    required_cols = ['prod_xprod', 'rotulo_cliente', 'rotulo_ia']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERRO: Colunas obrigatórias não encontradas: {missing_cols}")
        return

    resultados = []

    for _, row in df.iterrows():
        cliente = row["rotulo_cliente"]
        ia = row["rotulo_ia"]
        produto = row["prod_xprod"]

        print(f"Classificando: {produto}")

        # Primeira tentativa: apenas títulos
        contexto = carregar_contexto(path_capitulos, {cliente, ia})
        prompt = gerar_user_prompt(produto, cliente, ia, contexto)
        resp = perguntar_ao_modelo(SYSTEM_PROMPT, prompt)

        # Se inconclusivo: descrições dos dois capítulos
        if "INCONCLUSIVO" in resp.get("explicacao", "").upper():
            contexto = carregar_contexto(path_completo, {cliente, ia})
            prompt = gerar_user_prompt(produto, cliente, ia, contexto)
            resp = perguntar_ao_modelo(SYSTEM_PROMPT, prompt)

        # Se ainda inconclusivo: toda a NCM
        if "INCONCLUSIVO" in resp.get("explicacao", "").upper():
            contexto = carregar_contexto(path_completo, range(1, 98))
            prompt = gerar_user_prompt(produto, cliente, ia, contexto)
            resp = perguntar_ao_modelo(SYSTEM_PROMPT, prompt)

        resultados.append({
            "prod_xprod": produto,
            "rotulo_cliente": cliente,
            "rotulo_ia": ia,
            "capitulo_correto": resp.get("capitulo_correto", ""),
            "explicacao": resp.get("explicacao", "")
        })

    pd.DataFrame(resultados).to_csv(amostra_caio_com_tratamento.csv, index=False, encoding="utf-8")
    print(f"Arquivo '{amostra_caio_com_tratamento}' gerado com sucesso.")

# ---------------- EXECUÇÃO ----------------

if __name__ == "__main__":
    avaliar_arquivo()