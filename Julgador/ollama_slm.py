import pandas as pd
import httpx
import json
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma2:27b"
print("Configurações lidas com sucesso.")

SYSTEM_PROMPT = """
Você é um especialista em classificação fiscal segundo a Nomenclatura Comum do Mercosul (NCM). 
Ao receber a descrição de um produto e os rótulos atribuídos por um cliente e uma IA, 
você deve decidir qual capítulo NCM é o mais apropriado com base na descrição e nas definições oficiais da NCM.
A decisão deve considerar todas as opções da base NCM fornecida como contexto.

Responda no seguinte formato JSON:
{
  "capitulo_correto": "<código do capítulo>",
  "explicacao": "<IA | CLIENTE | INCONCLUSIVO>: justificativa"
}
"""
print("Prompt de sistema configurado com sucesso.") 

def carregar_contexto(path, caps):
    df = pd.read_csv(path)
    caps_str = [str(c).zfill(2) for c in caps]
    filtrado = df[df['Codigo'].astype(str).str.zfill(2).isin(caps_str)]
    linhas = [f"Capítulo {row['Codigo']}: {row['Descricao']}" for _, row in filtrado.iterrows()]
    return "\n".join(linhas)

def carregar_rag_filtrado(rag_path, caps_interesse):
    df = pd.read_csv(rag_path)
    caps_str = [str(c).zfill(2) for c in caps_interesse]

    filtrado = df[df['Codigo'].astype(str).str.zfill(2).isin(caps_str)]
    linhas = [f"Capítulo {row['Codigo']}: {row['Descricao']}" for _, row in filtrado.iterrows()]
    return "\n".join(linhas)

def carregar_rag(path="ncm_reduzido.csv") -> str:
    df = pd.read_csv(path)
    linhas = [f"Capítulo {row['Codigo']}: {row['Descricao']}" for _, row in df.iterrows()]
    return "\n".join(linhas)

def avaliar_arquivo(input_csv="dados.csv", output_csv="output.csv",
                    path_capitulos="ncm_capitulos.csv", path_completo="ncm_reduzido.csv"):

    df = pd.read_csv(input_csv)
    resultados = []

    for _, row in df.iterrows():
        cliente = row["rotulo_cliente"]
        ia = row["rotulo_ia"]
        produto = row["prod_xprod"]

        # Passo 1: títulos apenas
        contexto = carregar_contexto(path_capitulos, {cliente, ia})
        prompt = gerar_user_prompt(produto, cliente, ia, contexto)
        resp = perguntar_ao_modelo(SYSTEM_PROMPT, prompt)

        # Passo 2: se inconclusivo, descrições dos dois capítulos
        if "INCONCLUSIVO" in resp.get("explicacao", "").upper():
            contexto = carregar_contexto(path_completo, {cliente, ia})
            prompt = gerar_user_prompt(produto, cliente, ia, contexto)
            resp = perguntar_ao_modelo(SYSTEM_PROMPT, prompt)

        # Passo 3: se ainda inconclusivo, todo NCM
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

    pd.DataFrame(resultados).to_csv(output_csv, index=False, encoding="utf-8")


print("Base de dados NCM carregada com sucesso.")

SYSTEM_PROMPT = """
Você classifica produtos na NCM.
Entrada: descrição, capítulo do cliente e capítulo da IA, e contexto.
Saída: JSON curto com:
{"capitulo_correto": "XX", "explicacao": "<IA | CLIENTE | INCONCLUSIVO>: <motivo com até 10 palavras>"}
"""

def gerar_user_prompt(prod_xprod, rotulo_cliente, rotulo_ia, contexto_rag):
    return f"""
Produto: {prod_xprod}
Cliente: {rotulo_cliente}
IA: {rotulo_ia}
Capítulos disponíveis:
{contexto_rag}
"""
print("Função de geração de prompt configurada com sucesso.")

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

print("Função de consulta ao modelo configurada com sucesso.")

def avaliar_arquivo(input_csv="dados.csv", output_csv="output.csv", rag_path="ncm_reduzido.csv"):
    caps = {rotulo_cliente, rotulo_ia}
    contexto = carregar_rag_filtrado(rag_path, caps)
    
    # Tentar diferentes separadores e configurações
    try:
        # Primeiro tenta com TAB
        df = pd.read_csv(input_csv, sep='\t', encoding='utf-8')
        print(f"Arquivo lido com separador TAB. Shape: {df.shape}")
    except:
        try:
            # Depois tenta com vírgula
            df = pd.read_csv(input_csv, sep=',', encoding='utf-8')
            print(f"Arquivo lido com separador vírgula. Shape: {df.shape}")
        except:
            try:
                # Tenta com vírgula e tratamento de erros
                df = pd.read_csv(input_csv, sep=',', encoding='utf-8', error_bad_lines=False, warn_bad_lines=True)
                print(f"Arquivo lido com tratamento de erros. Shape: {df.shape}")
            except Exception as e:
                print(f"Erro ao ler o arquivo: {e}")
                return
    
    print("Colunas encontradas:", df.columns.tolist())
    print("Primeiras linhas:")
    print(df.head())
    
    # Verificar se as colunas necessárias existem
    required_cols = ['prod_xprod', 'rotulo_cliente', 'rotulo_ia']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"ERRO: Colunas obrigatórias não encontradas: {missing_cols}")
        print(f"Colunas disponíveis: {df.columns.tolist()}")
        return
    
    resultados = []

    for _, row in df.iterrows():
        prod_xprod = row["prod_xprod"]
        rotulo_cliente = row["rotulo_cliente"]
        rotulo_ia = row["rotulo_ia"]

        print(f"Classificando: {row['prod_xprod']}")

        prompt_user = gerar_user_prompt(prod_xprod, rotulo_cliente, rotulo_ia, contexto)
        resposta = perguntar_ao_modelo(SYSTEM_PROMPT, prompt_user)

        resultados.append({
            "prod_xprod": prod_xprod,
            "rotulo_cliente": rotulo_cliente,
            "rotulo_ia": rotulo_ia,
            "capitulo_correto": resposta.get("capitulo_correto", "ERRO"),
            "explicacao": resposta.get("explicacao", "Erro na resposta")
        })

    df_saida = pd.DataFrame(resultados)
    df_saida.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Arquivo '{output_csv}' gerado com sucesso.")

    df_saida = pd.DataFrame(resultados)
    df_saida.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Arquivo '{output_csv}' gerado com sucesso.")

print("Função de avaliação de arquivo configurada com sucesso.")

if __name__ == "__main__":
    avaliar_arquivo()