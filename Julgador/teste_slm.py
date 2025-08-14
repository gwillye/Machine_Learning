import pandas as pd
import time
import requests
import json
from datetime import datetime

def criar_dados_exemplo():
    """
    Cria um dataset de exemplo simulando descrições de capítulos NCM
    """
    dados_exemplo = [
        # Capítulo 10 - Cereais
        {"capitulo": 10, "descricao": "Trigo e trigo para semear"},
        {"capitulo": 10, "descricao": "Centeio para consumo humano"},
        {"capitulo": 10, "descricao": "Cevada cervejeira e malteada"},
        {"capitulo": 10, "descricao": "Aveia em grãos descascados"},
        {"capitulo": 10, "descricao": "Milho em grãos, exceto para semeadura"},
        {"capitulo": 10, "descricao": "Arroz com casca (arroz paddy)"},
        {"capitulo": 10, "descricao": "Sorgo granífero em grãos"},
        {"capitulo": 10, "descricao": "Trigo sarraceno, painço e alpiste"},

        # Capítulo 30 - Produtos farmacêuticos
        {"capitulo": 30, "descricao": "Medicamentos para uso humano contendo penicilinas"},
        {"capitulo": 30, "descricao": "Vacinas para medicina humana"},
        {"capitulo": 30, "descricao": "Soros terapêuticos e outros constituintes do sangue"},
        {"capitulo": 30, "descricao": "Medicamentos homeopáticos"},
        {"capitulo": 30, "descricao": "Gel lubrificante íntimo e produtos similares"},
        {"capitulo": 30, "descricao": "Produtos farmacêuticos para uso veterinário"},
        {"capitulo": 30, "descricao": "Preparações contraceptivas à base de hormônios"},
        {"capitulo": 30, "descricao": "Géis e cremes farmacêuticos para uso tópico"},
        {"capitulo": 30, "descricao": "Suplementos alimentares em forma de gel"},
        {"capitulo": 30, "descricao": "Produtos de higiene íntima em gel"},
        {"capitulo": 30, "descricao": "Lubrificantes pessoais e géis íntimos"},
        {"capitulo": 30, "descricao": "Géis para massagem terapêutica"},
        {"capitulo": 30, "descricao": "Produtos cosméticos em gel para volume capilar"},

        # Outros capítulos para confundir
        {"capitulo": 33, "descricao": "Óleos essenciais e produtos de perfumaria"},
        {"capitulo": 21, "descricao": "Preparações alimentícias diversas"},
        {"capitulo": 15, "descricao": "Gorduras e óleos animais ou vegetais"},
    ]

    # Simula mais dados para ter volume similar ao real
    dados_expandidos = []
    for i in range(600):  # Simula ~15k linhas
        for item in dados_exemplo:
            dados_expandidos.append({
                "capitulo": item["capitulo"],
                "descricao": f"{item['descricao']} - variação {i+1}"
            })

    return pd.DataFrame(dados_expandidos)

def chamar_ollama(prompt, modelo="gemma2:27b"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": modelo,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192
        }
    }

    tokens_entrada = None
    tokens_saida = None
    resposta_texto = []

    try:
        with requests.post(url, json=payload, stream=True, timeout=300) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Pega texto parcial
                if "response" in data:
                    resposta_texto.append(data["response"])

                # Pega tokens se já veio no stream
                if "prompt_eval_count" in data:
                    tokens_entrada = data["prompt_eval_count"]
                if "eval_count" in data:
                    tokens_saida = data["eval_count"]

        return "".join(resposta_texto), tokens_entrada, tokens_saida

    except requests.exceptions.RequestException as e:
        return f"Erro ao conectar com Ollama: {e}", tokens_entrada, tokens_saida
    except Exception as e:
        return f"Erro inesperado: {e}", tokens_entrada, tokens_saida

def preparar_contexto_rag(df, produto):
    """
    Prepara o contexto RAG filtrando descrições relevantes
    """
    # Busca por palavras-chave relevantes
    palavras_chave = ["gel", "volume", "bio", "cosmetic", "capilar", "lubrificant", "íntim"]

    contexto_relevante = []

    # Adiciona sempre algumas descrições dos capítulos 10 e 30
    cap_10 = df[df['capitulo'] == 10].head(5)
    cap_30 = df[df['capitulo'] == 30].head(10)

    for _, row in cap_10.iterrows():
        contexto_relevante.append(f"Capítulo {row['capitulo']}: {row['descricao']}")

    for _, row in cap_30.iterrows():
        contexto_relevante.append(f"Capítulo {row['capitulo']}: {row['descricao']}")

    # Busca descrições que contenham palavras-chave relacionadas ao produto
    for palavra in palavras_chave:
        relevantes = df[df['descricao'].str.contains(palavra, case=False, na=False)].head(3)
        for _, row in relevantes.iterrows():
            desc = f"Capítulo {row['capitulo']}: {row['descricao']}"
            if desc not in contexto_relevante:
                contexto_relevante.append(desc)

    return "\n".join(contexto_relevante[:20])  # Limita para não sobrecarregar o contexto

def main():
    print("Iniciando teste de RAG com Gemma2:27b para classificação NCM")
    print("=" * 60)

    # Criar dados de exemplo
    print("Criando dataset de exemplo...")
    inicio_criacao = time.time()
    df_rag = criar_dados_exemplo()
    tempo_criacao = time.time() - inicio_criacao

    print(f"Dataset criado com {len(df_rag)} registros em {tempo_criacao:.2f} segundos")
    print(f"Capítulos presentes: {sorted(df_rag['capitulo'].unique())}")
    print()

    # Preparar contexto RAG
    produto_teste = "BIOGELIS VOLUMAX"
    print(f"Preparando contexto RAG para produto: '{produto_teste}'")
    inicio_rag = time.time()

    contexto_rag = preparar_contexto_rag(df_rag, produto_teste)
    tempo_rag = time.time() - inicio_rag

    print(f"Contexto RAG preparado em {tempo_rag:.2f} segundos")
    linhas_contexto = len(contexto_rag.split("\n"))
    print(f"Linhas de contexto selecionadas: {linhas_contexto}")


    print()

    # Criar prompt para o modelo
    prompt = f"""Você é um especialista em classificação fiscal NCM (Nomenclatura Comum do Mercosul).

Sua tarefa é analisar se o produto "BIOGELIS VOLUMAX" pertence ao:
- Capítulo 10 (Cereais)
- Capítulo 30 (Produtos farmacêuticos)
- Nenhum dos dois

Use as seguintes informações da base NCM como referência:

{contexto_rag}

Baseado nas descrições acima, analise o produto "BIOGELIS VOLUMAX" e responda:

1. A qual capítulo este produto pertence? (10, 30, ou nenhum dos dois)
2. Justifique sua resposta explicando o raciocínio
3. Cite as descrições NCM que mais se aproximam do produto

Resposta:"""

    # Chamar o modelo
    print("🤖 Enviando consulta para Gemma2:27b...")
    print(f"⏰ Horário da consulta: {datetime.now().strftime('%H:%M:%S')}")

    inicio_consulta = time.time()
    resposta = chamar_ollama(prompt)
    tempo_consulta = time.time() - inicio_consulta

    print(f"Resposta recebida em {tempo_consulta:.2f} segundos")
    print()

    # Exibir resultados
    print("RESULTADOS:")
    print("=" * 60)
    print(f"Tempo para preparar RAG: {tempo_rag:.2f} segundos")
    print(f"Tempo para consulta ao modelo: {tempo_consulta:.2f} segundos")
    print(f"Tempo total: {(tempo_rag + tempo_consulta):.2f} segundos")
    print()

    print("RESPOSTA DO GEMMA2:27B:")
    print("-" * 40)
    print(resposta)
    print()

    print("ESTATÍSTICAS:")
    print(f"• Total de registros no RAG: {len(df_rag):,}")
    print(f"• Contexto enviado: {len(contexto_rag.split())} palavras")
    print(f"• Tamanho do prompt: {len(prompt)} caracteres")

    # Verificar se o Ollama está rodando
    if "Erro ao conectar" in resposta:
        print()
        print("ATENÇÃO:")
        print("• Certifique-se de que o Ollama está rodando: ollama serve")
        print("• Verifique se o modelo está instalado: ollama pull gemma2:27b")
        print("• URL padrão do Ollama: http://localhost:11434")

if __name__ == "__main__":
    main()