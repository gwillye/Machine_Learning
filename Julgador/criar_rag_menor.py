import pandas as pd

# Ler o CSV original
df = pd.read_csv("ncm_reduzido.csv")

# Filtrar linhas que sejam apenas títulos de capítulo
df_capitulos = df[df["Codigo"].astype(str).str.match(r"^\d{2}$")].copy()

# Ordenar pelo código numérico
df_capitulos["Codigo"] = df_capitulos["Codigo"].astype(str).str.zfill(2)
df_capitulos = df_capitulos.sort_values("Codigo")

# Salvar em um novo arquivo CSV
df_capitulos.to_csv("ncm_capitulos.csv", index=False)

print(f"Arquivo 'ncm_capitulos.csv' gerado com {len(df_capitulos)} capítulos.")
