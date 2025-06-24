import re
import yfinance as yf
import pandas as pd
from PyPDF2 import PdfReader
from tqdm import tqdm

def extrair_tickers_nomes(pdf_path):
    reader = PdfReader(pdf_path)
    texto = ""
    for page in reader.pages:
        texto += page.extract_text() + "\n"

    padrao = r"([A-Z]{4,5}\d{0,2}): (.+)"
    correspondencias = re.findall(padrao, texto)
    return correspondencias

def buscar_precos(tickers):
    precos = {}
    for ticker in tqdm(tickers, desc="Buscando preços"):
        try:
            t = yf.Ticker(ticker + ".SA")
            preco = t.fast_info.get("last_price")
            if preco is None:
                preco = t.info.get("regularMarketPrice") or t.info.get("previousClose")
            precos[ticker] = preco if preco is not None else "N/D"
        except Exception:
            precos[ticker] = "Erro"
    return precos

def salvar_csv(tabela, caminho="dados_acoes.csv"):
    df = pd.DataFrame(tabela, columns=["Ticker", "Nome", "Preço Atual"])
    df.to_csv(caminho, index=False, encoding="utf-8-sig")
    print(f"✅ Arquivo CSV salvo: {caminho}")

if __name__ == "__main__":
    caminho_pdf = "tickers_e_nomes.pdf"
    pares = extrair_tickers_nomes(caminho_pdf)

    tickers = [p[0] for p in pares]
    nomes = {p[0]: p[1] for p in pares}

    precos = buscar_precos(tickers)

    dados_finais = [(ticker, nomes[ticker], precos[ticker]) for ticker in tickers]
    salvar_csv(dados_finais)
