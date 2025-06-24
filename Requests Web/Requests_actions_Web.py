import requests
from bs4 import BeautifulSoup
import yfinance as yf
from tqdm import tqdm
import csv

def obter_acoes():
    url = "https://www.dadosdemercado.com.br/acoes"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Referer": "https://www.google.com/",
        "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    }

    res = requests.get(url, headers=headers)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")

    acoes = set()
    for a in soup.find_all("a", href=True):
        href = a['href']
        if href.startswith("/acoes/"):
            ticker = a.text.strip()
            if ticker:
                acoes.add(ticker)
    return sorted(acoes)

def obter_precos(acoes):
    resultados = []
    for ticker in tqdm(acoes, desc="Buscando preços"):
        yf_ticker = ticker + ".SA"
        try:
            acao = yf.Ticker(yf_ticker)
            info = acao.info
            preco = info.get("regularMarketPrice") or info.get("previousClose")
            if preco is not None:
                resultados.append((ticker, preco))
            else:
                resultados.append((ticker, "N/D"))
        except Exception:
            resultados.append((ticker, "Erro"))
    return resultados

def salvar_csv(dados, arquivo="precos_acoes.csv"):
    with open(arquivo, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Ticker", "Preço"])
        writer.writerows(dados)
    print(f"Dados salvos em {arquivo}")

if __name__ == "__main__":
    acoes = obter_acoes()
    print(f"Total de ações únicas encontradas: {len(acoes)}")

    precos = obter_precos(acoes)
    salvar_csv(precos)
