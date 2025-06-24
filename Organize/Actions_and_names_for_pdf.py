import pandas as pd
import yfinance as yf
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

def gerar_pdf_com_nomes(csv_entrada, pdf_saida):
    df = pd.read_csv(csv_entrada)
    tickers = df['Ticker'].dropna().unique()

    c = canvas.Canvas(pdf_saida, pagesize=letter)
    width, height = letter
    margem = 50
    y = height - margem
    linha_altura = 14

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margem, y, "Ticker e Nome das Empresas")
    y -= 2 * linha_altura

    c.setFont("Helvetica", 12)

    for ticker in tickers:
        yf_ticker = ticker + ".SA"
        try:
            acao = yf.Ticker(yf_ticker)
            info = acao.info
            nome_empresa = info.get("longName") or info.get("shortName") or "Nome não disponível"
        except Exception:
            nome_empresa = "Erro ao buscar nome"

        texto = f"{ticker}: {nome_empresa}"
        c.drawString(margem, y, texto)
        y -= linha_altura

        # Nova página se chegar no fim
        if y < margem:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - margem

    c.save()
    print(f"PDF gerado: {pdf_saida}")

if __name__ == "__main__":
    gerar_pdf_com_nomes("precos_acoes.csv", "tickers_e_nomes.pdf")
