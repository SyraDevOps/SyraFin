from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib.enums import TA_CENTER
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Arquivos
CSV_TICKERS = "precos_acoes.csv"
PDF_SAIDA = "relatorio_acoes_estiloso.pdf"
PASTA_GRAFICOS = "graficos_temp"

# Criar pasta temporária para gráficos
os.makedirs(PASTA_GRAFICOS, exist_ok=True)

# Preparar o documento
doc = SimpleDocTemplate(PDF_SAIDA, pagesize=A4)
styles = getSampleStyleSheet()
elements = []

# Capa
titulo_style = ParagraphStyle(name='TituloCapa', fontSize=28, alignment=TA_CENTER, textColor=colors.white, spaceAfter=20)
sub_style = ParagraphStyle(name='SubTituloCapa', fontSize=14, alignment=TA_CENTER, textColor=colors.white)

# Fundo da capa (manual)
def draw_background(canvas, doc):
    canvas.saveState()
    grad_start = colors.Color(0.3, 0, 0.4)
    grad_end = colors.black
    for i in range(100):
        canvas.setFillColorRGB(
            grad_start.red + (grad_end.red - grad_start.red) * (i / 100.0),
            grad_start.green + (grad_end.green - grad_start.green) * (i / 100.0),
            grad_start.blue + (grad_end.blue - grad_start.blue) * (i / 100.0)
        )
        canvas.rect(0, A4[1] * i / 100.0, A4[0], A4[1] / 100.0, stroke=0, fill=1)
    canvas.restoreState()

def capa(canvas, doc):
    draw_background(canvas, doc)
    canvas.setFont("Helvetica-Bold", 26)
    canvas.setFillColor(colors.white)
    canvas.drawCentredString(A4[0] / 2, A4[1] - 200, "Relatório de Ações B3")
    canvas.setFont("Helvetica", 14)
    canvas.drawCentredString(A4[0] / 2, A4[1] - 230, f"Data: {datetime.today().strftime('%d/%m/%Y')}")

# Página de Sumário (será gerado automaticamente)
sumario = [Paragraph("Sumário", styles['Heading1'])]
sumario.append(Spacer(1, 12))

# Lê os tickers do CSV
df = pd.read_csv(CSV_TICKERS)
tickers = df['Ticker'].dropna().unique()

# Loop por ticker
for i, ticker in enumerate(tickers):
    nome_img = f"{PASTA_GRAFICOS}/{ticker}.png"
    yf_ticker = ticker + ".SA"
    try:
        acao = yf.Ticker(yf_ticker)
        hist = acao.history(period="6mo")
        info = acao.info
        nome_empresa = info.get("longName") or info.get("shortName") or "Empresa desconhecida"
        preco_atual = info.get("regularMarketPrice") or info.get("previousClose", "N/D")
        setor = info.get("sector", "Setor desconhecido")

        # Gerar gráfico
        plt.figure(figsize=(6, 3))
        plt.plot(hist.index, hist['Close'], label='Fechamento')
        plt.title(f"{ticker} - {nome_empresa}")
        plt.xlabel("Data")
        plt.ylabel("Preço (R$)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(nome_img)
        plt.close()

        # Adiciona entrada ao sumário
        sumario.append(Paragraph(f"{ticker} - {nome_empresa}", styles['Normal']))
        sumario.append(Spacer(1, 6))

        # Adiciona página da ação
        elements.append(PageBreak())
        elements.append(Paragraph(f"{ticker} - {nome_empresa}", styles['Heading2']))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(f"Preço atual: R$ {preco_atual}", styles['Normal']))
        elements.append(Paragraph(f"Setor: {setor}", styles['Normal']))
        elements.append(Spacer(1, 12))
        elements.append(Image(nome_img, width=400, height=200))
        elements.append(Spacer(1, 12))

    except Exception as e:
        continue

# Montar PDF final
doc.build([PageBreak()] + [Spacer(1, 100)] + sumario + elements, onFirstPage=capa)

# Limpar gráficos temporários
import shutil
shutil.rmtree(PASTA_GRAFICOS)

