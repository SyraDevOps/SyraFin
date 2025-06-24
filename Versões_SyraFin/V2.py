# relatorio_acoes_SOTA_refatorado.py

import os
import shutil
from datetime import datetime
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. CONFIGURAÇÕES GERAIS ---
CONFIG = {
    "tickers_csv": "precos_acoes.csv",
    "output_pdf": "relatorio_acoes_SOTA_melhorado.pdf",
    "temp_graph_folder": "graficos_temp",
    "yfinance_period": "6mo",
    "min_historical_data": 60,  # Dias mínimos de histórico para processar a ação
    "prediction_days": 15,
    "max_tickers_to_process": 10,
    "rsi_window": 14,
    "ma_window": 20,
    "ema_short_window": 12,
    "ema_long_window": 26,
}

# --- 2. FUNÇÕES DE PROCESSAMENTO DE DADOS E MODELAGEM ---

def fetch_and_prepare_data(ticker: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Busca dados do yfinance e calcula indicadores técnicos."""
    try:
        yf_ticker = f"{ticker.strip().upper()}.SA"
        acao = yf.Ticker(yf_ticker)
        
        info = acao.info
        hist = acao.history(period=CONFIG["yfinance_period"])

        if hist.empty or len(hist) < CONFIG["min_historical_data"]:
            print(f"Info: Dados insuficientes para {ticker}. Pulando.")
            return None, None

        # Limpeza e cálculo de indicadores
        hist = hist.reset_index()
        hist['Days'] = (hist['Date'] - hist['Date'].min()).dt.days
        hist['MM20'] = hist['Close'].rolling(window=CONFIG["ma_window"]).mean()
        
        # Cálculo de RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=CONFIG["rsi_window"]).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=CONFIG["rsi_window"]).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))

        hist['EMA12'] = hist['Close'].ewm(span=CONFIG["ema_short_window"], adjust=False).mean()
        hist['EMA26'] = hist['Close'].ewm(span=CONFIG["ema_long_window"], adjust=False).mean()
        hist['MACD'] = hist['EMA12'] - hist['EMA26']
        hist['UpperBand'] = hist['MM20'] + 2 * hist['Close'].rolling(window=CONFIG["ma_window"]).std()
        hist['LowerBand'] = hist['MM20'] - 2 * hist['Close'].rolling(window=CONFIG["ma_window"]).std()
        hist['VolumeMedia'] = hist['Volume'].rolling(window=CONFIG["ma_window"]).mean()
        hist.dropna(inplace=True)

        acao_info = {
            "ticker": ticker,
            "nome": info.get("longName") or info.get("shortName") or ticker,
            "setor": info.get("sector", "Setor desconhecido"),
            "preco_atual": info.get("regularMarketPrice") or info.get("previousClose", "N/D"),
        }

        return hist, acao_info
    
    except Exception as e:
        print(f"Erro ao buscar dados para {ticker}: {e}")
        return None, None

def train_and_predict_autoregressive(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Treina um modelo e faz uma previsão autorregressiva (passo a passo)."""
    features_cols = ['Days', 'MM20', 'RSI', 'MACD']
    target_col = 'Close'
    
    df_train = df.copy()
    
    X = df_train[features_cols]
    y = df_train[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingRegressor(n_estimators=200, random_state=42)
    model.fit(X_scaled, y)

    # Estimar a incerteza do modelo com base nos erros de treinamento (resíduos)
    predictions_on_train = model.predict(X_scaled)
    residuals = y - predictions_on_train
    prediction_std_error = residuals.std()

    # Previsão autorregressiva
    future_predictions = []
    last_date = df_train['Date'].max()
    temp_df = df_train.copy()

    for i in range(CONFIG["prediction_days"]):
        last_row = temp_df.iloc[-1]
        
        # Preparar features para a próxima previsão
        next_day_features_values = last_row[features_cols].values.reshape(1, -1)
        next_day_features_scaled = scaler.transform(next_day_features_values)

        # Fazer a previsão para o próximo dia
        next_price = model.predict(next_day_features_scaled)[0]
        future_predictions.append(next_price)

        # Criar uma nova linha com os dados previstos para recalcular indicadores
        next_date = last_date + pd.Timedelta(days=i + 1)
        new_row = pd.Series({
            'Date': next_date,
            'Close': next_price,
            'Days': last_row['Days'] + 1
        })
        
        # Adicionar nova linha e recalcular indicadores dinamicamente
        temp_df = pd.concat([temp_df, new_row.to_frame().T], ignore_index=True)
        temp_df['MM20'] = temp_df['Close'].rolling(window=CONFIG["ma_window"], min_periods=1).mean()
        delta = temp_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=CONFIG["rsi_window"], min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=CONFIG["rsi_window"], min_periods=1).mean()
        rs = gain / loss
        temp_df['RSI'] = 100 - (100 / (1 + rs))
        temp_df['EMA12'] = temp_df['Close'].ewm(span=CONFIG["ema_short_window"], adjust=False).mean()
        temp_df['EMA26'] = temp_df['Close'].ewm(span=CONFIG["ema_long_window"], adjust=False).mean()
        temp_df['MACD'] = temp_df['EMA12'] - temp_df['EMA26']
        temp_df.fillna(method='bfill', inplace=True) # Preencher NaNs iniciais
    
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=CONFIG["prediction_days"])
    predictions = pd.Series(future_predictions, index=future_dates)
    
    # Calcular bandas de confiança com base no erro padrão
    conf_upper = predictions + 1.96 * prediction_std_error
    conf_lower = predictions - 1.96 * prediction_std_error

    return predictions, conf_lower, conf_upper


# --- 3. FUNÇÕES DE GERAÇÃO DE GRÁFICOS E PDF ---

def create_plots(ticker: str, hist: pd.DataFrame, forecast: pd.Series, conf_lower: pd.Series, conf_upper: pd.Series) -> Dict[str, str]:
    """Cria e salva todos os gráficos para um determinado ticker."""
    paths = {}
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Histórico de Preços e Bandas de Bollinger
    paths['hist'] = os.path.join(CONFIG["temp_graph_folder"], f"{ticker}_historico.png")
    plt.figure(figsize=(10, 4))
    plt.plot(hist['Date'], hist['Close'], label='Preço de Fechamento', color='cyan')
    plt.plot(hist['Date'], hist['MM20'], label=f'Média Móvel {CONFIG["ma_window"]} dias', color='orange', linestyle='--')
    plt.fill_between(hist['Date'], hist['LowerBand'], hist['UpperBand'], color='gray', alpha=0.2, label='Bandas de Bollinger')
    plt.title(f"{ticker} - Histórico de Preços e Volatilidade")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths['hist'])
    plt.close()

    # Gráfico Combinado (Histórico + Previsão)
    paths['combined'] = os.path.join(CONFIG["temp_graph_folder"], f"{ticker}_combinado.png")
    plt.figure(figsize=(10, 4))
    plt.plot(hist['Date'], hist['Close'], label='Histórico de Preços', color='cyan')
    plt.plot(forecast.index, forecast, label=f'Previsão {CONFIG["prediction_days"]} dias', color='gold', linewidth=2)
    plt.fill_between(forecast.index, conf_lower, conf_upper, color='gold', alpha=0.3, label='Intervalo de Confiança (95%)')
    plt.title(f"{ticker} - Histórico e Previsão de Preços")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths['combined'])
    plt.close()

    # Gráficos de Indicadores (RSI, MACD, Volume)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    paths['indicators'] = os.path.join(CONFIG["temp_graph_folder"], f"{ticker}_indicadores.png")

    # RSI
    axes[0].plot(hist['Date'], hist['RSI'], color='purple', label='RSI')
    axes[0].axhline(70, color='red', linestyle='--', alpha=0.5)
    axes[0].axhline(30, color='green', linestyle='--', alpha=0.5)
    axes[0].set_title("Índice de Força Relativa (RSI)")
    axes[0].legend()

    # MACD
    axes[1].plot(hist['Date'], hist['MACD'], color='blue', label='MACD')
    axes[1].axhline(0, linestyle='--', color='black', alpha=0.5)
    axes[1].set_title("Convergência/Divergência de Médias Móveis (MACD)")
    axes[1].legend()

    # Volume
    axes[2].bar(hist['Date'], hist['Volume'], color='gray', alpha=0.5, label='Volume Diário')
    axes[2].plot(hist['Date'], hist['VolumeMedia'], color='brown', label=f'Média de Volume ({CONFIG["ma_window"]} dias)')
    axes[2].set_title(f"Volume de Negociação")
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(paths['indicators'])
    plt.close()
    
    return paths

def generate_pdf_elements(info: Dict, hist: pd.DataFrame, forecast: pd.Series, plot_paths: Dict) -> Tuple[List, Paragraph, List]:
    """Gera os elementos do ReportLab para um ticker."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    
    ticker, nome = info['ticker'], info['nome']
    preco_atual = hist['Close'].iloc[-1]
    preco_futuro = forecast.iloc[-1]
    
    tendencia_num = (preco_futuro - preco_atual) / preco_atual
    if tendencia_num > 0.02:
        tendencia_str = "ALTA"
        cor_tendencia = "green"
    elif tendencia_num < -0.02:
        tendencia_str = "BAIXA"
        cor_tendencia = "red"
    else:
        tendencia_str = "NEUTRA"
        cor_tendencia = "darkorange"
        
    # Elementos principais desta ação
    main_elements = [
        PageBreak(),
        Paragraph(f"Análise Detalhada: {ticker} - {nome}", styles['h1']),
        Spacer(1, 12),
        Paragraph(f"<b>Setor:</b> {info['setor']}", styles['Normal']),
        Paragraph(f"<b>Preço Atual (último fechamento):</b> R$ {preco_atual:.2f}", styles['Normal']),
        Spacer(1, 24),
        
        Paragraph("Visão Geral: Histórico e Previsão", styles['h2']),
        Image(plot_paths['combined'], width=480, height=240),
        Spacer(1, 12),
        
        Paragraph(
            f"O modelo de previsão Gradient Boosting, treinado com dados dos últimos 6 meses, projeta uma tendência "
            f"<b><font color='{cor_tendencia}'>{tendencia_str}</font></b> para os próximos {CONFIG['prediction_days']} dias. "
            f"A estimativa de preço para o final do período é de <b>R$ {preco_futuro:.2f}</b>. "
            f"O gráfico acima ilustra a projeção (em amarelo) e seu intervalo de confiança de 95%, que reflete a "
            f"incerteza inerente ao modelo.", styles['Justify']),
        Spacer(1, 24),
        
        Paragraph("Análise de Indicadores Técnicos", styles['h2']),
        Image(plot_paths['indicators'], width=480, height=384),
        Spacer(1, 12),
        
        Paragraph(
            f"Os indicadores técnicos mostram: RSI em <b>{hist['RSI'].iloc[-1]:.1f}</b>, sugerindo que o ativo "
            f"{'não está em sobrecompra nem sobrevenda' if 30 < hist['RSI'].iloc[-1] < 70 else ('está em território de sobrecompra' if hist['RSI'].iloc[-1] >= 70 else 'está em território de sobrevenda')}. "
            f"O MACD de <b>{hist['MACD'].iloc[-1]:.2f}</b> {'indica um momento de alta' if hist['MACD'].iloc[-1] > 0 else 'indica um momento de baixa'}.",
            styles['Justify']
        ),
    ]

    # Texto para o sumário
    summary_link = Paragraph(f"<a href='#{ticker}'>{ticker} - {nome}</a>", styles['Normal'])
    
    # Tabela para o apêndice
    dados_tab = [["Data", "Fechamento", "MM20", "RSI", "MACD"]] + [
        [row['Date'].strftime("%d/%m/%y"), f"R$ {row['Close']:.2f}", f"R$ {row['MM20']:.2f}",
         f"{row['RSI']:.1f}", f"{row['MACD']:.2f}"]
        for _, row in hist.tail(15).iterrows()  # Apenas os últimos 15 dias para a tabela
    ]
    tabela = Table(dados_tab, colWidths=[70, 80, 80, 70, 70])
    tabela.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    appendix_elements = [
        PageBreak(),
        Paragraph(f"Apêndice: Dados Históricos Recentes - {ticker}", styles['h2']),
        Spacer(1, 12),
        tabela
    ]
    
    return main_elements, summary_link, appendix_elements

def create_cover_page(canvas, doc):
    """Cria a capa do relatório."""
    canvas.saveState()
    canvas.setFillColor(colors.darkblue)
    canvas.rect(0, 0, A4[0], A4[1], fill=1)
    canvas.setFont("Helvetica-Bold", 26)
    canvas.setFillColor(colors.white)
    canvas.drawCentredString(A4[0] / 2, A4[1] - 200, "Relatório Inteligente de Ações")
    canvas.setFont("Helvetica", 16)
    canvas.drawCentredString(A4[0] / 2, A4[1] - 250, "Análise Preditiva e Técnica")
    canvas.setFont("Helvetica", 14)
    canvas.drawCentredString(A4[0] / 2, A4[1] - 300, f"Gerado em: {datetime.today().strftime('%d/%m/%Y')}")
    canvas.restoreState()

# --- 4. FUNÇÃO PRINCIPAL (ORQUESTRADOR) ---

def main():
    """Função principal que orquestra a criação do relatório."""
    os.makedirs(CONFIG["temp_graph_folder"], exist_ok=True)
    
    try:
        df_tickers = pd.read_csv(CONFIG["tickers_csv"])
        tickers = df_tickers['Ticker'].dropna().unique()

        story = []
        summary_elements = [Paragraph("Sumário", getSampleStyleSheet()['h1']), Spacer(1, 12)]
        appendix_elements = []

        for ticker in tickers[:CONFIG["max_tickers_to_process"]]:
            print(f"Processando {ticker}...")
            
            # 1. Obter e preparar dados
            hist_data, acao_info = fetch_and_prepare_data(ticker)
            if hist_data is None:
                continue
            
            # 2. Treinar modelo e fazer previsão
            forecast, conf_lower, conf_upper = train_and_predict_autoregressive(hist_data)
            
            # 3. Gerar gráficos
            plot_paths = create_plots(ticker, hist_data, forecast, conf_lower, conf_upper)
            
            # 4. Montar elementos do PDF
            main_elems, summary_link, appendix_elems = generate_pdf_elements(
                acao_info, hist_data, forecast, plot_paths
            )
            
            # Adicionar um marcador de âncora para o sumário
            story.append(Paragraph(f"<a name='{ticker}'/>", getSampleStyleSheet()['Normal']))
            story.extend(main_elems)
            summary_elements.append(summary_link)
            appendix_elements.extend(appendix_elems)
            
            print(f"{ticker} processado com sucesso.")

        # Montagem final do documento
        doc = SimpleDocTemplate(CONFIG["output_pdf"], pagesize=A4,
                                topMargin=72, bottomMargin=72, leftMargin=72, rightMargin=72)

        final_story = [Spacer(1, 150)] + summary_elements + story + appendix_elements
        
        doc.build(final_story, onFirstPage=create_cover_page)
        
        print(f"\nRelatório '{CONFIG['output_pdf']}' gerado com sucesso!")

    finally:
        # Limpa a pasta de gráficos temporários
        if os.path.exists(CONFIG["temp_graph_folder"]):
            shutil.rmtree(CONFIG["temp_graph_folder"])
            print(f"Pasta temporária '{CONFIG['temp_graph_folder']}' removida.")


if __name__ == "__main__":
    main()