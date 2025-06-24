# relatorio_syrafin_completo.py

import os
import shutil
from datetime import datetime
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, Frame, Image, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# --- 1. CONFIGURAÇÕES GERAIS ---
CONFIG = {
    "tickers_csv": "precos_acoes.csv",
    "output_pdf": "relatorio_syrafin_completo.pdf",
    "temp_graph_folder": "graficos_temp",
    "yfinance_period": "6mo",
    "min_historical_data": 60,
    "prediction_days": 15,
    "max_tickers_to_process": 10, #Quantidades de tokens para analisar 
    "rsi_window": 14,
    "ma_window": 20,
    "ema_short_window": 12,
    "ema_long_window": 26,
}

# --- 2. FUNÇÕES DE DADOS E MODELAGEM (IDÊNTICAS À VERSÃO ANTERIOR) ---

def fetch_and_prepare_data(ticker: str) -> Tuple[pd.DataFrame | None, Dict | None]:
    """Busca dados do yfinance e calcula indicadores técnicos."""
    try:
        yf_ticker = f"{ticker.strip().upper()}.SA"
        acao = yf.Ticker(yf_ticker)
        
        info = acao.info
        hist = acao.history(period=CONFIG["yfinance_period"])

        if hist.empty or len(hist) < CONFIG["min_historical_data"]:
            print(f"Info: Dados insuficientes para {ticker}. Pulando.")
            return None, None

        hist = hist.reset_index()
        hist['Days'] = (hist['Date'] - hist['Date'].min()).dt.days
        hist['MM20'] = hist['Close'].rolling(window=CONFIG["ma_window"]).mean()
        
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
        }
        return hist, acao_info
    
    except Exception as e:
        print(f"Erro ao buscar dados para {ticker}: {e}")
        return None, None

def train_and_predict_autoregressive(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Treina um modelo e faz uma previsão autorregressiva."""
    features_cols = ['Days', 'MM20', 'RSI', 'MACD']
    target_col = 'Close'
    
    X = df[features_cols]
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingRegressor(n_estimators=200, random_state=42)
    model.fit(X_scaled, y)

    residuals = y - model.predict(X_scaled)
    prediction_std_error = residuals.std()

    future_predictions = []
    temp_df = df.copy()

    for i in range(CONFIG["prediction_days"]):
        last_row = temp_df.iloc[-1]
        next_day_features_df = pd.DataFrame([last_row[features_cols].values], columns=features_cols)
        next_day_features_scaled = scaler.transform(next_day_features_df)

        next_price = model.predict(next_day_features_scaled)[0]
        future_predictions.append(next_price)

        new_row = pd.Series({
            'Date': temp_df['Date'].max() + pd.Timedelta(days=1),
            'Close': next_price, 'Days': last_row['Days'] + 1,
        })
        
        temp_df = pd.concat([temp_df, new_row.to_frame().T], ignore_index=True)
        # Recalcular indicadores dinamicamente
        temp_df['MM20'] = temp_df['Close'].rolling(window=CONFIG["ma_window"], min_periods=1).mean()
        delta = temp_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=CONFIG["rsi_window"], min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=CONFIG["rsi_window"], min_periods=1).mean()
        rs = gain / loss
        temp_df['RSI'] = 100 - (100 / (1 + rs))
        temp_df['EMA12'] = temp_df['Close'].ewm(span=CONFIG["ema_short_window"], adjust=False).mean()
        temp_df['EMA26'] = temp_df['Close'].ewm(span=CONFIG["ema_long_window"], adjust=False).mean()
        temp_df['MACD'] = temp_df['EMA12'] - temp_df['EMA26']
        temp_df.bfill(inplace=True)
    
    future_dates = pd.date_range(df['Date'].max() + pd.Timedelta(days=1), periods=CONFIG["prediction_days"])
    predictions = pd.Series(future_predictions, index=future_dates)
    
    conf_upper = predictions + 1.96 * prediction_std_error
    conf_lower = predictions - 1.96 * prediction_std_error

    return predictions, conf_lower, conf_upper


def create_all_plots(ticker: str, hist: pd.DataFrame, forecast: pd.Series, conf_lower: pd.Series, conf_upper: pd.Series) -> Dict[str, str]:
    """Cria e salva TODOS os gráficos necessários para o relatório."""
    paths = {}
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # --- 1. Gráfico Combinado (Histórico + Previsão) ---
    paths['combined'] = os.path.join(CONFIG["temp_graph_folder"], f"{ticker}_combined.png")
    plt.figure(figsize=(10, 4.5))
    plt.plot(hist['Date'], hist['Close'], label='Histórico de Preços', color='#3498db', linewidth=1.5)
    plt.plot(forecast.index, forecast, label=f'Previsão {CONFIG["prediction_days"]} dias', color='#f1c40f', linewidth=2.5)
    plt.fill_between(forecast.index, conf_lower, conf_upper, color='#f1c40f', alpha=0.2, label='Intervalo de Confiança')
    plt.title(f"{ticker} - Histórico e Previsão de Preços", fontsize=12)
    plt.ylabel("Preço (R$)"); plt.xlabel("Data"); plt.legend()
    plt.tight_layout(); plt.savefig(paths['combined'], dpi=150); plt.close()

    # --- 2. Gráfico Apenas da Previsão (NOVO) ---
    paths['forecast_only'] = os.path.join(CONFIG["temp_graph_folder"], f"{ticker}_forecast_only.png")
    plt.figure(figsize=(10, 4.5))
    plt.plot(forecast.index, forecast, label='Previsão de Preço', color='#e67e22', marker='o', linestyle='--')
    plt.fill_between(forecast.index, conf_lower, conf_upper, color='#e67e22', alpha=0.2, label='Intervalo de Confiança')
    plt.title(f"{ticker} - Foco na Previsão para os Próximos {CONFIG['prediction_days']} Dias", fontsize=12)
    plt.ylabel("Preço Estimado (R$)"); plt.xlabel("Data da Previsão"); plt.legend()
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout(); plt.savefig(paths['forecast_only'], dpi=150); plt.close()

    # --- 3. Grade de Gráficos Técnicos (REINTRODUZIDO E MELHORADO) ---
    paths['deep_dive'] = os.path.join(CONFIG["temp_graph_folder"], f"{ticker}_deep_dive.png")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Análise Técnica Detalhada - {ticker}', fontsize=16, y=0.98)

    # a) Preço e Bandas de Bollinger
    axs[0, 0].plot(hist['Date'], hist['Close'], label='Fechamento', color='#3498db')
    axs[0, 0].plot(hist['Date'], hist['MM20'], label='MM20', color='orange', linestyle='--')
    axs[0, 0].fill_between(hist['Date'], hist['LowerBand'], hist['UpperBand'], color='gray', alpha=0.2)
    axs[0, 0].set_title("Preço e Bandas de Bollinger"); axs[0, 0].legend()

    # b) Índice de Força Relativa (RSI)
    axs[0, 1].plot(hist['Date'], hist['RSI'], color='#8e44ad', label='RSI')
    axs[0, 1].axhline(70, color='red', linestyle='--', alpha=0.5); axs[0, 1].axhline(30, color='green', linestyle='--', alpha=0.5)
    axs[0, 1].set_title("Índice de Força Relativa (RSI)"); axs[0, 1].set_ylim(0, 100)

    # c) MACD
    axs[1, 0].plot(hist['Date'], hist['MACD'], label='MACD', color='#27ae60')
    axs[1, 0].bar(hist['Date'], hist['MACD'], color='#27ae60', alpha=0.3)
    axs[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axs[1, 0].set_title("MACD"); axs[1, 0].legend()

    # d) Volume
    axs[1, 1].bar(hist['Date'], hist['Volume'], color='grey', alpha=0.4, label='Volume Diário')
    axs[1, 1].plot(hist['Date'], hist['VolumeMedia'], color='#c0392b', label='Média de Volume (20d)')
    axs[1, 1].set_title("Volume de Negociação"); axs[1, 1].legend()
    axs[1, 1].tick_params(axis='y', labelleft=False) # Remove y-axis labels to avoid clutter
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust rect to make space for suptitle
    plt.savefig(paths['deep_dive'], dpi=150)
    plt.close()
    
    return paths

# --- 3. CLASSE DE GERAÇÃO DO PDF ---

class ReportPDF:
    """Encapsula a criação do relatório SYRAFIN."""

    def __init__(self, filename: str):
        self.filename = filename
        self.doc = BaseDocTemplate(filename, pagesize=A4,
                                   leftMargin=2*28.35, rightMargin=2*28.35,
                                   topMargin=2*28.35, bottomMargin=2*28.35)
        self._setup_templates()
        self._setup_styles()
        self.story = []
        self.appendix_elements = []

    def _setup_styles(self):
        """Define os estilos de parágrafo."""
        styles = getSampleStyleSheet()
        self.styles = {
            'Title': ParagraphStyle(name='Title', fontName='Helvetica-Bold', fontSize=26, textColor='#16a085', alignment=TA_CENTER),
            'SubTitle': ParagraphStyle(name='SubTitle', fontName='Helvetica', fontSize=14, textColor='#2c3e50', alignment=TA_CENTER, spaceBefore=10),
            'h1': ParagraphStyle(name='h1', fontName='Helvetica-Bold', fontSize=20, textColor='#2c3e50', spaceBefore=12, spaceAfter=10, keepWithNext=1),
            'h2': ParagraphStyle(name='h2', fontName='Helvetica-Bold', fontSize=14, textColor='#2c3e50', spaceBefore=12, spaceAfter=6, keepWithNext=1),
            'Body': ParagraphStyle(name='Body', fontName='Helvetica', fontSize=10, textColor='#34495e', alignment=TA_JUSTIFY, leading=14),
            'Footer': ParagraphStyle(name='Footer', fontName='Helvetica', fontSize=8, textColor=colors.grey, alignment=TA_CENTER),
            'Header': ParagraphStyle(name='Header', fontName='Helvetica', fontSize=9, textColor=colors.grey, alignment=TA_RIGHT),
        }
    
    def _header_footer(self, canvas, doc):
        """Define o cabeçalho e rodapé."""
        canvas.saveState()
        # Header
        p_header = Paragraph("SYRAFIN - Relatório Quinzenal", self.styles['Header'])
        w_h, h_h = p_header.wrap(doc.width, doc.topMargin)
        p_header.drawOn(canvas, doc.leftMargin, A4[1] - doc.topMargin + 10)
        canvas.line(doc.leftMargin, A4[1] - doc.topMargin + 5, A4[0] - doc.rightMargin, A4[1] - doc.topMargin + 5)
        
        # Footer
        p_page = Paragraph(f"Página {doc.page}", self.styles['Footer'])
        w_p, h_p = p_page.wrap(doc.width, doc.bottomMargin)
        p_page.drawOn(canvas, 0, h_p, TA_CENTER)
        canvas.restoreState()

    def _setup_templates(self):
        frame = Frame(self.doc.leftMargin, self.doc.bottomMargin, self.doc.width, self.doc.height, id='normal')
        self.doc.addPageTemplates([
            PageTemplate(id='Cover', frames=[frame]),
            PageTemplate(id='Main', frames=[frame], onPage=self._header_footer),
        ])

    def _get_trend_info(self, current_price, future_price) -> Tuple[str, str]:
        change_pct = (future_price - current_price) / current_price
        if change_pct > 0.02: return "ALTA", "green"
        if change_pct < -0.02: return "BAIXA", "red"
        return "NEUTRA", "darkorange"

    def build_report(self, analyses_data: List[Dict]):
        """Constrói o PDF completo a partir dos dados de análise."""
        # 1. Capa
        self.story.append(Spacer(1, 150))
        self.story.append(Paragraph("SYRAFIN", self.styles['Title']))
        self.story.append(Paragraph("", self.styles['Title']))
        self.story.append(Paragraph("Relatório Quinzenal", self.styles['Title']))
        self.story.append(Spacer(1, 12))
        self.story.append(Paragraph("Análise Preditiva e Técnica de Ações da B3", self.styles['SubTitle']))
        self.story.append(Spacer(1, 250))
        self.story.append(Paragraph(f"Gerado em: {datetime.today().strftime('%d de %B de %Y')}", self.styles['SubTitle']))
        self.story.append(NextPageTemplate('Main'))
        self.story.append(PageBreak())

        # 2. Sumário Executivo
        summary_table_data = [["Ticker", "Nome da Empresa", "Tendência Prevista"]]
        for data in analyses_data:
            tendencia, _ = self._get_trend_info(data['hist']['Close'].iloc[-1], data['forecast'].iloc[-1])
            summary_table_data.append((data['info']['ticker'], data['info']['nome'], tendencia))
        
        self.story.append(Paragraph("Sumário Executivo", self.styles['h1']))
        self.story.append(Paragraph("A tabela abaixo resume as análises realizadas, apresentando a tendência preditiva para cada ativo.", self.styles['Body']))
        self.story.append(Spacer(1, 20))
        summary_table = Table(summary_table_data, colWidths=[60, 280, 100])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), '#16a085'), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')), ('GRID', (0, 0), (-1, -1), 1, '#2c3e50'),
        ]))
        self.story.append(summary_table)
        self.story.append(PageBreak())

        # 3. Análises Individuais
        for data in analyses_data:
            self._add_ticker_analysis(data)
            self._add_appendix_table_data(data['info']['ticker'], data['hist'])

        # 4. Apêndice
        if self.appendix_elements:
            self.story.append(PageBreak())
            self.story.append(Paragraph("Apêndice: Dados Históricos Detalhados", self.styles['h1']))
            self.story.extend(self.appendix_elements)

        # 5. Geração do arquivo
        self.doc.build(self.story)
        
    def _add_ticker_analysis(self, data: Dict):
        """Adiciona as várias páginas de análise para um único ticker."""
        info, hist, forecast, plots = data['info'], data['hist'], data['forecast'], data['plots']
        
        # --- PÁGINA 1: RESUMO E VISÃO GERAL ---
        self.story.append(Paragraph(f"Análise Completa: {info['ticker']}", self.styles['h1']))
        
        preco_atual = hist['Close'].iloc[-1]
        preco_futuro = forecast.iloc[-1]
        tendencia_str, _ = self._get_trend_info(preco_atual, preco_futuro)
        metrics_data = [
            [Paragraph('<b>Métrica Chave</b>', self.styles['Body']), Paragraph('<b>Valor</b>', self.styles['Body'])],
            [Paragraph('Preço Atual', self.styles['Body']), f"R$ {preco_atual:.2f}"],
            [Paragraph(f'Previsão ({CONFIG["prediction_days"]}d)', self.styles['Body']), f"R$ {preco_futuro:.2f}"],
            [Paragraph('Tendência', self.styles['Body']), Paragraph(f"<b>{tendencia_str}</b>", self.styles['Body'])],
            [Paragraph('RSI Atual', self.styles['Body']), f"{hist['RSI'].iloc[-1]:.1f}"],
        ]
        metrics_table = Table(metrics_data, colWidths=[150, '*'])
        metrics_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0),(-1,0), colors.lightgrey)]))
        self.story.append(metrics_table)
        self.story.append(Spacer(1, 12))
        
        self.story.append(Paragraph("Visão Geral Preditiva", self.styles['h2']))
        self.story.append(Image(plots['combined'], width=480, height=216))
        self.story.append(PageBreak())

        # --- PÁGINA 2: ANÁLISE TÉCNICA DETALHADA ---
        self.story.append(Paragraph("Análise Técnica Detalhada", self.styles['h2']))
        self.story.append(Image(plots['deep_dive'], width=500, height=333))
        self.story.append(PageBreak())

        # --- PÁGINA 3: FOCO NA PREVISÃO ---
        self.story.append(Paragraph("Foco na Previsão", self.styles['h2']))
        self.story.append(Image(plots['forecast_only'], width=480, height=216))
        self.story.append(PageBreak())

    def _add_appendix_table_data(self, ticker: str, hist: pd.DataFrame):
        """Prepara a tabela de dados históricos para o apêndice."""
        self.appendix_elements.append(Paragraph(f"Tabela de Dados - {ticker}", self.styles['h2']))
        dados_tab = [["Data", "Fech.", "MM20", "RSI", "MACD", "Volume"]] + [
            [row['Date'].strftime("%d/%m"), f"{row['Close']:.2f}", f"{row['MM20']:.2f}",
             f"{row['RSI']:.1f}", f"{row['MACD']:.2f}", f"{row['Volume'] / 1e6:.2f}M"]
            for _, row in hist.tail(30).iterrows() # Últimos 30 dias para não ficar gigante
        ]
        tabela = Table(dados_tab, colWidths=[50, 60, 60, 50, 60, 80])
        tabela.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkslategray), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        self.appendix_elements.append(tabela)
        self.appendix_elements.append(Spacer(1, 24))


# --- 4. FUNÇÃO PRINCIPAL (ORQUESTRADOR) ---

def main():
    """Função principal que orquestra a criação do relatório."""
    os.makedirs(CONFIG["temp_graph_folder"], exist_ok=True)
    
    try:
        df_tickers = pd.read_csv(CONFIG["tickers_csv"])
        tickers = df_tickers['Ticker'].dropna().unique()

        analyses_data = []
        for ticker in tickers[:CONFIG["max_tickers_to_process"]]:
            print(f"Processando {ticker}...")
            
            hist_data, acao_info = fetch_and_prepare_data(ticker)
            if hist_data is None: continue
            
            forecast, conf_lower, conf_upper = train_and_predict_autoregressive(hist_data)
            plot_paths = create_all_plots(ticker, hist_data, forecast, conf_lower, conf_upper)
            
            analyses_data.append({
                "info": acao_info, "hist": hist_data, "forecast": forecast,
                "plots": plot_paths,
            })
            print(f"{ticker} processado com sucesso.")

        if not analyses_data:
            print("Nenhum ticker foi processado com sucesso. Relatório não será gerado.")
            return

        pdf_report = ReportPDF(CONFIG["output_pdf"])
        pdf_report.build_report(analyses_data)
        
        print(f"\nRelatório '{CONFIG['output_pdf']}' gerado com sucesso!")

    finally:
        if os.path.exists(CONFIG["temp_graph_folder"]):
            shutil.rmtree(CONFIG["temp_graph_folder"])
            print(f"Pasta temporária '{CONFIG['temp_graph_folder']}' removida.")

if __name__ == "__main__":
    main()