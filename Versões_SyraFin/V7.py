# relatorio_syrafin_finalissimo_v2.py
# Versão final com a correção do AttributeError e do ImportError

import os
import shutil
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings

# --- BIBLIOTECAS ---
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
import matplotlib.pyplot as plt
import mplfinance as mpf
import qrcode

# --- REPORTLAB ---
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, Frame, Image, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)

# --- MACHINE LEARNING ---
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# --- 1. CONFIGURAÇÕES AVANÇADAS E LOGGING ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

CONFIG = {
    # Arquivos e diretórios
    "tickers_csv": "precos_acoes.csv",
    "output_pdf": "relatorio_syrafin_SOTA_final.pdf",
    "temp_graph_folder": "graficos_temp",
    "temp_qr_folder": "qr_codes_temp",
    # Parâmetros de dados
    "yfinance_period": "1y",
    "min_historical_data": 120,
    "filter_outliers_iqr": True,
    # Parâmetros de Indicadores Técnicos
    "rsi_window": 14,
    "ma_window": 20,
    "momentum_window": 10,
    "volatility_window": 20,
    # Parâmetros de Machine Learning
    "prediction_days": 15,
    "feature_window_size": 10,
    "walk_forward_splits": 5,
    "model_params": {
        "objective": "quantile",
        "metric": "quantile",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 20,
        "n_jobs": -1,
        "verbose": -1,
        "random_state": 42,
    },
    # Limites e Qualidade
    "max_tickers_to_process": 10,
    "min_r2_threshold": -0.3,
    # Visualização e Relatório
    "graph_style": "seaborn-v0_8-darkgrid",
    "qr_code_link": "https://www.b3.com.br/pt_br/educacao/dicas/para-investidores/como-analisar-uma-acao.htm"
}

# --- 2. FUNÇÕES DE DADOS E MODELAGEM ---

def buscar_e_preparar_dados(ticker: str) -> Tuple[pd.DataFrame | None, Dict | None]:
    """
    Busca dados, filtra outliers, calcula features técnicas avançadas
    e retorna informações da ação.
    """
    try:
        yf_ticker = f"{ticker.strip().upper()}.SA"
        acao = yf.Ticker(yf_ticker)
        
        info = acao.info
        hist = acao.history(period=CONFIG["yfinance_period"], auto_adjust=True)

        if hist.empty or len(hist) < CONFIG["min_historical_data"]:
            logging.warning(f"Dados históricos insuficientes para {ticker}. Pulando.")
            return None, None

        if CONFIG["filter_outliers_iqr"]:
            q1, q3 = hist['Close'].quantile(0.25), hist['Close'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            initial_rows = len(hist)
            hist = hist[(hist['Close'] >= lower_bound) & (hist['Close'] <= upper_bound)]
            if len(hist) < initial_rows:
                logging.info(f"Removidos {initial_rows - len(hist)} outliers de preço para {ticker}.")
        
        hist['Return'] = hist['Close'].pct_change()
        hist['Volatility'] = hist['Return'].rolling(window=CONFIG["volatility_window"]).std()
        hist['Momentum'] = hist['Close'].diff(CONFIG["momentum_window"])
        hist['Volume_change'] = hist['Volume'].pct_change()

        hist.replace([np.inf, -np.inf], np.nan, inplace=True)
        hist.fillna(0, inplace=True)

        hist['MM20'] = hist['Close'].rolling(window=CONFIG["ma_window"]).mean()
        std_dev = hist['Close'].rolling(window=CONFIG["ma_window"]).std()
        hist['UpperBand'] = hist['MM20'] + 2 * std_dev
        hist['LowerBand'] = hist['MM20'] - 2 * std_dev
        
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=CONFIG["rsi_window"]).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=CONFIG["rsi_window"]).mean()
        rs = gain / (loss + 1e-10)
        hist['RSI'] = 100 - (100 / (1 + rs))

        ema_short = hist['Close'].ewm(span=12, adjust=False).mean()
        ema_long = hist['Close'].ewm(span=26, adjust=False).mean()
        hist['MACD'] = ema_short - ema_long
        
        hist['VolumeMedia'] = hist['Volume'].rolling(window=CONFIG["ma_window"]).mean()
        
        hist.dropna(inplace=True)

        acao_info = {
            "ticker": ticker,
            "nome": info.get("longName") or info.get("shortName") or ticker,
            "setor": info.get("sector", "Setor desconhecido"),
        }
        
        hist['Date'] = hist.index

        return hist, acao_info
    
    except Exception as e:
        logging.error(f"Erro ao buscar/preparar dados para {ticker}: {e}")
        return None, None

def criar_janelas_deslizantes(data: pd.DataFrame, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Cria janelas de dados para o modelo usar o passado para prever o futuro. """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data.iloc[i:i+window_size].values.flatten())
        y.append(data.iloc[i+window_size].values[0])
    return np.array(X), np.array(y)


def treinar_avaliar_e_prever(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, Dict]:
    """
    Utiliza validação Walk-Forward, treina 3 modelos LGBM (mediana, quartil inferior e superior)
    e faz la previsão para o futuro.
    """
    features_cols = ['Close', 'Return', 'Volatility', 'Momentum', 'Volume_change', 'MM20', 'RSI', 'MACD']
    
    X, y = criar_janelas_deslizantes(df[features_cols], CONFIG["feature_window_size"])
    
    tscv = TimeSeriesSplit(n_splits=CONFIG["walk_forward_splits"])
    y_true_all, y_pred_all = [], []
    
    model_median = LGBMRegressor(alpha=0.5, **CONFIG["model_params"])
    model_lower = LGBMRegressor(alpha=0.05, **CONFIG["model_params"])
    model_upper = LGBMRegressor(alpha=0.95, **CONFIG["model_params"])

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model_median.fit(X_train_scaled, y_train)
        y_pred = model_median.predict(X_test_scaled)
        
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    metrics = {
        "r2": r2_score(y_true_all, y_pred_all),
        "mae": mean_absolute_error(y_true_all, y_pred_all),
        "rmse": np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    }
    
    scaler_full = StandardScaler().fit(X)
    X_scaled = scaler_full.transform(X)

    model_median.fit(X_scaled, y)
    model_lower.fit(X_scaled, y)
    model_upper.fit(X_scaled, y)
    
    future_predictions, future_lower, future_upper = [], [], []
    last_window_data = df[features_cols].tail(CONFIG["feature_window_size"]).copy()

    for _ in range(CONFIG["prediction_days"]):
        last_window_flat = last_window_data.values.flatten().reshape(1, -1)
        last_window_scaled = scaler_full.transform(last_window_flat)
        
        next_price = model_median.predict(last_window_scaled)[0]
        next_lower = model_lower.predict(last_window_scaled)[0]
        next_upper = model_upper.predict(last_window_scaled)[0]
        
        future_predictions.append(next_price)
        future_lower.append(next_lower)
        future_upper.append(next_upper)

        last_date = last_window_data.index[-1]
        next_date = last_date + pd.Timedelta(days=1)

        temp_df_extended = pd.concat([df[features_cols], pd.DataFrame(index=pd.date_range(end=next_date, periods=1))])
        temp_df_extended.loc[next_date, 'Close'] = next_price
        
        temp_df_extended['Volume'] = df['Volume']
        temp_df_extended['Volume'] = temp_df_extended['Volume'].ffill()

        temp_df_extended['Return'] = temp_df_extended['Close'].pct_change()
        temp_df_extended['Volatility'] = temp_df_extended['Return'].rolling(CONFIG["volatility_window"]).std()
        temp_df_extended['Momentum'] = temp_df_extended['Close'].diff(CONFIG["momentum_window"])
        temp_df_extended['Volume_change'] = temp_df_extended['Volume'].pct_change()
        temp_df_extended['MM20'] = temp_df_extended['Close'].rolling(CONFIG["ma_window"]).mean()
        delta = temp_df_extended['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=CONFIG["rsi_window"]).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=CONFIG["rsi_window"]).mean()
        rs = gain / (loss + 1e-10)
        temp_df_extended['RSI'] = 100 - (100 / (1 + rs))
        ema_short = temp_df_extended['Close'].ewm(span=12, adjust=False).mean()
        ema_long = temp_df_extended['Close'].ewm(span=26, adjust=False).mean()
        temp_df_extended['MACD'] = ema_short - ema_long

        new_row = temp_df_extended[features_cols].iloc[-1].ffill().fillna(0)
        
        last_window_data = pd.concat([last_window_data.iloc[1:], new_row.to_frame(name=next_date).T])

    future_dates = pd.date_range(df.index.max() + pd.Timedelta(days=1), periods=CONFIG["prediction_days"])
    predictions = pd.Series(future_predictions, index=future_dates)
    conf_lower = pd.Series(future_lower, index=future_dates)
    conf_upper = pd.Series(future_upper, index=future_dates)

    return predictions, conf_lower, conf_upper, metrics


def criar_todos_os_graficos(ticker: str, hist: pd.DataFrame, forecast: pd.Series, conf_lower: pd.Series, conf_upper: pd.Series) -> Dict[str, str]:
    """ Cria e salva TODOS os gráficos, incluindo candlestick, com tema profissional e de forma robusta. """
    paths = {}
    os.makedirs(CONFIG["temp_graph_folder"], exist_ok=True)
    plt.style.use(CONFIG["graph_style"])
    
    # 1. Gráfico Combinado (Histórico + Previsão) - Sem alterações aqui
    path_combined = os.path.join(CONFIG["temp_graph_folder"], f"{ticker}_combined.png")
    plt.figure(figsize=(10, 4.5))
    plt.plot(hist['Date'], hist['Close'], label='Histórico de Preços', color='#3498db', linewidth=1.5)
    plt.plot(forecast.index, forecast, label=f'Previsão {CONFIG["prediction_days"]} dias (Mediana)', color='#f1c40f', linewidth=2.5)
    plt.fill_between(forecast.index, conf_lower, conf_upper, color='#f1c40f', alpha=0.2, label='Intervalo de Confiança Quantílico')
    plt.title(f"{ticker} - Histórico e Previsão de Preços", fontsize=12)
    plt.ylabel("Preço (R$)"); plt.xlabel("Data"); plt.legend()
    plt.tight_layout(); plt.savefig(path_combined, dpi=150); plt.close()
    paths['combined'] = path_combined

    # 2. Grade de Gráficos Técnicos com Candlestick
    path_deep_dive = os.path.join(CONFIG["temp_graph_folder"], f"{ticker}_deep_dive.png")
    
    hist_plot = hist.set_index('Date').tail(90)
    if hist_plot.empty or len(hist_plot) < 2:
        logging.warning(f"Dados insuficientes para gerar o gráfico detalhado de {ticker}. Pulando.")
        paths['deep_dive'] = None
        return paths

    # *** CORREÇÃO FINAL: Usando os eixos diretamente do plt.subplots ***
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 2)
    fig.suptitle(f'Análise Técnica Detalhada - {ticker}', fontsize=16)

    # Definindo os eixos a partir do GridSpec
    ax1 = fig.add_subplot(gs[0:2, 0]) # Gráfico principal de candles
    ax2 = fig.add_subplot(gs[0, 1])   # RSI
    ax3 = fig.add_subplot(gs[1, 1])   # MACD
    ax4 = fig.add_subplot(gs[2, :])   # Volume na linha de baixo

    # a) Candlestick e Bandas de Bollinger
    mpf.plot(hist_plot, type='candle', style='charles', ax=ax1,
             volume=False, ylabel='Preço (R$)')
    ax1.plot(hist_plot.index, hist_plot['UpperBand'], color='cyan', linestyle='--', alpha=0.7, label='Banda Superior')
    ax1.plot(hist_plot.index, hist_plot['LowerBand'], color='cyan', linestyle='--', alpha=0.7, label='Banda Inferior')
    ax1.set_title("Preço e Bandas de Bollinger")
    ax1.legend()

    # b) Índice de Força Relativa (RSI)
    ax2.plot(hist_plot.index, hist_plot['RSI'], color='#8e44ad', label='RSI')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.7, label='Sobrecompra (70)')
    ax2.axhline(30, color='green', linestyle='--', alpha=0.7, label='Sobrevenda (30)')
    ax2.set_title("Índice de Força Relativa (RSI)"); ax2.set_ylim(0, 100); ax2.legend()
    
    # c) MACD
    ax3.plot(hist_plot.index, hist_plot['MACD'], label='Linha MACD', color='#27ae60')
    ax3.bar(hist_plot.index, hist_plot['MACD'], color=['#c0392b' if x < 0 else '#27ae60' for x in hist_plot['MACD']], alpha=0.4)
    ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax3.set_title("MACD")
    
    # d) Volume
    ax4.bar(hist_plot.index, hist_plot['Volume'], color='grey', alpha=0.6, label='Volume Diário')
    ax4.plot(hist_plot.index, hist_plot['VolumeMedia'], color='#c0392b', label=f'Média de Volume ({CONFIG["ma_window"]}d)')
    ax4.set_title("Volume de Negociação"); ax4.legend()
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.96]) # Ajusta o layout para caber o suptitle
    plt.savefig(path_deep_dive, dpi=150); plt.close(fig)
    paths['deep_dive'] = path_deep_dive
    
    return paths

# --- 3. CLASSE DE GERAÇÃO DO PDF ---
class GeradorRelatorioPDF:
    def __init__(self, filename: str):
        self.filename = filename
        self.doc = BaseDocTemplate(filename, pagesize=A4,
                                   leftMargin=2*28.35, rightMargin=2*28.35,
                                   topMargin=2*28.35, bottomMargin=2*28.35)
        self._setup_styles()
        self._setup_templates()
        self.story = []
        os.makedirs(CONFIG["temp_qr_folder"], exist_ok=True)

    def _setup_styles(self):
        styles = getSampleStyleSheet()
        self.styles = {
            'Title': ParagraphStyle(name='Title', fontName='Helvetica-Bold', fontSize=26, textColor='#16a085', alignment=TA_CENTER),
            'SubTitle': ParagraphStyle(name='SubTitle', fontName='Helvetica', fontSize=14, textColor='#2c3e50', alignment=TA_CENTER, spaceBefore=10),
            'h1': ParagraphStyle(name='h1', fontName='Helvetica-Bold', fontSize=20, textColor='#2c3e50', spaceBefore=20, spaceAfter=10, keepWithNext=1),
            'h2': ParagraphStyle(name='h2', fontName='Helvetica-Bold', fontSize=14, textColor='#2c3e50', spaceBefore=12, spaceAfter=6, keepWithNext=1),
            'Body': ParagraphStyle(name='Body', fontName='Helvetica', fontSize=10, textColor='#34495e', alignment=TA_JUSTIFY, leading=14),
            'Footer': ParagraphStyle(name='Footer', fontName='Helvetica', fontSize=8, textColor=colors.grey, alignment=TA_CENTER),
            'Header': ParagraphStyle(name='Header', fontName='Helvetica', fontSize=9, textColor=colors.grey, alignment=TA_RIGHT),
            'Disclaimer': ParagraphStyle(name='Disclaimer', fontName='Helvetica-Oblique', fontSize=9, textColor=colors.darkgrey, alignment=TA_JUSTIFY, leading=12),
        }
        
    def _header_footer(self, canvas, doc):
        canvas.saveState()
        p_header = Paragraph("SYRAFIN - Relatório Preditivo de Ações (SOTA Edition)", self.styles['Header'])
        w_h, h_h = p_header.wrap(doc.width, doc.topMargin)
        p_header.drawOn(canvas, doc.leftMargin, A4[1] - doc.topMargin + 10)
        p_page = Paragraph(f"Página {doc.page}", self.styles['Footer'])
        w_p, h_p = p_page.wrap(doc.width, doc.bottomMargin)
        p_page.drawOn(canvas, doc.leftMargin, doc.bottomMargin - 20)
        canvas.restoreState()

    def _setup_templates(self):
        frame = Frame(self.doc.leftMargin, self.doc.bottomMargin, self.doc.width, self.doc.height, id='normal')
        self.doc.addPageTemplates([
            PageTemplate(id='Capa', frames=[frame]),
            PageTemplate(id='Principal', frames=[frame], onPage=self._header_footer),
        ])

    def _get_tendencia_info(self, preco_atual, preco_futuro) -> Tuple[str, colors.Color]:
        variacao = (preco_futuro - preco_atual) / preco_atual
        if variacao > 0.02: return "ALTA", colors.green
        if variacao < -0.02: return "BAIXA", colors.red
        return "NEUTRA", colors.darkorange

    def _get_r2_color(self, r2: float) -> colors.Color:
        if r2 > 0.7: return colors.green
        if r2 > 0.4: return colors.darkorange
        return colors.red

    def construir_relatorio(self, dados_analises: List[Dict]):
        self._criar_pagina_capa()
        self._criar_sumario_executivo(dados_analises)

        for dados in dados_analises:
            if dados['plots']['deep_dive'] is not None:
                self._adicionar_analise_ticker(dados)
                self.story.append(PageBreak())

        self._criar_pagina_aviso_legal()
        self.doc.build(self.story)
        
    def _criar_pagina_capa(self):
        self.story.append(Spacer(1, 150))
        self.story.append(Paragraph("SYRAFIN SOTA", self.styles['Title']))
        self.story.append(Paragraph("Relatório Preditivo de Ações State-of-the-Art", self.styles['SubTitle']))
        self.story.append(Spacer(1, 12))
        self.story.append(Paragraph("Análise Técnica e Preditiva com Machine Learning Avançado", self.styles['SubTitle']))
        self.story.append(Spacer(1, 250))
        data_geracao = datetime.today().strftime('%d de %B de %Y')
        self.story.append(Paragraph(f"Gerado em: {data_geracao}", self.styles['SubTitle']))
        self.story.append(NextPageTemplate('Principal'))
        self.story.append(PageBreak())

    def _criar_sumario_executivo(self, dados_analises: List[Dict]):
        self.story.append(Paragraph("Sumário Executivo", self.styles['h1']))
        self.story.append(Paragraph("A tabela resume as previsões para os próximos 15 dias, incluindo métricas de confiança do modelo.", self.styles['Body']))
        self.story.append(Spacer(1, 20))
        
        dados_tabela = [["Ticker", "Preço Atual", "Previsão", "Variação (%)", "Tendência", "Confiança (R²)"]]
        for data in dados_analises:
            preco_atual = data['hist']['Close'].iloc[-1]
            preco_futuro = data['forecast'].iloc[-1]
            variacao = (preco_futuro - preco_atual) / preco_atual * 100
            tendencia, cor_tend = self._get_tendencia_info(preco_atual, preco_futuro)
            r2 = data['metrics']['r2']
            cor_r2 = self._get_r2_color(r2)
            
            dados_tabela.append([
                data['info']['ticker'], f"R$ {preco_atual:.2f}", f"R$ {preco_futuro:.2f}",
                Paragraph(f"{variacao:.2f}%", ParagraphStyle(name='var', textColor=cor_tend)),
                Paragraph(tendencia, ParagraphStyle(name='tend', textColor=cor_tend)),
                Paragraph(f"{r2:.2f}", ParagraphStyle(name='r2', textColor=cor_r2)),
            ])
        
        tabela_sumario = Table(dados_tabela, colWidths=[50, 80, 80, 80, 70, 80])
        tabela_sumario.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), '#16a085'), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')), ('GRID', (0, 0), (-1, -1), 1, '#2c3e50'),
        ]))
        self.story.append(tabela_sumario)
        self.story.append(PageBreak())

    def _adicionar_analise_ticker(self, data: Dict):
        info, hist, forecast, plots, metrics = data['info'], data['hist'], data['forecast'], data['plots'], data['metrics']
        
        self.story.append(Paragraph(f"Análise Completa: {info['nome']} ({info['ticker']})", self.styles['h1']))
        self.story.append(Paragraph(f"<b>Setor:</b> {info['setor']}", self.styles['Body']))
        self.story.append(Spacer(1, 12))

        preco_atual = hist['Close'].iloc[-1]
        preco_futuro = forecast.iloc[-1]
        tendencia_str, cor_tendencia = self._get_tendencia_info(preco_atual, preco_futuro)
        r2, mae, rmse = metrics['r2'], metrics['mae'], metrics['rmse']
        cor_r2 = self._get_r2_color(r2)
        
        dados_metricas = [
            [Paragraph('<b>Métrica Chave</b>', self.styles['Body']), Paragraph('<b>Valor</b>', self.styles['Body'])],
            ['Preço Atual (último fech.)', f"R$ {preco_atual:.2f}"],
            [f'Previsão ({CONFIG["prediction_days"]}d)', f"R$ {preco_futuro:.2f}"],
            ['Tendência Prevista', Paragraph(f"<b>{tendencia_str}</b>", ParagraphStyle(name='Tendencia', textColor=cor_tendencia))],
            ['RSI Atual', f"{hist['RSI'].iloc[-1]:.1f}"],
            ['Confiança do Modelo (R²)', Paragraph(f"<b>{r2:.2f}</b>", ParagraphStyle(name='R2', textColor=cor_r2))],
            ['Erro Médio Absoluto (MAE)', f"R$ {mae:.2f}"],
            ['Raiz do Erro Quadrático (RMSE)', f"R$ {rmse:.2f}"],
        ]
        tabela_metricas = Table(dados_metricas, colWidths=[180, '*'])
        tabela_metricas.setStyle(TableStyle([
            ('GRID', (0,0),(-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0),(-1,0), colors.lightgrey),
            ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('LEFTPADDING', (0,0),(-1,-1), 6)
        ]))
        self.story.append(tabela_metricas)
        self.story.append(Spacer(1, 12))
        
        self.story.append(Paragraph("Visão Geral Preditiva", self.styles['h2']))
        self.story.append(Image(plots['combined'], width=480, height=216))
        self.story.append(Spacer(1, 20))
        self.story.append(Paragraph("Análise Técnica Detalhada", self.styles['h2']))
        self.story.append(Image(plots['deep_dive'], width=500, height=333))

    def _gerar_qr_code(self, link: str, path: str):
        img = qrcode.make(link)
        img.save(path)

    def _criar_pagina_aviso_legal(self):
        self.story.append(Paragraph("Aviso Legal e Recursos", self.styles['h1']))
        texto_aviso = """
        Este relatório foi gerado automaticamente por meio do sistema SYRAFIN e destina-se exclusivamente a fins informativos
        e educacionais. As análises, previsões e indicadores aqui apresentados são baseados em dados históricos e modelos
        estatísticos, que possuem limitações inerentes e não garantem resultados futuros.
        <br/><br/>
        As informações contidas neste documento <b>NÃO constituem uma recomendação de compra, venda ou manutenção de
        qualquer ativo financeiro</b>. O investimento no mercado de ações envolve riscos, incluindo a possibilidade de
        perda do capital investido.
        <br/><br/>
        O leitor é o único responsável por suas decisões de investimento e é fortemente encorajado a buscar
        aconselhamento de um profissional financeiro qualificado antes de tomar qualquer decisão. Os autores e
        desenvolvedores do sistema SYRAFIN isentam-se de qualquer responsabilidade por perdas ou danos de qualquer
        natureza que possam surgir do uso das informações contidas neste relatório.
        """
        self.story.append(Paragraph(texto_aviso, self.styles['Disclaimer']))
        self.story.append(Spacer(1, 20))
        
        self.story.append(Paragraph("Aprenda a Interpretar", self.styles['h2']))
        self.story.append(Paragraph("Use a câmera do seu celular para escanear o QR Code abaixo e acessar um guia sobre como interpretar os indicadores técnicos apresentados neste relatório.", self.styles['Body']))
        qr_path = os.path.join(CONFIG["temp_qr_folder"], "info_qr_code.png")
        self._gerar_qr_code(CONFIG["qr_code_link"], qr_path)
        self.story.append(Spacer(1, 10))
        self.story.append(Image(qr_path, width=100, height=100))

# --- 4. FUNÇÃO PRINCIPAL (ORQUESTRADOR) ---

def main():
    # Suprime avisos benignos para um log mais limpo
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    try:
        df_tickers = pd.read_csv(CONFIG["tickers_csv"])
        tickers = df_tickers['Ticker'].dropna().unique()
    except FileNotFoundError:
        logging.error(f"Arquivo de tickers '{CONFIG['tickers_csv']}' não encontrado.")
        return

    dados_completos_analises = []
    
    process_iterator = tqdm(tickers[:CONFIG["max_tickers_to_process"]], desc="Processando tickers", unit="ticker")

    for ticker in process_iterator:
        process_iterator.set_postfix_str(ticker)
        
        hist_data, acao_info = buscar_e_preparar_dados(ticker)
        if hist_data is None:
            continue
        
        try:
            forecast, conf_lower, conf_upper, metrics = treinar_avaliar_e_prever(hist_data)
            
            if metrics["r2"] < CONFIG["min_r2_threshold"]:
                logging.warning(f"Modelo para {ticker} com R² muito baixo ({metrics['r2']:.2f}). Descartando do relatório.")
                continue

            plot_paths = criar_todos_os_graficos(ticker, hist_data, forecast, conf_lower, conf_upper)
            
            if plot_paths.get('deep_dive') is None:
                continue

            dados_completos_analises.append({
                "info": acao_info, "hist": hist_data, "forecast": forecast,
                "plots": plot_paths, "metrics": metrics,
            })
            logging.info(f"{ticker} processado com sucesso (R²: {metrics['r2']:.2f}, MAE: {metrics['mae']:.2f}).")
        except Exception as e:
            logging.error(f"Falha no pipeline de ML para {ticker}: {e}", exc_info=True)

    if not dados_completos_analises:
        logging.warning("Nenhum ticker foi processado com sucesso. O relatório não será gerado.")
    else:
        try:
            logging.info("Iniciando a geração do relatório PDF SOTA...")
            gerador_pdf = GeradorRelatorioPDF(CONFIG["output_pdf"])
            gerador_pdf.construir_relatorio(dados_completos_analises)
            logging.info(f"Relatório '{CONFIG['output_pdf']}' gerado com sucesso!")
        except Exception as e:
            logging.error(f"Ocorreu um erro ao gerar o PDF: {e}")

if __name__ == "__main__":
    main()
    for folder in [CONFIG["temp_graph_folder"], CONFIG["temp_qr_folder"]]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                logging.info(f"Pasta temporária '{folder}' removida.")