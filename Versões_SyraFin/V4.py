# relatorio_syrafin_melhorado.py

import os
import shutil
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, Frame, Image, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# --- 1. CONFIGURAÇÕES E LOGGING (MELHORADO) ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

CONFIG = {
    # Arquivos e diretórios
    "tickers_csv": "precos_acoes.csv",
    "output_pdf": "relatorio_syrafin_melhorado.pdf",
    "temp_graph_folder": "graficos_temp",
    # Parâmetros de dados
    "yfinance_period": "6mo",
    "min_historical_data": 60,
    # Parâmetros de Indicadores Técnicos
    "rsi_window": 14,
    "ma_window": 20,
    "ema_short_window": 12,
    "ema_long_window": 26,
    # Parâmetros de Machine Learning
    "prediction_days": 15,
    "test_size_for_eval": 0.2,
    "model_params": {
        "n_estimators": 200,
        "random_state": 42,
        "learning_rate": 0.05,
        "max_depth": 3
    },
    # Limites
    "max_tickers_to_process": 10,
}

# --- 2. FUNÇÕES DE DADOS E MODELAGEM (REESTRUTURADO E OTIMIZADO) ---

def buscar_e_preparar_dados(ticker: str) -> Tuple[pd.DataFrame | None, Dict | None]:
    """
    Busca dados do yfinance para um ticker, calcula indicadores técnicos
    e retorna informações básicas da ação.
    """
    try:
        yf_ticker = f"{ticker.strip().upper()}.SA"
        acao = yf.Ticker(yf_ticker)
        
        info = acao.info
        hist = acao.history(period=CONFIG["yfinance_period"])

        if hist.empty or len(hist) < CONFIG["min_historical_data"]:
            logging.warning(f"Dados históricos insuficientes para {ticker}. Pulando.")
            return None, None

        hist = hist.reset_index()
        hist['Days'] = (hist['Date'] - hist['Date'].min()).dt.days
        
        # Média Móvel Simples (MMS) e Bandas de Bollinger
        hist['MM20'] = hist['Close'].rolling(window=CONFIG["ma_window"]).mean()
        std_dev = hist['Close'].rolling(window=CONFIG["ma_window"]).std()
        hist['UpperBand'] = hist['MM20'] + 2 * std_dev
        hist['LowerBand'] = hist['MM20'] - 2 * std_dev
        
        # Índice de Força Relativa (RSI)
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=CONFIG["rsi_window"]).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=CONFIG["rsi_window"]).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        hist['EMA12'] = hist['Close'].ewm(span=CONFIG["ema_short_window"], adjust=False).mean()
        hist['EMA26'] = hist['Close'].ewm(span=CONFIG["ema_long_window"], adjust=False).mean()
        hist['MACD'] = hist['EMA12'] - hist['EMA26']
        
        # Volume
        hist['VolumeMedia'] = hist['Volume'].rolling(window=CONFIG["ma_window"]).mean()
        
        hist.dropna(inplace=True)

        acao_info = {
            "ticker": ticker,
            "nome": info.get("longName") or info.get("shortName") or ticker,
            "setor": info.get("sector", "Setor desconhecido"),
        }
        return hist, acao_info
    
    except Exception as e:
        logging.error(f"Erro ao buscar dados para {ticker}: {e}")
        return None, None

def treinar_avaliar_e_prever(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, float]:
    """
    Treina, avalia um modelo de regressão e faz uma previsão autorregressiva.
    
    Retorna:
        - predictions: Série de preços previstos.
        - conf_lower: Limite inferior do intervalo de confiança.
        - conf_upper: Limite superior do intervalo de confiança.
        - r2: R² score do modelo na validação.
    """
    features_cols = ['Days', 'MM20', 'RSI', 'MACD']
    target_col = 'Close'
    
    X = df[features_cols]
    y = df[target_col]

    # (NOVO) Divisão para avaliação do modelo
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["test_size_for_eval"], random_state=CONFIG["model_params"]["random_state"], shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingRegressor(**CONFIG["model_params"])
    model.fit(X_train_scaled, y_train)

    # (NOVO) Avaliação do modelo
    y_pred_test = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred_test)

    # (NOVO) Treino com todos os dados para a previsão final
    X_full_scaled = scaler.fit_transform(X)
    model.fit(X_full_scaled, y)
    
    residuals = y - model.predict(X_full_scaled)
    prediction_std_error = residuals.std()

    # --- (OTIMIZADO) Loop de Previsão Autorregressiva ---
    future_predictions = []
    last_known_data = df.copy()

    for i in range(CONFIG["prediction_days"]):
        last_row = last_known_data.iloc[-1]
        
        # Prepara features do próximo dia
        next_day_features = last_row[features_cols].values.reshape(1, -1)
        next_day_features_scaled = scaler.transform(next_day_features)

        # Faz a previsão
        next_price = model.predict(next_day_features_scaled)[0]
        future_predictions.append(next_price)

        # Adiciona a nova previsão aos dados para calcular o próximo passo
        # Este é o passo autorregressivo
        new_row = pd.Series({
            'Date': last_row['Date'] + pd.Timedelta(days=1),
            'Close': next_price,
            'Days': last_row['Days'] + 1,
            'Volume': last_row['Volume'] # Assume-se o último volume
        })
        
        # Atualiza os dados para o próximo loop
        temp_df_for_indicators = pd.concat([last_known_data, new_row.to_frame().T], ignore_index=True)
        
        # Recalcula indicadores para a nova linha
        new_row['MM20'] = temp_df_for_indicators['Close'].rolling(window=CONFIG["ma_window"]).mean().iloc[-1]
        
        delta = temp_df_for_indicators['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=CONFIG["rsi_window"]).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(window=CONFIG["rsi_window"]).mean().iloc[-1]
        rs = gain / (loss + 1e-10) # Evita divisão por zero
        new_row['RSI'] = 100 - (100 / (1 + rs))

        new_row['EMA12'] = temp_df_for_indicators['Close'].ewm(span=CONFIG["ema_short_window"], adjust=False).mean().iloc[-1]
        new_row['EMA26'] = temp_df_for_indicators['Close'].ewm(span=CONFIG["ema_long_window"], adjust=False).mean().iloc[-1]
        new_row['MACD'] = new_row['EMA12'] - new_row['EMA26']
        
        # Adiciona a linha completa ao dataframe para o próximo passo
        last_known_data = pd.concat([last_known_data, new_row.to_frame().T], ignore_index=True).bfill()

    future_dates = pd.date_range(df['Date'].max() + pd.Timedelta(days=1), periods=CONFIG["prediction_days"])
    predictions = pd.Series(future_predictions, index=future_dates)
    
    conf_interval = 1.96 * prediction_std_error
    conf_upper = predictions + conf_interval
    conf_lower = predictions - conf_interval

    return predictions, conf_lower, conf_upper, r2


def criar_todos_os_graficos(ticker: str, hist: pd.DataFrame, forecast: pd.Series, conf_lower: pd.Series, conf_upper: pd.Series) -> Dict[str, str]:
    """Cria e salva TODOS os gráficos necessários para o relatório."""
    paths = {}
    os.makedirs(CONFIG["temp_graph_folder"], exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Gráfico Combinado (Histórico + Previsão)
    path_combined = os.path.join(CONFIG["temp_graph_folder"], f"{ticker}_combined.png")
    plt.figure(figsize=(10, 4.5))
    plt.plot(hist['Date'], hist['Close'], label='Histórico de Preços', color='#3498db', linewidth=1.5)
    plt.plot(forecast.index, forecast, label=f'Previsão {CONFIG["prediction_days"]} dias', color='#f1c40f', linewidth=2.5)
    plt.fill_between(forecast.index, conf_lower, conf_upper, color='#f1c40f', alpha=0.2, label='Intervalo de Confiança (95%)')
    plt.title(f"{ticker} - Histórico e Previsão de Preços", fontsize=12)
    plt.ylabel("Preço (R$)"); plt.xlabel("Data"); plt.legend()
    plt.tight_layout(); plt.savefig(path_combined, dpi=150); plt.close()
    paths['combined'] = path_combined

    # 2. Grade de Gráficos Técnicos
    path_deep_dive = os.path.join(CONFIG["temp_graph_folder"], f"{ticker}_deep_dive.png")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(f'Análise Técnica Detalhada - {ticker}', fontsize=16)

    # a) Preço e Bandas de Bollinger
    axs[0, 0].plot(hist['Date'], hist['Close'], label='Fechamento', color='#3498db')
    axs[0, 0].plot(hist['Date'], hist['MM20'], label=f'MM{CONFIG["ma_window"]}', color='orange', linestyle='--')
    axs[0, 0].fill_between(hist['Date'], hist['LowerBand'], hist['UpperBand'], color='gray', alpha=0.2, label='Bandas de Bollinger')
    axs[0, 0].set_title("Preço e Bandas de Bollinger"); axs[0, 0].legend()

    # b) Índice de Força Relativa (RSI)
    axs[0, 1].plot(hist['Date'], hist['RSI'], color='#8e44ad', label='RSI')
    axs[0, 1].axhline(70, color='red', linestyle='--', alpha=0.5, label='Sobrecompra (70)')
    axs[0, 1].axhline(30, color='green', linestyle='--', alpha=0.5, label='Sobrevenda (30)')
    axs[0, 1].set_title("Índice de Força Relativa (RSI)"); axs[0, 1].set_ylim(0, 100); axs[0, 1].legend()

    # c) MACD
    axs[1, 0].plot(hist['Date'], hist['MACD'], label='Linha MACD', color='#27ae60')
    axs[1, 0].bar(hist['Date'], hist['MACD'], color=['#c0392b' if x < 0 else '#27ae60' for x in hist['MACD']], alpha=0.4, label='Histograma MACD')
    axs[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axs[1, 0].set_title("MACD"); axs[1, 0].legend()

    # d) Volume
    axs[1, 1].bar(hist['Date'], hist['Volume'], color='grey', alpha=0.4, label='Volume Diário')
    axs[1, 1].plot(hist['Date'], hist['VolumeMedia'], color='#c0392b', label=f'Média de Volume ({CONFIG["ma_window"]}d)')
    axs[1, 1].set_title("Volume de Negociação"); axs[1, 1].legend()
    axs[1, 1].tick_params(axis='y', labelleft=False)
    
    plt.savefig(path_deep_dive, dpi=150); plt.close()
    paths['deep_dive'] = path_deep_dive
    
    return paths

# --- 3. CLASSE DE GERAÇÃO DO PDF (MELHORADA) ---

class GeradorRelatorioPDF:
    """Encapsula toda a lógica de criação do relatório SYRAFIN em PDF."""

    def __init__(self, filename: str):
        self.filename = filename
        self.doc = BaseDocTemplate(filename, pagesize=A4,
                                   leftMargin=2*28.35, rightMargin=2*28.35,
                                   topMargin=2*28.35, bottomMargin=2*28.35)
        self._setup_styles()
        self._setup_templates()
        self.story = []

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
        # Cabeçalho
        p_header = Paragraph("SYRAFIN - Relatório Preditivo de Ações", self.styles['Header'])
        w_h, h_h = p_header.wrap(doc.width, doc.topMargin)
        p_header.drawOn(canvas, doc.leftMargin, A4[1] - doc.topMargin + 10)
        
        # Rodapé
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

    def construir_relatorio(self, dados_analises: List[Dict]):
        """Constrói o PDF completo a partir dos dados de análise."""
        self._criar_pagina_capa()
        self._criar_sumario_executivo(dados_analises)

        for dados in dados_analises:
            self._adicionar_analise_ticker(dados)
            self.story.append(PageBreak())

        self._criar_pagina_aviso_legal()
        self.doc.build(self.story)
        
    def _criar_pagina_capa(self):
        self.story.append(Spacer(1, 150))
        self.story.append(Paragraph("SYRAFIN", self.styles['Title']))
        self.story.append(Paragraph("Relatório Preditivo de Ações", self.styles['SubTitle']))
        self.story.append(Spacer(1, 12))
        self.story.append(Paragraph("Análise Técnica e Preditiva de Ativos da B3", self.styles['SubTitle']))
        self.story.append(Spacer(1, 250))
        data_geracao = datetime.today().strftime('%d de %B de %Y')
        self.story.append(Paragraph(f"Gerado em: {data_geracao}", self.styles['SubTitle']))
        self.story.append(NextPageTemplate('Principal'))
        self.story.append(PageBreak())

    def _criar_sumario_executivo(self, dados_analises: List[Dict]):
        self.story.append(Paragraph("Sumário Executivo", self.styles['h1']))
        self.story.append(Paragraph("A tabela abaixo resume as análises, apresentando a tendência preditiva para cada ativo nos próximos 15 dias.", self.styles['Body']))
        self.story.append(Spacer(1, 20))
        
        dados_tabela = [["Ticker", "Nome da Empresa", "Tendência Prevista"]]
        for data in dados_analises:
            tendencia, _ = self._get_tendencia_info(data['hist']['Close'].iloc[-1], data['forecast'].iloc[-1])
            dados_tabela.append([data['info']['ticker'], data['info']['nome'], tendencia])
        
        tabela_sumario = Table(dados_tabela, colWidths=[60, 280, 100])
        tabela_sumario.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), '#16a085'), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')), ('GRID', (0, 0), (-1, -1), 1, '#2c3e50'),
        ]))
        self.story.append(tabela_sumario)
        self.story.append(PageBreak())

    def _adicionar_analise_ticker(self, data: Dict):
        info, hist, forecast, plots, r2 = data['info'], data['hist'], data['forecast'], data['plots'], data['r2']
        
        self.story.append(Paragraph(f"Análise Completa: {info['nome']} ({info['ticker']})", self.styles['h1']))
        self.story.append(Paragraph(f"<b>Setor:</b> {info['setor']}", self.styles['Body']))
        self.story.append(Spacer(1, 12))

        # Tabela de Métricas
        preco_atual = hist['Close'].iloc[-1]
        preco_futuro = forecast.iloc[-1]
        tendencia_str, cor_tendencia = self._get_tendencia_info(preco_atual, preco_futuro)
        
        # (NOVO) Lógica para cor do R²
        cor_r2 = colors.green if r2 > 0.7 else (colors.darkorange if r2 > 0.4 else colors.red)
        
        dados_metricas = [
            [Paragraph('<b>Métrica Chave</b>', self.styles['Body']), Paragraph('<b>Valor</b>', self.styles['Body'])],
            ['Preço Atual (último fech.)', f"R$ {preco_atual:.2f}"],
            [f'Previsão ({CONFIG["prediction_days"]}d)', f"R$ {preco_futuro:.2f}"],
            ['Tendência Prevista', Paragraph(f"<b>{tendencia_str}</b>", ParagraphStyle(name='Tendencia', textColor=cor_tendencia))],
            ['RSI Atual', f"{hist['RSI'].iloc[-1]:.1f}"],
            ['Performance do Modelo (R²)', Paragraph(f"<b>{r2:.2f}</b>", ParagraphStyle(name='R2', textColor=cor_r2))],
        ]
        tabela_metricas = Table(dados_metricas, colWidths=[150, '*'])
        tabela_metricas.setStyle(TableStyle([
            ('GRID', (0,0),(-1,-1), 0.5, colors.grey),
            ('BACKGROUND', (0,0),(-1,0), colors.lightgrey),
            ('VALIGN', (0,0),(-1,-1), 'MIDDLE'),
        ]))
        self.story.append(tabela_metricas)
        self.story.append(Spacer(1, 12))
        
        # Gráficos
        self.story.append(Paragraph("Visão Geral Preditiva", self.styles['h2']))
        self.story.append(Image(plots['combined'], width=480, height=216))
        self.story.append(Spacer(1, 20))
        self.story.append(Paragraph("Análise Técnica Detalhada", self.styles['h2']))
        self.story.append(Image(plots['deep_dive'], width=500, height=333))

    def _criar_pagina_aviso_legal(self):
        self.story.append(Paragraph("Aviso Legal", self.styles['h1']))
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


# --- 4. FUNÇÃO PRINCIPAL (ORQUESTRADOR) ---

def main():
    """Função principal que orquestra a busca de dados, análise e criação do relatório."""
    
    try:
        df_tickers = pd.read_csv(CONFIG["tickers_csv"])
        tickers = df_tickers['Ticker'].dropna().unique()
    except FileNotFoundError:
        logging.error(f"Arquivo de tickers '{CONFIG['tickers_csv']}' não encontrado. Crie um CSV com uma coluna 'Ticker'.")
        return

    dados_completos_analises = []
    
    # (NOVO) Barra de progresso com TQDM
    process_iterator = tqdm(
        tickers[:CONFIG["max_tickers_to_process"]], 
        desc="Processando tickers",
        unit="ticker"
    )

    for ticker in process_iterator:
        process_iterator.set_postfix_str(ticker)
        
        hist_data, acao_info = buscar_e_preparar_dados(ticker)
        if hist_data is None:
            continue
        
        try:
            forecast, conf_lower, conf_upper, r2_model = treinar_avaliar_e_prever(hist_data)
            plot_paths = criar_todos_os_graficos(ticker, hist_data, forecast, conf_lower, conf_upper)
            
            dados_completos_analises.append({
                "info": acao_info,
                "hist": hist_data,
                "forecast": forecast,
                "plots": plot_paths,
                "r2": r2_model,
            })
            logging.info(f"{ticker} processado com sucesso (R² do modelo: {r2_model:.2f}).")
        except Exception as e:
            logging.error(f"Falha ao treinar ou prever para {ticker}: {e}")

    if not dados_completos_analises:
        logging.warning("Nenhum ticker foi processado com sucesso. O relatório não será gerado.")
        return

    try:
        logging.info("Iniciando a geração do relatório PDF...")
        gerador_pdf = GeradorRelatorioPDF(CONFIG["output_pdf"])
        gerador_pdf.construir_relatorio(dados_completos_analises)
        logging.info(f"Relatório '{CONFIG['output_pdf']}' gerado com sucesso!")
    except Exception as e:
        logging.error(f"Ocorreu um erro ao gerar o PDF: {e}")
    finally:
        if os.path.exists(CONFIG["temp_graph_folder"]):
            shutil.rmtree(CONFIG["temp_graph_folder"])
            logging.info(f"Pasta temporária '{CONFIG['temp_graph_folder']}' removida.")

if __name__ == "__main__":
    main()