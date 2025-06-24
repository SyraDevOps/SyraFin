# syrafin_institutional_grade_v2.py

# --- Core Libraries ---
import os
import shutil
import logging
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

# --- Data & Analysis ---
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import shap

# --- Machine Learning ---
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import qrcode
from PIL import Image as PILImage

# --- PDF Generation ---
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, Image as RLImage, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Spacer, Table,
                                TableStyle, Flowable)

# --- Import de tqdm para a correção ---
import tqdm # <--- CORREÇÃO AQUI: Importação correta

# --- Initial Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
pd.options.mode.chained_assignment = None

# --- 1. CONFIGURATION MODULE ---
class Config:
    TICKERS_CSV = "dados_acoes.csv"
    OUTPUT_PDF = "Relatorio_SYRAFIN_Institucional_v2.pdf"
    TEMP_FOLDER = "syrafin_temp_output"
    LOGO_FILE = "syrafin_logo.png"
    
    YFINANCE_PERIOD = "2y"
    MIN_HISTORICAL_DATA = 200

    PREDICTION_HORIZON = 30
    ENSEMBLE_WEIGHTS = {'gbr': 0.6, 'rfr': 0.3, 'lr': 0.1}
    
    COLOR_PRIMARY = "#003f5c"
    COLOR_SECONDARY = "#58508d"
    COLOR_ACCENT = "#ffa600"
    COLOR_SUCCESS = "#2E7D32"
    COLOR_DANGER = "#C62828"
    COLOR_NEUTRAL = "#616161"
    
    MAX_TICKERS_TO_PROCESS = 20 #tickers
    MAX_NEWS_ARTICLES_PER_TICKER = 5
    CPU_WORKERS = os.cpu_count() or 1
    
    try:
        SIA = SentimentIntensityAnalyzer()
    except LookupError:
        import nltk; nltk.download('vader_lexicon'); SIA = SentimentIntensityAnalyzer()

# --- 2. DATA HANDLING MODULE ---
class DataHandler:
    def __init__(self, ticker: str):
        self.ticker_str = ticker.strip().upper()
        self.yf_ticker_str = f"{self.ticker_str}.SA"
        self.yf_ticker = yf.Ticker(self.yf_ticker_str)
        self.hist_data, self.info, self.news = None, None, None
        self.features_cols = []

    def fetch_base_data(self) -> bool:
        try:
            self.hist_data = self.yf_ticker.history(period=Config.YFINANCE_PERIOD)
            if self.hist_data.empty or len(self.hist_data) < Config.MIN_HISTORICAL_DATA:
                logging.warning(f"Insufficient data for {self.ticker_str}."); return False
            self.hist_data.index = pd.to_datetime(self.hist_data.index).tz_localize(None)
            self.info = self.yf_ticker.info
            self.news = self.yf_ticker.news
            return True
        except Exception as e:
            logging.error(f"Failed to fetch base data for {self.ticker_str}: {e}"); return False

    def engineer_features(self):
        df = self.hist_data.copy()
        df.ta.bbands(length=20, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df['returns_1d'] = df['Close'].pct_change(1)
        df['returns_5d'] = df['Close'].pct_change(5)
        df['volume_delta'] = df['Volume'].diff()
        df['RSI_change'] = df['RSI_14'].diff()
        df['target_direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        df.dropna(inplace=True)
        self.hist_data = df
        
        self.features_cols = ['BBM_20_2.0', 'RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9',
                              'returns_1d', 'returns_5d', 'volume_delta', 'RSI_change']

    def get_historical_performance(self) -> Dict[str, float]:
        perf = {}
        for months in [3, 6, 12]:
            try:
                past_date = self.hist_data.index[-1] - timedelta(days=months * 30)
                past_price = self.hist_data.loc[self.hist_data.index.asof(past_date), 'Close']
                current_price = self.hist_data['Close'].iloc[-1]
                perf[f'{months}M'] = ((current_price - past_price) / past_price) * 100
            except (KeyError, IndexError):
                perf[f'{months}M'] = 0.0
        return perf

    def process(self):
        if not self.fetch_base_data(): return None
        self.engineer_features()
        if len(self.hist_data) < Config.MIN_HISTORICAL_DATA:
            logging.warning(f"Data for {self.ticker_str} insufficient after feature engineering.")
            return None
        
        performance = self.get_historical_performance()
        return self.hist_data, self.info, self.news, performance, self.features_cols

# --- 3. SENTIMENT ANALYSIS MODULE ---
class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', min_df=1)

    def analyze(self, news_list: List[Dict]):
        if not news_list: return [], 0.0
        
        valid_news = [n for n in news_list if 'title' in n and n['title']]
        if not valid_news: return [], 0.0
        
        titles = [n['title'] for n in valid_news]
        try:
            tfidf_matrix = self.vectorizer.fit_transform(titles)
            relevance_scores = tfidf_matrix.sum(axis=1).A1
        except ValueError:
            relevance_scores = [0.0] * len(titles)

        processed_news, sentiment_scores = [], []
        for i, article in enumerate(valid_news):
            sentiment = Config.SIA.polarity_scores(article['title'])
            article['sentiment_score'] = sentiment['compound']
            article['relevance_score'] = relevance_scores[i]
            article['publish_time'] = datetime.fromtimestamp(article['providerPublishTime']).strftime('%d/%m/%Y') if 'providerPublishTime' in article else 'N/A'
            processed_news.append(article)
            sentiment_scores.append(sentiment['compound'])
        
        processed_news.sort(key=lambda x: x['relevance_score'], reverse=True)
        overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        return processed_news[:Config.MAX_NEWS_ARTICLES_PER_TICKER], overall_sentiment

# --- 4. MACHINE LEARNING MODULE ---
class ModelSuite:
    def __init__(self, df: pd.DataFrame, features: List[str]):
        self.df = df; self.features_cols = features
        self.scaler = StandardScaler()
        self.models, self.classifier = {}, None
        self.explainer, self.feature_names = None, None

    def _prepare_data(self):
        X = self.df[self.features_cols]
        self.feature_names = X.columns.tolist()
        y_reg = self.df['Close']
        y_cls = self.df['target_direction']
        
        X_train, X_test, self.y_reg_train, self.y_reg_test, self.y_cls_train, self.y_cls_test = \
            train_test_split(X, y_reg, y_cls, test_size=0.2, shuffle=False)
        
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)

    def train(self):
        self.models['gbr'] = GradientBoostingRegressor(random_state=42)
        self.models['rfr'] = RandomForestRegressor(random_state=42)
        self.models['lr'] = LinearRegression()
        for model in self.models.values(): model.fit(self.X_train_scaled, self.y_reg_train)
        
        self.classifier = GradientBoostingClassifier(random_state=42)
        self.classifier.fit(self.X_train_scaled, self.y_cls_train)

    def evaluate(self):
        ensemble_preds = sum(self.models[name].predict(self.X_test_scaled) * w for name, w in Config.ENSEMBLE_WEIGHTS.items())
        cls_preds = self.classifier.predict(self.X_test_scaled)
        return {'r2_score': r2_score(self.y_reg_test, ensemble_preds),
                'accuracy': accuracy_score(self.y_cls_test, cls_preds)}

    def generate_shap(self):
        self.explainer = shap.TreeExplainer(self.models['gbr'])
        shap_data_sample = self.X_train_scaled[:100]
        self.shap_values = self.explainer(shap_data_sample)

    def predict_future(self):
        last_known_data = self.df.copy()
        future_predictions, future_probabilities = [], []
        
        for _ in range(Config.PREDICTION_HORIZON):
            last_features_df = last_known_data[self.features_cols].iloc[-1:]
            last_features_scaled = self.scaler.transform(last_features_df)
            
            pred = sum(self.models[name].predict(last_features_scaled)[0] * w for name, w in Config.ENSEMBLE_WEIGHTS.items())
            prob = self.classifier.predict_proba(last_features_scaled)[0][1]
            future_predictions.append(pred); future_probabilities.append(prob)

            new_row_data = last_known_data.iloc[-1:].copy()
            new_row_data.index += timedelta(days=1)
            new_row_data['Close'] = pred
            
            temp_df = pd.concat([last_known_data.iloc[-(len(self.df)//2):], new_row_data])
            temp_df.ta.bbands(length=20, append=True); temp_df.ta.rsi(length=14, append=True)
            temp_df.ta.macd(fast=12, slow=26, signal=9, append=True)
            temp_df['returns_1d'] = temp_df['Close'].pct_change(1)
            temp_df['returns_5d'] = temp_df['Close'].pct_change(5)
            temp_df['volume_delta'] = temp_df['Volume'].diff()
            temp_df['RSI_change'] = temp_df['RSI_14'].diff()
            last_known_data = temp_df.bfill().infer_objects(copy=False)

        future_dates = pd.date_range(self.df.index.max() + timedelta(days=1), periods=Config.PREDICTION_HORIZON)
        return pd.Series(future_predictions, index=future_dates), pd.Series(future_probabilities, index=future_dates)

    def run_all(self):
        try:
            self._prepare_data(); self.train(); eval_metrics = self.evaluate(); self.generate_shap()
            future_prices, future_probs = self.predict_future()
            return {"evaluation": eval_metrics, "future_prices": future_prices,
                    "future_probabilities": future_probs, "shap_values": self.shap_values,
                    "explainer": self.explainer, "feature_names": self.feature_names}
        except Exception as e:
            logging.error(f"Error in ML pipeline for {self.df.iloc[0].name if not self.df.empty else 'Unknown Ticker'}: {e}", exc_info=True); return None

# --- 5. PLOTTING MODULE ---
class Plotter:
    def __init__(self, ticker_str, temp_folder):
        self.ticker = ticker_str
        self.folder = temp_folder
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette([Config.COLOR_PRIMARY, Config.COLOR_ACCENT, Config.COLOR_SECONDARY])

    def plot_price_and_prediction(self, hist_df, forecast_prices, news):
        path = os.path.join(self.folder, f"{self.ticker}_price.png")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # <--- CORREÇÃO AQUI ---
        # Aplica .tail() no DataFrame/Série ANTES de passar para o plot
        hist_subset = hist_df.tail(90)
        ax.plot(hist_subset.index, hist_subset['Close'], label='Histórico (90d)', color=Config.COLOR_PRIMARY, lw=2)
        # --- FIM DA CORREÇÃO ---
        
        ax.plot(forecast_prices.index, forecast_prices, label=f'Previsão ({Config.PREDICTION_HORIZON}d)', color=Config.COLOR_ACCENT, lw=2.5, marker='o', ls='--')
        
        for n in news:
            try:
                news_date = pd.to_datetime(n['publish_time'], format='%d/%m/%Y')
                if hist_df.index.min() <= news_date <= hist_df.index.max():
                    price_on_date = hist_df.asof(news_date)['Close']
                    ax.annotate(f"📰", (news_date, price_on_date), xytext=(0, 15), textcoords="offset points",
                                ha='center', fontsize=14, arrowprops=dict(arrowstyle="->", color=Config.COLOR_SECONDARY))
            except (ValueError, TypeError):
                continue # Ignora notícias com data inválida

        ax.set_title(f"{self.ticker} - Previsão de Preço com Eventos", fontsize=14)
        ax.set_ylabel("Preço (R$)"); ax.legend(); ax.grid(True, which='both', linestyle=':')
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def plot_correlation_heatmap(self, df, features):
        path = os.path.join(self.folder, f"{self.ticker}_corr.png")
        corr = df[features + ['Close']].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
        plt.title(f'Correlação de Indicadores - {self.ticker}', fontsize=14)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def plot_shap_summary(self, shap_values, feature_names):
        path = os.path.join(self.folder, f"{self.ticker}_shap.png")
        plt.figure()
        shap.summary_plot(shap_values, feature_names=feature_names, plot_type="bar", show=False, color=Config.COLOR_SECONDARY)
        plt.title(f'Impacto das Features na Previsão - {self.ticker}', fontsize=14)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return path

    def generate_interactive_plot(self, hist_df, forecast_prices, news):
        path = os.path.join(self.folder, f"{self.ticker}_interactive.html")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Close'], mode='lines', name='Histórico', line=dict(color=Config.COLOR_PRIMARY)))
        fig.add_trace(go.Scatter(x=forecast_prices.index, y=forecast_prices, mode='lines+markers', name='Previsão', line=dict(color=Config.COLOR_ACCENT, dash='dash')))
        
        for n in news:
            try:
                news_date = pd.to_datetime(n['publish_time'], format='%d/%m/%Y')
                if hist_df.index.min() <= news_date <= hist_df.index.max():
                    fig.add_vline(x=news_date, line_width=1, line_dash="dash", line_color=Config.COLOR_SECONDARY, annotation_text=n['title'][:30], annotation_position="top left")
            except (ValueError, TypeError):
                continue

        fig.update_layout(title=f'Análise Interativa - {self.ticker}', xaxis_title='Data', yaxis_title='Preço (R$)', template='plotly_white')
        fig.write_html(path, auto_open=False)
        return path
        
    def generate_qr_code(self):
        path = os.path.join(self.folder, f"{self.ticker}_qr.png")
        url = f"https://finance.yahoo.com/quote/{self.ticker}.SA"
        qr_img = qrcode.make(url)
        qr_img.save(path)
        return path

# --- 6. PDF REPORTING MODULE ---
class ReportGenerator:
    def __init__(self, all_analyses_data: List[Dict]):
        self.data = all_analyses_data
        self.doc = BaseDocTemplate(Config.OUTPUT_PDF, pagesize=A4, rightMargin=0.5*inch, leftMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
        self._setup_styles()
        self.story = []
        
    def _setup_styles(self):
        s = getSampleStyleSheet()
        self.styles = {
            'Title': ParagraphStyle('Title', parent=s['Title'], fontName='Helvetica-Bold', fontSize=28, textColor=Config.COLOR_PRIMARY),
            'SubTitle': ParagraphStyle('SubTitle', parent=s['h2'], alignment=TA_CENTER, fontSize=14, textColor=Config.COLOR_SECONDARY),
            'h1': ParagraphStyle('h1', parent=s['h1'], fontName='Helvetica-Bold', fontSize=18, textColor=Config.COLOR_PRIMARY, spaceBefore=18, spaceAfter=12, borderPadding=(5,2,10,2), borderLeft=3, borderColorLeft=Config.COLOR_ACCENT),
            'h2': ParagraphStyle('h2', parent=s['h2'], fontName='Helvetica-Bold', fontSize=14, textColor=Config.COLOR_SECONDARY, spaceBefore=12, spaceAfter=6),
            'Body': ParagraphStyle('Body', parent=s['Normal'], alignment=TA_JUSTIFY, leading=14),
            'Footer': ParagraphStyle('Footer', parent=s['Normal'], fontSize=8, alignment=TA_CENTER, textColor=colors.grey),
            'Disclaimer': ParagraphStyle('Disclaimer', parent=s['Normal'], fontSize=9, alignment=TA_JUSTIFY, backColor=colors.HexColor("#F5F5F5"), borderPadding=10, leading=12),
            'NewsTitle': ParagraphStyle('NewsTitle', fontName='Helvetica-Bold', fontSize=9),
            'NewsText': ParagraphStyle('NewsText', fontSize=9, leftIndent=10, alignment=TA_JUSTIFY),
        }
        self.table_header_style = TableStyle([('BACKGROUND', (0,0), (-1,0), Config.COLOR_PRIMARY), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                                              ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                                              ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('BOTTOMPADDING', (0,0), (-1,0), 10)])
        self.table_body_style = TableStyle([('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F0F0F0')), ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                                            ('FONTSIZE', (0,1), (-1,-1), 9), ('VALIGN', (0,1), (-1,-1), 'TOP')])

    def _header_footer(self, canvas, doc):
        canvas.saveState()
        p_header = Paragraph("SYRAFIN - Relatório Institucional", self.styles['Footer'])
        p_header.wrapOn(canvas, doc.width, doc.topMargin); p_header.drawOn(canvas, doc.leftMargin, A4[1] - 0.4*inch)
        p_footer = Paragraph(f"Página {doc.page}", self.styles['Footer'])
        p_footer.wrapOn(canvas, doc.width, doc.bottomMargin); p_footer.drawOn(canvas, doc.leftMargin, 0.3*inch)
        if os.path.exists(Config.LOGO_FILE):
            canvas.saveState(); canvas.translate(A4[0]/2, A4[1]/2); canvas.rotate(45)
            canvas.setFillColor(colors.grey, alpha=0.08); canvas.setFont("Helvetica", 100)
            canvas.drawCentredString(0, 0, "SYRAFIN"); canvas.restoreState()
        canvas.restoreState()

    def build_report(self):
        frame = Frame(self.doc.leftMargin, self.doc.bottomMargin, self.doc.width, self.doc.height, id='normal')
        self.doc.addPageTemplates([PageTemplate(id='Main', frames=frame, onPage=self._header_footer)])
        
        self.story.append(NextPageTemplate('Main'))
        self._create_cover_page()
        self._create_ranking_chapter()
        self._create_news_guide()
        
        for analysis in self.data:
            self._create_ticker_page(analysis)
        
        self._create_disclaimer_page()
        logging.info("Building PDF document..."); self.doc.build(self.story)

    def _create_cover_page(self):
        self.story.extend([
            Spacer(1, 2*inch),
            RLImage(Config.LOGO_FILE, width=1.5*inch, height=1.5*inch) if os.path.exists(Config.LOGO_FILE) else Spacer(0,0),
            Paragraph("SYRAFIN", self.styles['Title']),
            Paragraph("Relatório de Análise Quantitativa Institucional", self.styles['SubTitle']),
            Spacer(1, 4*inch),
            Paragraph(f"Gerado em: {datetime.now().strftime('%d de %B de %Y, %H:%M')}", self.styles['SubTitle']),
            PageBreak()
        ])
        
    def _create_ranking_chapter(self):
        self.story.append(Paragraph("Ranking de Ativos Analisados", self.styles['h1']))
        self.story.append(Paragraph("Os ativos são classificados com base em uma combinação de potencial de alta previsto pelo modelo, aderência do modelo (R²) e sentimento geral das notícias.", self.styles['Body']))
        
        for d in self.data:
            pred_change = (d['ml_results']['future_prices'].iloc[-1] - d['historical_data']['Close'].iloc[-1]) / d['historical_data']['Close'].iloc[-1]
            r2 = d['ml_results']['evaluation']['r2_score']
            sentiment = d['sentiment_score']
            d['composite_score'] = (0.5 * pred_change) + (0.3 * r2) + (0.2 * sentiment)

        self.data.sort(key=lambda x: x['composite_score'], reverse=True)
        
        table_data = [["Rank", "Ticker", "Previsão de Variação (30d)", "R² do Modelo", "Sentimento"]]
        for i, d in enumerate(self.data):
            pred_change = (d['ml_results']['future_prices'].iloc[-1] - d['historical_data']['Close'].iloc[-1]) / d['historical_data']['Close'].iloc[-1] * 100
            table_data.append([f"#{i+1}", d['info'].get('symbol',''), f"{pred_change:+.2f}%", f"{d['ml_results']['evaluation']['r2_score']:.2f}", f"{d['sentiment_score']:.2f}"])
        
        t = Table(table_data, colWidths=[0.6*inch, 1*inch, 2*inch, 1.5*inch, 1.5*inch])
        t.setStyle(self.table_header_style); t.setStyle(self.table_body_style)
        self.story.extend([Spacer(1, 0.2*inch), t, PageBreak()])

    def _create_news_guide(self):
        self.story.append(Paragraph("Guia de Notícias Relevantes", self.styles['h1']))
        all_news = []
        for d in self.data:
            for news_item in d['processed_news']:
                news_item['ticker'] = d['info'].get('symbol', '')
                all_news.append(news_item)
        
        all_news.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        table_data = [["Ticker", "Data", "Manchete da Notícia", "Sentimento"]]
        for n in all_news[:15]:
            score = n['sentiment_score']
            sent_str = "Positivo" if score > 0.1 else ("Negativo" if score < -0.1 else "Neutro")
            table_data.append([n['ticker'], n['publish_time'], Paragraph(n['title'], self.styles['NewsText']), sent_str])
        
        t = Table(table_data, colWidths=[0.8*inch, 0.8*inch, 4.5*inch, 1*inch])
        t.setStyle(self.table_header_style); t.setStyle(self.table_body_style)
        self.story.extend([Spacer(1, 0.2*inch), t, PageBreak()])
        
    def _create_ticker_page(self, analysis: Dict):
        info = analysis['info']; ticker = info.get('symbol', 'N/A')
        self.story.append(Paragraph(f"Análise Detalhada: {info.get('longName', ticker)} ({ticker})", self.styles['h1']))
        self._add_summary_table(analysis)
        plots = analysis['plots']
        t = Table([[RLImage(plots['price'], width=4.8*inch, height=2.4*inch), RLImage(plots['shap'], width=2.2*inch, height=2.4*inch)]], colWidths=[5*inch, 2.3*inch])
        t.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
        self.story.append(t)
        self._add_prediction_and_news_section(analysis)
        self.story.append(PageBreak())
        
    def _add_summary_table(self, analysis):
        info, hist_perf, ml_res, qr_path = analysis['info'], analysis['historical_performance'], analysis['ml_results'], analysis['plots']['qr']
        pred_price, curr_price = ml_res['future_prices'].iloc[-1], analysis['historical_data']['Close'].iloc[-1]
        
        data1 = [[Paragraph(f'<b>{k}</b>', self.styles['Body']), Paragraph(v, self.styles['Body']) if isinstance(v, str) else f"{v:.2f}"] for k, v in {'Setor': info.get('sector', 'N/A'), 'Indústria': info.get('industry', 'N/A'), 'Preço Atual': curr_price, 'Previsão (30d)': pred_price}.items()]
        data2 = [[Paragraph(f'<b>{k}</b>', self.styles['Body']), f"{v:.2f}%" if '%' not in k else f"{v:.2f}"] for k, v in {f'Perf. {m}': hist_perf.get(m, 0) for m in ['3M', '6M', '12M']}.items()]
        data2.append([Paragraph('<b>Aderência (R²)</b>', self.styles['Body']), f"{ml_res['evaluation']['r2_score']:.2f}"])
        
        t = Table([[Table(data1), Table(data2), RLImage(qr_path, width=1*inch, height=1*inch)]], colWidths=[3.2*inch, 2.8*inch, 1.2*inch])
        self.story.extend([t, Spacer(1, 0.2*inch)])
        
    def _add_prediction_and_news_section(self, analysis):
        prob_alta = analysis['ml_results']['future_probabilities'].iloc[-1] * 100
        self.story.append(Paragraph(f"<b>Diagnóstico Preditivo:</b> A probabilidade estimada de <b>alta</b> no próximo mês é de <b>{prob_alta:.1f}%</b>. O modelo projeta um movimento em direção a R$ {analysis['ml_results']['future_prices'].iloc[-1]:.2f}.", self.styles['Body']))
        
        news_data = [[Paragraph('<b>Notícias Relevantes</b>', self.styles['NewsTitle']), Paragraph('<b>Sent.</b>', self.styles['NewsTitle'])]]
        for n in analysis['processed_news']:
            score = n['sentiment_score']; sent_str = "▲" if score > 0.1 else ("▼" if score < -0.1 else "▬")
            news_data.append([Paragraph(f"({n['publish_time']}) {n['title']}", self.styles['NewsText']), sent_str])
        
        t_news = Table(news_data, colWidths=[6.7*inch, 0.5*inch]); t_news.setStyle(self.table_body_style)
        t_section = Table([[RLImage(analysis['plots']['corr'], width=3.5*inch, height=2.8*inch), t_news]], colWidths=[3.7*inch, '*'])
        t_section.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')])); self.story.extend([Spacer(1, 0.2*inch), t_section])

    def _create_disclaimer_page(self):
        self.story.append(PageBreak()); self.story.append(Paragraph("Aviso Legal e Metodologia", self.styles['h1']))
        disclaimer_text = """Este relatório foi gerado pelo sistema SYRAFIN e destina-se a fins informativos. As análises e previsões são baseadas em dados históricos e modelos estatísticos, não constituindo recomendação de compra ou venda. Investimentos envolvem riscos. O desempenho passado não é garantia de resultados futuros. Consulte sempre um profissional financeiro qualificado.<br/><br/><b>Metodologia:</b> O sistema utiliza um ensemble de modelos (Gradient Boosting, Random Forest, Regressão Linear) para prever preços e um classificador para prever a direção do movimento. Features incluem indicadores técnicos (RSI, MACD, Bandas de Bollinger) e métricas de retorno e volume. A análise de sentimento de notícias utiliza VADER e a relevância é calculada com TF-IDF. A explicabilidade do modelo é fornecida por valores SHAP."""
        self.story.append(Paragraph(disclaimer_text, self.styles['Disclaimer']))

# --- 7. ORCHESTRATION ---
def process_ticker_pipeline(ticker: str) -> Dict | None:
    logging.info(f"Starting analysis for {ticker}...")
    try:
        data_handler = DataHandler(ticker); processed_data = data_handler.process()
        if not processed_data: return None
        hist_df, info, news, performance, features = processed_data

        sentiment_analyzer = SentimentAnalyzer(); processed_news, overall_sentiment = sentiment_analyzer.analyze(news)
        model_suite = ModelSuite(hist_df, features); ml_results = model_suite.run_all()
        if not ml_results: return None
        
        plotter = Plotter(ticker, Config.TEMP_FOLDER)
        plots = {'price': plotter.plot_price_and_prediction(hist_df, ml_results['future_prices'], processed_news),
                 'corr': plotter.plot_correlation_heatmap(hist_df, features),
                 'shap': plotter.plot_shap_summary(ml_results['shap_values'], ml_results['feature_names']),
                 'interactive': plotter.generate_interactive_plot(hist_df, ml_results['future_prices'], processed_news),
                 'qr': plotter.generate_qr_code()}
        
        logging.info(f"Successfully completed analysis for {ticker}.")
        return {"info": info, "historical_data": hist_df, "processed_news": processed_news,
                "sentiment_score": overall_sentiment, "historical_performance": performance,
                "ml_results": ml_results, "plots": plots}
    except Exception as e:
        logging.error(f"CRITICAL FAILURE in pipeline for {ticker}: {e}", exc_info=False); return None

def main():
    shutil.rmtree(Config.TEMP_FOLDER, ignore_errors=True); os.makedirs(Config.TEMP_FOLDER, exist_ok=True)
    try:
        tickers = pd.read_csv(Config.TICKERS_CSV)['Ticker'].dropna().unique()
    except FileNotFoundError:
        logging.error(f"Ticker file not found: {Config.TICKERS_CSV}"); return

    all_analyses = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=Config.CPU_WORKERS) as executor:
        future_to_ticker = {executor.submit(process_ticker_pipeline, t): t for t in tickers[:Config.MAX_TICKERS_TO_PROCESS]}
        # <--- CORREÇÃO AQUI ---
        # A chamada correta é tqdm.tqdm()
        progress = tqdm.tqdm(concurrent.futures.as_completed(future_to_ticker), total=len(future_to_ticker), desc="Analyzing Tickers")
        # --- FIM DA CORREÇÃO ---
        for future in progress:
            result = future.result()
            if result: all_analyses.append(result)

    if not all_analyses:
        logging.warning("No tickers were successfully analyzed. Report will not be generated.")
    else:
        report_generator = ReportGenerator(all_analyses)
        report_generator.build_report()
        logging.info(f"Process complete. Report saved as {Config.OUTPUT_PDF}.")
        
    logging.info(f"Temporary files are in '{Config.TEMP_FOLDER}'. You can delete this folder.")

if __name__ == '__main__':
    main()