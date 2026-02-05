import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Libraries untuk analisis teknikal
import ta
from ta import add_all_ta_features
from ta.utils import dropna

# Libraries untuk sentimen
from transformers import pipeline
import feedparser  # Untuk Google News RSS
import snscrape.modules.twitter as sntwitter  # Untuk Twitter scraping
from datetime import datetime, timedelta
import re

# Libraries untuk ML
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

# Konfigurasi Streamlit
st.set_page_config(
    page_title="Analisis Saham IHSG - Teknikal & Sentimen",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .stock-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .prediction-positive {
        color: green;
        font-weight: bold;
    }
    .prediction-negative {
        color: red;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Daftar saham IHSG populer
ihsg_stocks = {
    'IHSG': '^JKSE',
    'BBCA': 'BBCA.JK',
    'BBRI': 'BBRI.JK',
    'TLKM': 'TLKM.JK',
    'ASII': 'ASII.JK',
    'UNVR': 'UNVR.JK',
    'ICBP': 'ICBP.JK',
    'INDF': 'INDF.JK',
    'GOTO': 'GOTO.JK',
    'BMRI': 'BMRI.JK',
    'BBNI': 'BBNI.JK',
    'BSDE': 'BSDE.JK',
    'ANTM': 'ANTM.JK',
    'PGAS': 'PGAS.JK'
}

class StockAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
    
    def get_stock_data(self, ticker, period="1y"):
        """Ambil data saham dari Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty:
                st.error(f"Tidak ada data untuk {ticker}")
                return None
                
            # Tambahkan informasi tambahan
            info = stock.info
            return df, info
        except Exception as e:
            st.error(f"Error mengambil data: {str(e)}")
            return None, {}
    
    def calculate_technical_indicators(self, df):
        """Hitung semua indikator teknikal penting"""
        # Buat copy dataframe
        df = df.copy()
        
        # 1. Moving Averages
        df['MA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['MA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['MA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        
        # 2. RSI (Relative Strength Index)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # 3. MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()
        
        # 4. Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Width'] = bb.bollinger_wband()
        
        # 5. Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3
        )
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # 6. Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        
        # 7. ATR (Average True Range) untuk volatilitas
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        
        # 8. Ichimoku Cloud (simplified)
        df['Ichimoku_Conversion'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
        df['Ichimoku_Base'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
        df['Ichimoku_SpanA'] = ((df['Ichimoku_Conversion'] + df['Ichimoku_Base']) / 2).shift(26)
        
        # 9. Williams %R
        df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)
        
        # 10. CCI (Commodity Channel Index)
        df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)
        
        # Generate trading signals
        df['Signal'] = 0
        df.loc[(df['RSI'] < 30) & (df['MACD'] > df['MACD_Signal']), 'Signal'] = 1  # Buy
        df.loc[(df['RSI'] > 70) & (df['MACD'] < df['MACD_Signal']), 'Signal'] = -1  # Sell
        
        return df
    
    def get_news_sentiment(self, stock_name):
        """Analisis sentimen dari Google News RSS"""
        try:
            # Format query untuk RSS
            query = f"{stock_name} saham Indonesia"
            rss_url = f"https://news.google.com/rss/search?q={query}&hl=id&gl=ID&ceid=ID:id"
            
            feed = feedparser.parse(rss_url)
            sentiments = []
            news_items = []
            
            for entry in feed.entries[:10]:  # Ambil 10 berita terbaru
                title = entry.title
                # Analisis sentimen
                result = self.sentiment_analyzer(title[:512])[0]
                
                # Konversi ke skor -1 sampai 1
                if "1 star" in result['label'] or "2 star" in result['label']:
                    sentiment_score = -1 * result['score']
                elif "4 star" in result['label'] or "5 star" in result['label']:
                    sentiment_score = 1 * result['score']
                else:
                    sentiment_score = 0
                
                sentiments.append(sentiment_score)
                news_items.append({
                    'title': title,
                    'sentiment': sentiment_score,
                    'label': result['label'],
                    'score': result['score']
                })
            
            if sentiments:
                avg_sentiment = np.mean(sentiments)
            else:
                avg_sentiment = 0
                
            return avg_sentiment, news_items
            
        except Exception as e:
            st.warning(f"Tidak dapat mengambil berita: {str(e)}")
            return 0, []
    
    def get_twitter_sentiment(self, stock_symbol):
        """Analisis sentimen dari Twitter (menggunakan snscrape)"""
        try:
            # Format query
            query = f"{stock_symbol} saham OR IHSG lang:id since:{(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')}"
            
            tweets = []
            sentiments = []
            
            # Scrape tweets (limit 100 untuk kecepatan)
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
                if i >= 100:
                    break
                
                # Analisis sentimen tweet
                result = self.sentiment_analyzer(tweet.content[:512])[0]
                
                # Konversi ke skor -1 sampai 1
                if "1 star" in result['label'] or "2 star" in result['label']:
                    sentiment_score = -1 * result['score']
                elif "4 star" in result['label'] or "5 star" in result['label']:
                    sentiment_score = 1 * result['score']
                else:
                    sentiment_score = 0
                
                sentiments.append(sentiment_score)
                tweets.append({
                    'content': tweet.content[:200] + "...",
                    'date': tweet.date,
                    'sentiment': sentiment_score,
                    'label': result['label']
                })
            
            if sentiments:
                avg_sentiment = np.mean(sentiments)
            else:
                avg_sentiment = 0
                
            return avg_sentiment, tweets
            
        except Exception as e:
            st.warning(f"Tidak dapat mengambil tweet: {str(e)}")
            return 0, []
    
    def prepare_features_for_prediction(self, df, news_sentiment=0, twitter_sentiment=0):
        """Siapkan fitur untuk model prediksi"""
        # Ambil data teknikal terbaru
        latest = df.iloc[-1].copy()
        
        features = {
            'RSI': latest.get('RSI', 50),
            'MACD': latest.get('MACD', 0),
            'MACD_Signal': latest.get('MACD_Signal', 0),
            'BB_Width': latest.get('BB_Width', 0),
            'Stoch_K': latest.get('Stoch_K', 50),
            'Volume_Ratio': latest.get('Volume', 0) / latest.get('Volume_MA', 1) if latest.get('Volume_MA', 0) > 0 else 1,
            'ATR_Ratio': latest.get('ATR', 0) / latest['Close'] if latest['Close'] > 0 else 0,
            'MA_20_Ratio': (latest['Close'] - latest.get('MA_20', latest['Close'])) / latest['Close'] if latest['Close'] > 0 else 0,
            'MA_50_Ratio': (latest['Close'] - latest.get('MA_50', latest['Close'])) / latest['Close'] if latest['Close'] > 0 else 0,
            'News_Sentiment': news_sentiment,
            'Twitter_Sentiment': twitter_sentiment,
            'Williams_R': latest.get('Williams_R', 0),
            'CCI': latest.get('CCI', 0)
        }
        
        return pd.DataFrame([features])
    
    def train_prediction_model(self, df, forecast_days=5):
        """Train model untuk prediksi harga"""
        try:
            # Buat target (harga di masa depan)
            df = df.copy()
            df['Target'] = df['Close'].shift(-forecast_days)
            df = df.dropna()
            
            if len(df) < 50:
                st.warning("Data tidak cukup untuk training model")
                return None, None
            
            # Pilih fitur
            feature_cols = ['RSI', 'MACD', 'MACD_Signal', 'BB_Width', 'Stoch_K', 
                          'Volume', 'ATR', 'MA_20', 'MA_50', 'Williams_R', 'CCI']
            
            # Pastikan kolom ada
            available_cols = [col for col in feature_cols if col in df.columns]
            
            X = df[available_cols]
            y = df['Target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            # Standardize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model (gunakan XGBoost)
            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluasi
            y_pred = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            return model, scaler, mae, r2
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return None, None, 0, 0
    
    def predict_future_prices(self, model, scaler, latest_features, df, days_ahead=5):
        """Prediksi harga untuk beberapa hari ke depan"""
        if model is None or scaler is None:
            return None
        
        predictions = []
        current_features = latest_features.copy()
        
        for day in range(days_ahead):
            # Scale features
            scaled_features = scaler.transform(current_features)
            
            # Predict
            pred_price = model.predict(scaled_features)[0]
            predictions.append(pred_price)
            
            # Update features untuk prediksi berikutnya (simulasi sederhana)
            # Note: Dalam implementasi nyata, ini perlu diperbaiki
            current_features.iloc[0]['RSI'] *= 0.99
            current_features.iloc[0]['MACD'] *= 0.95
        
        return predictions

# Inisialisasi analyzer
analyzer = StockAnalyzer()

# Sidebar
st.sidebar.title("⚙️ Konfigurasi Analisis")

# Pilih saham
selected_stock = st.sidebar.selectbox(
    "Pilih Saham IHSG:",
    list(ihsg_stocks.keys()),
    index=0
)

ticker = ihsg_stocks[selected_stock]

# Periode data
period_options = {
    "1 Bulan": "1mo",
    "3 Bulan": "3mo", 
    "6 Bulan": "6mo",
    "1 Tahun": "1y",
    "2 Tahun": "2y",
    "5 Tahun": "5y"
}

selected_period = st.sidebar.selectbox(
    "Periode Data:",
    list(period_options.keys()),
    index=3
)

period = period_options[selected_period]

# Konfigurasi analisis
st.sidebar.subheader("📊 Analisis Teknikal")
show_all_indicators = st.sidebar.checkbox("Tampilkan Semua Indikator", value=True)

st.sidebar.subheader("💭 Analisis Sentimen")
enable_news = st.sidebar.checkbox("Analisis Berita", value=True)
enable_twitter = st.sidebar.checkbox("Analisis Twitter", value=True)

st.sidebar.subheader("🔮 Prediksi")
forecast_days = st.sidebar.slider("Hari Prediksi", 1, 30, 5)

# Tombol analisis
if st.sidebar.button("🚀 Jalankan Analisis Lengkap", type="primary"):
    with st.spinner(f"Menganalisis {selected_stock}..."):
        # 1. Ambil data saham
        df, info = analyzer.get_stock_data(ticker, period)
        
        if df is not None:
            # 2. Analisis teknikal
            df = analyzer.calculate_technical_indicators(df)
            
            # 3. Analisis sentimen
            news_sentiment = 0
            twitter_sentiment = 0
            news_items = []
            tweets = []
            
            if enable_news:
                news_sentiment, news_items = analyzer.get_news_sentiment(selected_stock)
            
            if enable_twitter:
                twitter_sentiment, tweets = analyzer.get_twitter_sentiment(selected_stock)
            
            total_sentiment = (news_sentiment + twitter_sentiment) / 2 if (enable_news or enable_twitter) else 0
            
            # 4. Siapkan model prediksi
            latest_features = analyzer.prepare_features_for_prediction(
                df, news_sentiment, twitter_sentiment
            )
            
            # 5. Train model
            model, scaler, mae, r2 = analyzer.train_prediction_model(df, forecast_days)
            
            # 6. Prediksi
            predictions = []
            if model is not None:
                predictions = analyzer.predict_future_prices(
                    model, scaler, latest_features, df, forecast_days
                )
            
            # Tampilkan hasil di tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Dashboard", 
                "📊 Teknikal", 
                "💭 Sentimen", 
                "🔮 Prediksi",
                "📋 Data"
            ])
            
            with tab1:
                st.markdown(f"<h1 class='main-header'>📈 {selected_stock} - {info.get('longName', 'N/A')}</h1>", unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    current_price = df['Close'].iloc[-1]
                    prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
                    change = ((current_price - prev_price) / prev_price) * 100
                    
                    st.metric(
                        "Harga Saat Ini",
                        f"Rp {current_price:,.0f}",
                        f"{change:+.2f}%"
                    )
                
                with col2:
                    st.metric(
                        "Sentimen",
                        f"{total_sentiment:.2f}",
                        "Positif" if total_sentiment > 0 else "Negatif" if total_sentiment < 0 else "Netral"
                    )
                
                with col3:
                    volume = df['Volume'].iloc[-1]
                    avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
                    volume_ratio = (volume / avg_volume) if avg_volume > 0 else 1
                    
                    st.metric(
                        "Volume",
                        f"{volume:,.0f}",
                        f"{volume_ratio:.1f}x rata-rata"
                    )
                
                with col4:
                    rsi = df['RSI'].iloc[-1]
                    rsi_status = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Normal"
                    st.metric("RSI", f"{rsi:.1f}", rsi_status)
                
                # Grafik utama
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.6, 0.2, 0.2],
                    subplot_titles=('Harga Saham', 'Volume', 'RSI')
                )
                
                # Candlestick
                fig.add_trace(
                    go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name='OHLC'
                    ),
                    row=1, col=1
                )
                
                # Moving averages
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['MA_20'], name='MA 20', line=dict(color='orange')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['MA_50'], name='MA 50', line=dict(color='green')),
                    row=1, col=1
                )
                
                # Bollinger Bands
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', 
                              line=dict(color='gray', dash='dash'), opacity=0.5),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
                              line=dict(color='gray', dash='dash'), opacity=0.5,
                              fill='tonexty'),
                    row=1, col=1
                )
                
                # Volume
                colors = ['green' if row['Close'] >= row['Open'] else 'red' 
                         for _, row in df.iterrows()]
                fig.add_trace(
                    go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors),
                    row=2, col=1
                )
                
                # RSI
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
                    row=3, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
                
                fig.update_layout(
                    height=800,
                    showlegend=True,
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("📊 Analisis Teknikal Lengkap")
                
                # Pilih indikator
                indicator_options = [
                    'RSI', 'MACD', 'Stochastic', 'Bollinger Bands', 
                    'Volume', 'ATR', 'Williams %R', 'CCI', 'Ichimoku'
                ]
                selected_indicators = st.multiselect(
                    "Pilih Indikator:",
                    indicator_options,
                    default=['RSI', 'MACD', 'Bollinger Bands']
                )
                
                # Grafik untuk setiap indikator
                for indicator in selected_indicators:
                    if indicator == 'RSI':
                        fig_rsi = go.Figure()
                        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
                        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                        fig_rsi.update_layout(title="RSI (14)")
                        st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    elif indicator == 'MACD':
                        fig_macd = go.Figure()
                        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
                        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal'))
                        fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD_Diff'], name='Histogram'))
                        fig_macd.update_layout(title="MACD")
                        st.plotly_chart(fig_macd, use_container_width=True)
                    
                    elif indicator == 'Bollinger Bands':
                        fig_bb = go.Figure()
                        fig_bb.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='Upper'))
                        fig_bb.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], name='Middle'))
                        fig_bb.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='Lower'))
                        fig_bb.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='black')))
                        fig_bb.update_layout(title="Bollinger Bands")
                        st.plotly_chart(fig_bb, use_container_width=True)
                
                # Tabel sinyal trading
                st.subheader("🎯 Sinyal Trading")
                
                # Hitung sinyal terbaru
                latest_signal = df['Signal'].iloc[-1]
                signal_text = "BELI 🟢" if latest_signal == 1 else "JUAL 🔴" if latest_signal == -1 else "TUNGGU 🟡"
                
                st.markdown(f"### Sinyal Saat Ini: {signal_text}")
                
                # Buat dataframe sinyal
                signals_df = df[['Close', 'RSI', 'MACD', 'Signal']].tail(20)
                st.dataframe(signals_df.style.applymap(
                    lambda x: 'background-color: lightgreen' if x == 1 else 
                             ('background-color: lightcoral' if x == -1 else ''),
                    subset=['Signal']
                ))
            
            with tab3:
                st.subheader("💭 Analisis Sentimen")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Sentimen Berita", f"{news_sentiment:.2f}")
                    if news_items:
                        st.subheader("Berita Terbaru")
                        for i, news in enumerate(news_items[:5]):
                            sentiment_color = "green" if news['sentiment'] > 0 else "red" if news['sentiment'] < 0 else "gray"
                            st.markdown(f"""
                            <div style='border-left: 4px solid {sentiment_color}; padding-left: 10px; margin: 10px 0;'>
                            <b>{news['title']}</b><br>
                            <small>Sentimen: {news['sentiment']:.2f} ({news['label']})</small>
                            </div>
                            """, unsafe_allow_html=True)
                
                with col2:
                    st.metric("Sentimen Twitter", f"{twitter_sentiment:.2f}")
                    if tweets:
                        st.subheader("Tweet Terbaru")
                        for i, tweet in enumerate(tweets[:5]):
                            sentiment_color = "green" if tweet['sentiment'] > 0 else "red" if tweet['sentiment'] < 0 else "gray"
                            st.markdown(f"""
                            <div style='border-left: 4px solid {sentiment_color}; padding-left: 10px; margin: 10px 0;'>
                            <b>{tweet['date'].strftime('%Y-%m-%d')}</b><br>
                            {tweet['content']}<br>
                            <small>Sentimen: {tweet['sentiment']:.2f}</small>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Grafik sentimen
                if enable_news or enable_twitter:
                    fig_sentiment = go.Figure()
                    
                    # Simulasi data sentimen historis
                    dates = df.index[-30:]
                    sentiment_history = np.random.uniform(-0.5, 0.5, len(dates))
                    
                    fig_sentiment.add_trace(go.Scatter(
                        x=dates, y=sentiment_history,
                        mode='lines+markers',
                        name='Sentimen Historis'
                    ))
                    fig_sentiment.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_sentiment.update_layout(
                        title="Trend Sentimen (Simulasi)",
                        xaxis_title="Tanggal",
                        yaxis_title="Skor Sentimen"
                    )
                    st.plotly_chart(fig_sentiment, use_container_width=True)
            
            with tab4:
                st.subheader("🔮 Prediksi Harga")
                
                if predictions:
                    # Tampilkan prediksi
                    last_date = df.index[-1]
                    future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
                    
                    fig_pred = go.Figure()
                    
                    # Data historis
                    fig_pred.add_trace(go.Scatter(
                        x=df.index[-50:], y=df['Close'].iloc[-50:],
                        mode='lines',
                        name='Data Historis',
                        line=dict(color='blue')
                    ))
                    
                    # Prediksi
                    fig_pred.add_trace(go.Scatter(
                        x=future_dates, y=predictions,
                        mode='lines+markers',
                        name='Prediksi',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig_pred.update_layout(
                        title=f"Prediksi Harga {selected_stock} ({forecast_days} hari ke depan)",
                        xaxis_title="Tanggal",
                        yaxis_title="Harga (Rp)",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Tabel prediksi
                    pred_df = pd.DataFrame({
                        'Tanggal': future_dates,
                        'Prediksi Harga': predictions,
                        'Perubahan %': [((pred - df['Close'].iloc[-1]) / df['Close'].iloc[-1] * 100) for pred in predictions]
                    })
                    
                    st.dataframe(pred_df.style.format({
                        'Prediksi Harga': 'Rp {:,.0f}',
                        'Perubahan %': '{:.2f}%'
                    }))
                    
                    # Metrics akurasi model
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")
                    with col2:
                        st.metric("R² Score", f"{r2:.2f}")
                else:
                    st.warning("Model prediksi tidak dapat dibuat. Data mungkin tidak cukup.")
            
            with tab5:
                st.subheader("📋 Data Lengkap")
                
                # Tampilkan data dengan indikator
                display_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                              'RSI', 'MACD', 'MA_20', 'MA_50', 'Signal']
                
                available_cols = [col for col in display_cols if col in df.columns]
                
                st.dataframe(df[available_cols].tail(50))
                
                # Download data
                csv = df[available_cols].to_csv()
                st.download_button(
                    label="📥 Download Data (CSV)",
                    data=csv,
                    file_name=f"{selected_stock}_data.csv",
                    mime="text/csv"
                )
                
                # Informasi saham
                st.subheader("ℹ️ Informasi Saham")
                info_cols = {
                    'Sektor': 'sector',
                    'Industri': 'industry',
                    'Market Cap': 'marketCap',
                    'P/E Ratio': 'trailingPE',
                    'Dividend Yield': 'dividendYield',
                    'Beta': 'beta'
                }
                
                info_data = {}
                for label, key in info_cols.items():
                    value = info.get(key, 'N/A')
                    if isinstance(value, (int, float)):
                        if label == 'Market Cap':
                            value = f"Rp {value:,.0f}"
                        elif label == 'Dividend Yield' and value:
                            value = f"{value*100:.2f}%"
                    info_data[label] = value
                
                st.json(info_data)
        else:
            st.error("Gagal mengambil data saham. Silakan coba lagi.")

else:
    # Tampilan awal
    st.markdown("<h1 class='main-header'>📊 Analisis Saham IHSG - Teknikal & Sentimen</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ## Selamat Datang di Dashboard Analisis Saham IHSG
    
    Aplikasi ini menggabungkan:
    
    ### 📊 **Analisis Teknikal Lengkap**
    - Moving Averages (20, 50, 200 hari)
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
    - Stochastic Oscillator
    - Volume Analysis
    - Dan 10+ indikator lainnya
    
    ### 💭 **Analisis Sentimen Real-time**
    - Sentimen dari berita finansial
    - Sentimen dari media sosial (Twitter)
    - Analisis menggunakan model NLP terbaru
    
    ### 🔮 **Prediksi Machine Learning**
    - Model XGBoost untuk prediksi harga
    - Integrasi teknikal + sentimen
    - Backtesting dan evaluasi
    
    ### 🚀 **Cara Menggunakan:**
    1. Pilih saham dari sidebar
    2. Konfigurasi periode dan analisis
    3. Klik **"Jalankan Analisis Lengkap"**
    4. Jelajahi hasil di berbagai tab
    
    ### ⚠️ **Disclaimer:**
    Analisis ini hanya untuk edukasi dan penelitian. 
    Investasi saham mengandung risiko. Selalu lakukan due diligence.
    """)
    
    # Tampilkan contoh saham
    st.subheader("📈 Saham IHSG Populer")
    
    cols = st.columns(4)
    for i, (name, ticker) in enumerate(list(ihsg_stocks.items())[:8]):
        with cols[i % 4]:
            st.markdown(f"""
            <div class='stock-card'>
                <h4>{name}</h4>
                <p>{ticker}</p>
            </div>
            """, unsafe_allow_html=True)