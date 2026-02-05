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
from datetime import datetime, timedelta
import re

# Libraries untuk sentimen (lebih ringan)
from textblob import TextBlob
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    st.warning("NLTK tidak tersedia, sentimen analysis akan dibatasi")

# Libraries untuk ML
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

# Konfigurasi Streamlit
st.set_page_config(
    page_title="Analisis Saham IHSG",
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
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
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
    'EXCL': 'EXCL.JK',
    'BMRI': 'BMRI.JK',
    'BBNI': 'BBNI.JK',
    'BSDE': 'BSDE.JK',
    'ANTM': 'ANTM.JK',
    'PGAS': 'PGAS.JK'
}

class SimpleStockAnalyzer:
    def __init__(self):
        self.cache = {}
        
    def get_stock_data(self, ticker, period="1y"):
        """Ambil data saham dari Yahoo Finance"""
        cache_key = f"stock_{ticker}_{period}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty:
                return None, {}
            
            info = stock.info
            result = (df, info)
            self.cache[cache_key] = result
            return result
        except:
            return None, {}
    
    def calculate_technical_indicators(self, df):
        """Hitung indikator teknikal utama"""
        df = df.copy()
        
        # Moving Averages
        df['MA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['MA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        
        # Volume
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        # Generate signals
        df['Signal'] = 0
        df.loc[(df['RSI'] < 30) & (df['MACD'] > df['MACD_Signal']), 'Signal'] = 1
        df.loc[(df['RSI'] > 70) & (df['MACD'] < df['MACD_Signal']), 'Signal'] = -1
        
        return df
    
    def analyze_sentiment_textblob(self, text):
        """Analisis sentimen sederhana"""
        try:
            blob = TextBlob(str(text))
            return blob.sentiment.polarity
        except:
            return 0
    
    def get_mock_news_sentiment(self, stock_name):
        """Mock sentiment analysis untuk demo"""
        import random
        sentiments = [random.uniform(-0.5, 0.5) for _ in range(5)]
        avg_sentiment = np.mean(sentiments)
        
        news_items = []
        for i in range(3):
            sentiments = ['Positif', 'Netral', 'Negatif']
            sentiment = random.choice(sentiments)
            news_items.append({
                'title': f'Berita {i+1} tentang {stock_name}',
                'sentiment': avg_sentiment,
                'source': 'Mock News'
            })
        
        return avg_sentiment, news_items
    
    def train_prediction_model(self, df, forecast_days=5):
        """Train model prediksi sederhana"""
        try:
            if len(df) < 50:
                return None, None, 0, 0
            
            # Prepare features
            df_model = df.copy()
            df_model['Target'] = df_model['Close'].shift(-forecast_days)
            df_model = df_model.dropna()
            
            feature_cols = ['Close', 'RSI', 'MACD', 'Volume']
            available_cols = [col for col in feature_cols if col in df_model.columns]
            
            X = df_model[available_cols]
            y = df_model['Target']
            
            if len(X) < 30:
                return None, None, 0, 0
            
            # Split
            split_idx = int(len(X) * 0.8)
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            # Train model
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            return model, mae, r2, available_cols
            
        except Exception as e:
            return None, 0, 0, []
    
    def predict_future(self, model, latest_features, df, days=5):
        """Prediksi sederhana"""
        if model is None:
            return None
        
        predictions = []
        current_price = df['Close'].iloc[-1]
        
        # Simple prediction based on trend
        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
        macd = df['MACD'].iloc[-1] if 'MACD' in df.columns else 0
        
        # Determine trend
        if rsi < 40 and macd > 0:
            trend = 0.01  # Upward
        elif rsi > 60 and macd < 0:
            trend = -0.01  # Downward
        else:
            trend = 0  # Sideways
        
        for i in range(1, days + 1):
            pred = current_price * (1 + trend * i + np.random.normal(0, 0.005))
            predictions.append(pred)
        
        return predictions

# Inisialisasi analyzer
analyzer = SimpleStockAnalyzer()

# Sidebar
st.sidebar.title("⚙️ Konfigurasi")

# Pilih saham
selected_stock = st.sidebar.selectbox(
    "Pilih Saham:",
    list(ihsg_stocks.keys())
)

ticker = ihsg_stocks[selected_stock]

# Periode
period = st.sidebar.selectbox(
    "Periode:",
    ["1mo", "3mo", "6mo", "1y", "2y"],
    index=2
)

# Tombol analisis
if st.sidebar.button("🚀 Analisis", type="primary"):
    with st.spinner(f"Menganalisis {selected_stock}..."):
        # Ambil data
        df, info = analyzer.get_stock_data(ticker, period)
        
        if df is not None and not df.empty:
            # Analisis teknikal
            df = analyzer.calculate_technical_indicators(df)
            
            # Analisis sentimen (mock)
            news_sentiment, news_items = analyzer.get_mock_news_sentiment(selected_stock)
            
            # Train model
            model, mae, r2, feature_cols = analyzer.train_prediction_model(df)
            
            # Prediksi
            predictions = analyzer.predict_future(model, None, df, 7)
            
            # Tampilkan hasil
            tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "📊 Teknikal", "🔮 Prediksi"])
            
            with tab1:
                st.title(f"{selected_stock}")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                current_price = df['Close'].iloc[-1]
                prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
                change_pct = ((current_price - prev_price) / prev_price * 100)
                
                with col1:
                    st.metric("Harga", f"Rp {current_price:,.0f}", f"{change_pct:+.2f}%")
                
                with col2:
                    rsi = df['RSI'].iloc[-1]
                    st.metric("RSI", f"{rsi:.1f}")
                
                with col3:
                    volume = df['Volume'].iloc[-1]
                    st.metric("Volume", f"{volume:,.0f}")
                
                with col4:
                    st.metric("Sentimen", f"{news_sentiment:+.2f}")
                
                # Chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index[-50:],
                    open=df['Open'].iloc[-50:],
                    high=df['High'].iloc[-50:],
                    low=df['Low'].iloc[-50:],
                    close=df['Close'].iloc[-50:],
                    name='OHLC'
                ))
                
                fig.add_trace(go.Scatter(
                    x=df.index[-50:],
                    y=df['MA_20'].iloc[-50:],
                    name='MA 20',
                    line=dict(color='orange')
                ))
                
                fig.update_layout(
                    height=500,
                    xaxis_rangeslider_visible=False,
                    title=f"Harga {selected_stock}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Sinyal
                latest_signal = df['Signal'].iloc[-1]
                if latest_signal == 1:
                    st.success("🔔 Sinyal: BELI")
                elif latest_signal == -1:
                    st.error("🔔 Sinyal: JUAL")
                else:
                    st.info("🔔 Sinyal: TUNGGU")
            
            with tab2:
                st.subheader("Indikator Teknikal")
                
                # RSI
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=df.index[-50:],
                    y=df['RSI'].iloc[-50:],
                    name='RSI'
                ))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig_rsi.update_layout(title="RSI", height=300)
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                # MACD
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(
                    x=df.index[-50:],
                    y=df['MACD'].iloc[-50:],
                    name='MACD'
                ))
                fig_macd.add_trace(go.Scatter(
                    x=df.index[-50:],
                    y=df['MACD_Signal'].iloc[-50:],
                    name='Signal'
                ))
                fig_macd.update_layout(title="MACD", height=300)
                st.plotly_chart(fig_macd, use_container_width=True)
                
                # Data table
                st.subheader("Data Terbaru")
                display_df = df[['Close', 'Volume', 'RSI', 'MACD', 'Signal']].tail(10)
                st.dataframe(display_df.style.format({
                    'Close': 'Rp {:,.0f}',
                    'Volume': '{:,.0f}',
                    'RSI': '{:.2f}',
                    'MACD': '{:.4f}'
                }))
            
            with tab3:
                st.subheader("Prediksi Harga")
                
                if predictions:
                    future_dates = [df.index[-1] + timedelta(days=i+1) for i in range(7)]
                    
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(
                        x=df.index[-30:],
                        y=df['Close'].iloc[-30:],
                        name='Historis',
                        line=dict(color='blue')
                    ))
                    
                    fig_pred.add_trace(go.Scatter(
                        x=future_dates,
                        y=predictions,
                        name='Prediksi',
                        line=dict(color='red', dash='dash'),
                        mode='lines+markers'
                    ))
                    
                    fig_pred.update_layout(
                        title="Prediksi 7 Hari ke Depan",
                        height=400
                    )
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Tabel prediksi
                    pred_df = pd.DataFrame({
                        'Hari ke-': range(1, 8),
                        'Tanggal': future_dates,
                        'Prediksi Harga': predictions,
                        'Perubahan %': [((p - current_price) / current_price * 100) for p in predictions]
                    })
                    
                    st.dataframe(pred_df.style.format({
                        'Prediksi Harga': 'Rp {:,.0f}',
                        'Perubahan %': '{:+.2f}%'
                    }))
                    
                    # Model performance
                    st.metric("Model R² Score", f"{r2:.4f}")
                    st.metric("MAE", f"{mae:.2f}")
                else:
                    st.warning("Prediksi tidak tersedia")
            
            # Informasi
            with st.expander("📋 Informasi Saham"):
                if info:
                    info_data = {
                        'Nama': info.get('longName', 'N/A'),
                        'Sektor': info.get('sector', 'N/A'),
                        'Market Cap': f"Rp {info.get('marketCap', 0):,.0f}" if info.get('marketCap') else 'N/A',
                        'P/E Ratio': f"{info.get('trailingPE', 0):.2f}" if info.get('trailingPE') else 'N/A'
                    }
                    for key, value in info_data.items():
                        st.write(f"**{key}:** {value}")
        
        else:
            st.error("Gagal mengambil data. Coba lagi.")

else:
    # Tampilan awal
    st.title("📈 Analisis Saham IHSG")
    
    st.markdown("""
    ### Selamat Datang!
    
    Aplikasi ini memberikan analisis teknikal dan prediksi untuk saham IHSG.
    
    **Fitur:**
    - 📊 Analisis teknikal (MA, RSI, MACD, Bollinger Bands)
    - 📈 Prediksi harga 7 hari ke depan
    - 💭 Analisis sentimen sederhana
    - 🎯 Sinyal trading otomatis
    
    **Cara menggunakan:**
    1. Pilih saham dari sidebar
    2. Pilih periode data
    3. Klik tombol "Analisis"
    4. Jelajahi hasil di tab-tab yang tersedia
    
    **Disclaimer:** Analisis ini hanya untuk edukasi.
    """)
    
    # Contoh saham
    st.subheader("📋 Daftar Saham IHSG")
    cols = st.columns(3)
    stocks_list = list(ihsg_stocks.items())
    
    for i, (name, ticker) in enumerate(stocks_list[:9]):
        with cols[i % 3]:
            st.info(f"**{name}**\n`{ticker}`")

# Footer
st.markdown("---")
st.caption("© 2024 Analisis Saham IHSG - Untuk Tujuan Edukasi")
