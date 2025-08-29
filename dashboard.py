import streamlit as st
import pandas as pd
import numpy as np
import time
from pykalman import KalmanFilter
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Streamlit 페이지 기본 설정 ---
st.set_page_config(page_title="실시간 Regime 대시보드", layout="wide")
st.title("📈 실시간 Regime 대시보드")
st.markdown("지정한 거래소와 티커의 데이터를 실시간으로 분석하여 시장 국면(Regime)을 시각화합니다.")

# --- 2. 핵심 계산 함수 (백테스트 코드와 동일) ---
def calculate_kalman_indicator(data, transition_covariance, vol_lookback):
    df = data.copy()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    kf = KalmanFilter(
        transition_matrices=[1], observation_matrices=[1],
        initial_state_mean=0, initial_state_covariance=1,
        observation_covariance=1.0, transition_covariance=transition_covariance
    )
    state_means, _ = kf.filter(df['log_return'].values)
    df['kalman_smoothed_mean'] = state_means.flatten()
    df['residuals_sq'] = (df['log_return'] - df['kalman_smoothed_mean'])**2
    df['kalman_smoothed_vol'] = np.sqrt(df['residuals_sq'].ewm(span=vol_lookback).mean())
    df['regime_indicator'] = df['kalman_smoothed_mean'] / (df['kalman_smoothed_vol'] + 1e-10)
    return df

def label_regime_percentile_thresholds(data, lookback, bull_percentile, bear_percentile):
    df = data.copy()
    df['bull_threshold'] = df['regime_indicator'].rolling(window=lookback).quantile(bull_percentile)
    df['bear_threshold'] = df['regime_indicator'].rolling(window=lookback).quantile(bear_percentile)
    conditions = [
        df['regime_indicator'] > df['bull_threshold'],
        df['regime_indicator'] < df['bear_threshold']
    ]
    choices = ['Bull', 'Bear']
    df['regime'] = np.select(conditions, choices, default='Neutral')
    return df

# --- 3. 데이터 로딩 및 캐싱 함수 ---
# @st.cache_data: 입력값이 바뀌지 않으면 함수를 재실행하지 않고 이전 결과(캐시)를 반환하여 API 호출을 최소화합니다.
# ttl (time-to-live): 캐시 유효 시간(초). 60 * 5 = 5분
@st.cache_data(ttl=60 * 5)
def fetch_and_process_data(exchange_name, ticker, timeframe, limit, params):
    try:
        # ccxt를 통해 거래소 객체 생성
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class()
        
        # 데이터 로드
        ohlcv = exchange.fetch_ohlcv(ticker, timeframe=timeframe, limit=limit)
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # 지표 계산
        df_regime = calculate_kalman_indicator(df, params['TC'], params['VOL_LOOKBACK'])
        df_final = label_regime_percentile_thresholds(df_regime, params['LOOKBACK'], params['BULL_PERCENTILE'], params['BEAR_PERCENTILE'])
        return df_final
    except Exception as e:
        st.error(f"데이터를 가져오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

# --- 4. 사이드바 (사용자 입력 UI) ---
with st.sidebar:
    st.header("⚙️ 설정")
    exchange_name = st.selectbox("거래소 선택", ["binance", "upbit", "bithumb", "bybit"])
    
    if exchange_name == "binance":
        ticker = st.text_input("티커 (예: BTC/USDT)", "BTC/USDT")
    else:
        ticker = st.text_input("티커 (예: BTC/KRW)", "BTC/KRW")
        
    timeframe = st.selectbox("타임프레임", ["1h", "4h", "1d"], index=1)
    limit = st.slider("데이터 개수", 200, 3000, 500)

    st.subheader("전략 파라미터")
    params = {
        'TC': st.slider("Transition Covariance (TC)", 0.000001, 0.01, 0.00005, 0.00001, format="%.5f"),
        'VOL_LOOKBACK': st.slider("Volatility Lookback", 10, 100, 30),
        'LOOKBACK': st.slider("Percentile Lookback", 20, 200, 60),
        'BULL_PERCENTILE': st.slider("Bull Percentile", 0.5, 1.0, 0.90),
        'BEAR_PERCENTILE': st.slider("Bear Percentile", 0.0, 0.7, 0.05)
    }
    
    # 새로고침 버튼
    if st.button("데이터 새로고침"):
        st.cache_data.clear() # 캐시 삭제

# --- 5. 메인 대시보드 ---
# 데이터 로드
df_live = fetch_and_process_data(exchange_name, ticker, timeframe, limit, params)

if not df_live.empty:
    # 최신 정보 추출
    latest_data = df_live.iloc[-1]
    current_time = latest_data.name
    current_regime = latest_data['regime']
    current_price = latest_data['close']
    price_change_pct = (df_live['close'].pct_change().iloc[-1]) * 100
    
    st.subheader(f"{ticker} ({timeframe}) 실시간 현황")
    
    # KPI 카드 표시
    col1, col2, col3 = st.columns(3)
    with col1:
        if current_regime == 'Bull':
            st.metric(label="현재 시장 국면", value=current_regime, delta="매수 우위")
        elif current_regime == 'Bear':
            st.metric(label="현재 시장 국면", value=current_regime, delta="매도 우위", delta_color="inverse")
        else:
            st.metric(label="현재 시장 국면", value=current_regime, delta="중립/관망", delta_color="off")
    with col2:
        st.metric(label="현재 가격", value=f"{current_price:,.4f}", delta=f"{price_change_pct:.2f}%")
    with col3:
        st.metric(label="업데이트 시간", value=current_time.strftime('%H:%M:%S'))

    # Plotly를 이용한 인터랙티브 차트 생성
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])

    # 차트 1: 가격 및 Regime 배경
    fig.add_trace(go.Scatter(x=df_live.index, y=df_live['close'], name='Price', line=dict(color='gray')), row=1, col=1)
    
    # Regime에 따라 배경색 추가
    df_bull = df_live[df_live['regime'] == 'Bull']
    for i in range(len(df_bull)):
        fig.add_vrect(x0=df_bull.index[i] - pd.Timedelta(hours=2 if timeframe=='4h' else 0.5), 
                      x1=df_bull.index[i] + pd.Timedelta(hours=2 if timeframe=='4h' else 0.5), 
                      fillcolor="blue", opacity=0.1, line_width=0, row=1, col=1)

    df_bear = df_live[df_live['regime'] == 'Bear']
    for i in range(len(df_bear)):
        fig.add_vrect(x0=df_bear.index[i] - pd.Timedelta(hours=2 if timeframe=='4h' else 0.5), 
                      x1=df_bear.index[i] + pd.Timedelta(hours=2 if timeframe=='4h' else 0.5),
                      fillcolor="red", opacity=0.1, line_width=0, row=1, col=1)
    
    # 차트 2: Regime Indicator 및 동적 임계값
    fig.add_trace(go.Scatter(x=df_live.index, y=df_live['regime_indicator'], name='Indicator', line=dict(color='purple')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_live.index, y=df_live['bull_threshold'], name='Bull Thr.', line=dict(color='royalblue', dash='dash')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_live.index, y=df_live['bear_threshold'], name='Bear Thr.', line=dict(color='crimson', dash='dash')), row=2, col=1)
    
    # 차트 레이아웃 업데이트
    fig.update_layout(
        title_text=f"{ticker} Regime 분석",
        height=600,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Indicator Value", row=2, col=1)
    
    # Streamlit에 차트 표시
    st.plotly_chart(fig, use_container_width=True)

    # 데이터 테이블 (펼치기/접기)
    with st.expander("상세 데이터 보기"):
        st.dataframe(df_live.tail(100))

else:
    st.warning("설정에 맞는 데이터를 불러올 수 없습니다. 거래소, 티커, 타임프레임을 확인해주세요.")