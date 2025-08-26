"""
Live Trading Monitor page for ML Trading Dashboard.
Real-time position tracking, P&L monitoring, and trade execution.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional

# Import custom components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.dashboard.theme import DashboardTheme
from src.dashboard.components.charts import TradingCharts
from src.dashboard.components.metrics import MetricCards
from src.dashboard.websocket_client import StreamlitWebSocketManager, MockDataGenerator

def render_live_trading_page():
    """Render the live trading monitor page."""
    
    # Initialize theme
    theme = st.session_state.get('theme', 'dark')
    theme_config = DashboardTheme.THEMES[theme]
    
    # Page header
    st.markdown("# 🔴 Live Trading Monitor")
    st.markdown("Real-time position tracking and market monitoring")
    
    # Initialize WebSocket manager
    ws_manager = StreamlitWebSocketManager()
    
    # Connection status
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        ws_manager.render_connection_status()
    
    st.markdown("---")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Market Overview",
        "💼 Positions",
        "📈 Live Chart",
        "🚦 Signals",
        "🔔 Alerts"
    ])
    
    with tab1:
        render_market_overview(ws_manager, theme_config)
    
    with tab2:
        render_positions_tab(ws_manager, theme_config)
    
    with tab3:
        render_live_chart(ws_manager, theme_config)
    
    with tab4:
        render_signals_tab(ws_manager, theme_config)
    
    with tab5:
        render_alerts_tab(ws_manager, theme_config)
    
    # Auto-refresh
    if st.sidebar.checkbox("🔄 Auto-refresh", value=True, key="live_refresh"):
        refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 1, 10, 2, key="live_refresh_rate")
        time.sleep(refresh_rate)
        st.rerun()


def render_market_overview(ws_manager: StreamlitWebSocketManager, theme: Dict):
    """Render market overview section."""
    
    st.markdown("### 📊 Market Overview")
    
    # Get market data (use mock if no WebSocket)
    market_data = ws_manager.get_market_data()
    if not market_data:
        market_data = MockDataGenerator.generate_market_data()
    
    # Market metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        price_color = "🟢" if market_data.get('change_24h', 0) >= 0 else "🔴"
        st.metric(
            "BTC Price",
            f"${market_data.get('price', 0):,.2f}",
            f"{market_data.get('change_24h', 0):.2%} {price_color}"
        )
    
    with col2:
        st.metric(
            "24h Volume",
            f"${market_data.get('volume', 0):,.0f}",
            f"{((market_data.get('volume', 0) / 1000) - 3):.1f}K"
        )
    
    with col3:
        spread = market_data.get('ask', 0) - market_data.get('bid', 0)
        st.metric(
            "Bid/Ask Spread",
            f"${spread:.2f}",
            f"{(spread / market_data.get('price', 1)) * 100:.3f}%"
        )
    
    with col4:
        st.metric(
            "24h High/Low",
            f"${market_data.get('high_24h', 0):,.0f}",
            f"${market_data.get('low_24h', 0):,.0f}"
        )
    
    # Market sentiment indicators
    st.markdown("### 📊 Market Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Fear & Greed Index (mock)
        fear_greed = np.random.randint(20, 80)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fear_greed,
            title={'text': "Fear & Greed Index"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': theme['primary']},
                'steps': [
                    {'range': [0, 25], 'color': theme['danger']},
                    {'range': [25, 50], 'color': theme['warning']},
                    {'range': [50, 75], 'color': theme['info']},
                    {'range': [75, 100], 'color': theme['success']}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': fear_greed
                }
            }
        ))
        fig.update_layout(
            height=250,
            paper_bgcolor=theme['chart_bg'],
            font={'color': theme['text_primary']}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Funding Rate
        funding_rate = np.random.uniform(-0.01, 0.01)
        funding_color = theme['success'] if funding_rate > 0 else theme['danger']
        
        fig = go.Figure(go.Indicator(
            mode="number+delta",
            value=funding_rate * 100,
            title={'text': "Funding Rate (%)"},
            number={'suffix': "%", 'font': {'color': funding_color}},
            delta={'reference': 0, 'relative': False},
            domain={'x': [0, 1], 'y': [0, 1]}
        ))
        fig.update_layout(
            height=250,
            paper_bgcolor=theme['chart_bg'],
            font={'color': theme['text_primary']}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Open Interest
        open_interest = np.random.uniform(1, 5) * 1e9
        oi_change = np.random.uniform(-10, 10)
        
        fig = go.Figure(go.Indicator(
            mode="number+delta",
            value=open_interest / 1e9,
            title={'text': "Open Interest"},
            number={'prefix': "$", 'suffix': "B"},
            delta={'reference': open_interest / 1e9 - oi_change/100, 'relative': True},
            domain={'x': [0, 1], 'y': [0, 1]}
        ))
        fig.update_layout(
            height=250,
            paper_bgcolor=theme['chart_bg'],
            font={'color': theme['text_primary']}
        )
        st.plotly_chart(fig, use_container_width=True)


def render_positions_tab(ws_manager: StreamlitWebSocketManager, theme: Dict):
    """Render positions tab."""
    
    st.markdown("### 💼 Open Positions")
    
    # Get positions (use mock if no WebSocket)
    positions = ws_manager.get_positions()
    if not positions:
        positions = MockDataGenerator.generate_positions()
    
    if positions:
        # Convert to DataFrame
        df = pd.DataFrame(positions)
        
        # Summary metrics
        total_value = df['value'].sum()
        total_pnl = df['pnl'].sum()
        total_pnl_pct = df['pnl_pct'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Positions", len(df))
        
        with col2:
            st.metric("Total Value", f"${total_value:,.2f}")
        
        with col3:
            pnl_color = "🟢" if total_pnl >= 0 else "🔴"
            st.metric(
                "Total P&L",
                f"${total_pnl:,.2f}",
                f"{total_pnl_pct:.2%} {pnl_color}"
            )
        
        with col4:
            st.metric("Avg Position", f"${total_value/len(df):,.2f}")
        
        # Position details table
        st.markdown("### Position Details")
        
        # Format DataFrame for display
        display_df = df[['symbol', 'side', 'entry_price', 'current_price', 'quantity', 'value', 'pnl', 'pnl_pct']].copy()
        display_df['pnl_pct'] = display_df['pnl_pct'].apply(lambda x: f"{x:.2f}%")
        display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:,.2f}")
        display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:,.2f}")
        display_df['value'] = display_df['value'].apply(lambda x: f"${x:,.2f}")
        display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:,.2f}")
        
        # Apply color coding
        def color_pnl(val):
            if '$' in str(val):
                num = float(val.replace('$', '').replace(',', ''))
                color = theme['success'] if num >= 0 else theme['danger']
            elif '%' in str(val):
                num = float(val.replace('%', ''))
                color = theme['success'] if num >= 0 else theme['danger']
            else:
                color = theme['text_primary']
            return f'color: {color}'
        
        styled_df = display_df.style.applymap(color_pnl, subset=['pnl', 'pnl_pct'])
        st.dataframe(styled_df, use_container_width=True)
        
        # P&L Distribution chart
        st.markdown("### P&L Distribution")
        
        fig = go.Figure()
        colors = [theme['success'] if p >= 0 else theme['danger'] for p in df['pnl']]
        
        fig.add_trace(go.Bar(
            x=df['symbol'],
            y=df['pnl'],
            marker_color=colors,
            text=[f"${p:,.0f}" for p in df['pnl']],
            textposition='outside'
        ))
        
        fig.update_layout(
            height=300,
            paper_bgcolor=theme['chart_bg'],
            plot_bgcolor=theme['chart_bg'],
            font={'color': theme['text_primary']},
            xaxis={'gridcolor': theme['chart_grid']},
            yaxis={'gridcolor': theme['chart_grid'], 'title': 'P&L ($)'},
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No open positions")


def render_live_chart(ws_manager: StreamlitWebSocketManager, theme: Dict):
    """Render live price chart."""
    
    st.markdown("### 📈 Live Price Chart")
    
    # Chart settings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h"], index=2)
    
    with col2:
        chart_type = st.selectbox("Chart Type", ["Candlestick", "Line", "Area"], index=0)
    
    with col3:
        indicators = st.multiselect("Indicators", ["SMA", "EMA", "Bollinger", "RSI", "MACD"], default=["SMA"])
    
    # Generate mock OHLCV data
    periods = 100
    dates = pd.date_range(end=datetime.now(), periods=periods, freq='15T')
    
    # Random walk for price
    returns = np.random.randn(periods) * 0.002
    price = 50000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': price * (1 + np.random.randn(periods) * 0.001),
        'high': price * (1 + np.abs(np.random.randn(periods)) * 0.002),
        'low': price * (1 - np.abs(np.random.randn(periods)) * 0.002),
        'close': price,
        'volume': np.random.uniform(1000, 5000, periods)
    }, index=dates)
    
    # Add indicators
    if "SMA" in indicators:
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
    
    if "EMA" in indicators:
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
    
    if "Bollinger" in indicators:
        sma = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        df['bb_upper'] = sma + (std * 2)
        df['bb_lower'] = sma - (std * 2)
    
    if "RSI" in indicators:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
    
    if "MACD" in indicators:
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Create chart
    indicator_list = [i.lower() for i in indicators]
    chart = TradingCharts.create_candlestick_chart(
        df.dropna(),
        theme,
        indicators=indicator_list,
        height=600
    )
    
    st.plotly_chart(chart, use_container_width=True)


def render_signals_tab(ws_manager: StreamlitWebSocketManager, theme: Dict):
    """Render trading signals tab."""
    
    st.markdown("### 🚦 Trading Signals")
    
    # Get latest prediction
    prediction = ws_manager.get_latest_prediction()
    if not prediction:
        prediction = MockDataGenerator.generate_prediction()
    
    # Signal display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        signal = prediction.get('signal', 'NEUTRAL')
        signal_color = {
            'LONG': theme['success'],
            'SHORT': theme['danger'],
            'NEUTRAL': theme['warning']
        }.get(signal, theme['text_primary'])
        
        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px; background-color: {theme['bg_card']}; border-radius: 10px;">
                <h2 style="color: {signal_color}; margin: 0;">{signal}</h2>
                <p style="color: {theme['text_muted']};">Current Signal</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        confidence = prediction.get('confidence', 0)
        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px; background-color: {theme['bg_card']}; border-radius: 10px;">
                <h2 style="color: {theme['primary']}; margin: 0;">{confidence:.1%}</h2>
                <p style="color: {theme['text_muted']};">Confidence</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        prediction_val = prediction.get('prediction', 0.5)
        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px; background-color: {theme['bg_card']}; border-radius: 10px;">
                <h2 style="color: {theme['info']}; margin: 0;">{prediction_val:.3f}</h2>
                <p style="color: {theme['text_muted']};">Probability</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Signal history
    st.markdown("### 📜 Signal History")
    
    # Generate mock signal history
    signal_history = []
    for i in range(20):
        timestamp = datetime.now() - timedelta(minutes=15*i)
        pred = MockDataGenerator.generate_prediction()
        pred['timestamp'] = timestamp
        signal_history.append(pred)
    
    df = pd.DataFrame(signal_history)
    
    # Signal accuracy chart
    fig = go.Figure()
    
    colors = [theme['success'] if s == 'LONG' else theme['danger'] if s == 'SHORT' else theme['warning'] 
              for s in df['signal']]
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['prediction'],
        mode='markers+lines',
        marker=dict(size=10, color=colors),
        line=dict(color=theme['primary'], width=1),
        name='Predictions'
    ))
    
    # Add threshold lines
    fig.add_hline(y=0.65, line_dash="dash", line_color=theme['success'], opacity=0.5)
    fig.add_hline(y=0.35, line_dash="dash", line_color=theme['danger'], opacity=0.5)
    
    fig.update_layout(
        height=300,
        paper_bgcolor=theme['chart_bg'],
        plot_bgcolor=theme['chart_bg'],
        font={'color': theme['text_primary']},
        xaxis={'gridcolor': theme['chart_grid']},
        yaxis={'gridcolor': theme['chart_grid'], 'title': 'Probability', 'range': [0, 1]},
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_alerts_tab(ws_manager: StreamlitWebSocketManager, theme: Dict):
    """Render alerts tab."""
    
    st.markdown("### 🔔 System Alerts")
    
    # Get alerts
    alerts = ws_manager.get_alerts(limit=20)
    
    # Generate mock alerts if none
    if not alerts:
        alerts = [
            {'timestamp': datetime.now() - timedelta(minutes=5), 'type': 'trade', 'message': 'Long position opened on BTCUSDT', 'severity': 'info'},
            {'timestamp': datetime.now() - timedelta(minutes=10), 'type': 'risk', 'message': 'Drawdown approaching limit', 'severity': 'warning'},
            {'timestamp': datetime.now() - timedelta(minutes=15), 'type': 'signal', 'message': 'Strong buy signal detected', 'severity': 'success'},
            {'timestamp': datetime.now() - timedelta(minutes=30), 'type': 'system', 'message': 'Model retrained successfully', 'severity': 'info'},
            {'timestamp': datetime.now() - timedelta(hours=1), 'type': 'error', 'message': 'API rate limit reached', 'severity': 'error'},
        ]
    
    # Display alerts
    for alert in alerts:
        severity_icon = {
            'info': '💡',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌'
        }.get(alert['severity'], '📢')
        
        severity_color = {
            'info': theme['info'],
            'success': theme['success'],
            'warning': theme['warning'],
            'error': theme['danger']
        }.get(alert['severity'], theme['text_primary'])
        
        st.markdown(
            f"""
            <div style="padding: 10px; margin-bottom: 10px; background-color: {theme['bg_card']}; 
                        border-left: 4px solid {severity_color}; border-radius: 5px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 1.2em; margin-right: 10px;">{severity_icon}</span>
                        <span style="color: {theme['text_primary']}; font-weight: 500;">
                            {alert['message']}
                        </span>
                    </div>
                    <span style="color: {theme['text_muted']}; font-size: 0.875em;">
                        {alert['timestamp'].strftime('%H:%M:%S')}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Alert settings
    st.markdown("### ⚙️ Alert Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.checkbox("📧 Email notifications", value=False)
        st.checkbox("🔔 Desktop notifications", value=True)
        st.checkbox("📱 Mobile push notifications", value=False)
    
    with col2:
        st.multiselect(
            "Alert types",
            ["Trade Execution", "Risk Warnings", "Signal Changes", "System Events", "Errors"],
            default=["Trade Execution", "Risk Warnings", "Errors"]
        )
        
        st.slider("Alert threshold (confidence)", 0.0, 1.0, 0.7, 0.05)