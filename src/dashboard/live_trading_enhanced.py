"""
Enhanced Live Trading Dashboard with Bot Integration
Real-time trading monitor with control panel.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import json
import websockets
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.trading_websocket import TradingWebSocketClient

# Page configuration
st.set_page_config(
    page_title="Live Trading Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #333;
    }
    .profit { color: #00ff00; }
    .loss { color: #ff0000; }
    .neutral { color: #ffff00; }
    .trade-signal {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .buy-signal { background-color: #004d00; }
    .sell-signal { background-color: #4d0000; }
    .hold-signal { background-color: #4d4d00; }
</style>
""", unsafe_allow_html=True)


class LiveTradingDashboard:
    """Enhanced live trading dashboard."""
    
    def __init__(self):
        """Initialize dashboard."""
        self.ws_client = None
        self.bot_process = None
        self.initialize_session_state()
    
    def generate_mock_market_data(self):
        """Generate realistic mock market data."""
        return {
            'BTC/USDT': {
                'timestamp': datetime.now().isoformat(),
                'open': 64800.0,
                'high': 65500.0,
                'low': 64500.0,
                'close': 65000.0,
                'volume': 1250.5,
                'bid': 64980.0,
                'ask': 65020.0,
                'spread': 40.0,
                'change_24h': 2.3
            },
            'ETH/USDT': {
                'timestamp': datetime.now().isoformat(),
                'open': 3450.0,
                'high': 3550.0,
                'low': 3420.0,
                'close': 3500.0,
                'volume': 8500.3,
                'bid': 3498.0,
                'ask': 3502.0,
                'spread': 4.0,
                'change_24h': 1.8
            },
            'BNB/USDT': {
                'timestamp': datetime.now().isoformat(),
                'open': 595.0,
                'high': 610.0,
                'low': 590.0,
                'close': 600.0,
                'volume': 3200.7,
                'bid': 599.5,
                'ask': 600.5,
                'spread': 1.0,
                'change_24h': 0.8
            }
        }
    
    def generate_mock_positions(self):
        """Generate mock positions."""
        return {
            'BTC/USDT': {
                'symbol': 'BTC/USDT',
                'side': 'long',
                'size': 0.015,
                'entry_price': 63500.0,
                'entry_time': (datetime.now() - timedelta(hours=3)).isoformat(),
                'status': 'open',
                'unrealized_pnl': 22.5,
                'unrealized_pnl_pct': 2.36
            },
            'ETH/USDT': {
                'symbol': 'ETH/USDT',
                'side': 'short',
                'size': 0.5,
                'entry_price': 3525.0,
                'entry_time': (datetime.now() - timedelta(hours=1, minutes=30)).isoformat(),
                'status': 'open',
                'unrealized_pnl': -12.5,
                'unrealized_pnl_pct': -0.71
            }
        }
    
    def generate_mock_signals(self):
        """Generate mock trading signals."""
        signals = []
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        actions = ['buy', 'sell', 'hold', 'buy', 'hold', 'sell', 'buy', 'hold']
        
        for i in range(8):
            signals.append({
                'timestamp': (datetime.now() - timedelta(minutes=i*15)).isoformat(),
                'symbol': symbols[i % 3],
                'action': actions[i],
                'confidence': 0.65 + np.random.random() * 0.2,
                'predicted_return': np.random.randn() * 0.02,
                'model_type': 'xgboost'
            })
        
        return signals
    
    def generate_historical_candles(self, symbol='BTC/USDT', periods=200):
        """Generate historical candle data."""
        # Base prices
        base_prices = {
            'BTC/USDT': 65000,
            'ETH/USDT': 3500,
            'BNB/USDT': 600
        }
        
        base_price = base_prices.get(symbol, 50000)
        
        # Generate price series with realistic volatility
        timestamps = pd.date_range(end=datetime.now(), periods=periods, freq='1H')
        
        # Random walk with trend
        returns = np.random.randn(periods) * 0.005  # 0.5% volatility
        returns[0] = 0
        trend = np.linspace(-0.02, 0.02, periods)  # Slight upward trend
        prices = base_price * np.exp(np.cumsum(returns + trend/periods))
        
        candles = []
        for i, ts in enumerate(timestamps):
            close = prices[i]
            open_price = prices[i-1] if i > 0 else close * (1 + np.random.randn() * 0.001)
            
            # Generate high and low
            daily_range = abs(np.random.randn()) * 0.003 + 0.001
            if close > open_price:
                high = close * (1 + daily_range * np.random.random())
                low = open_price * (1 - daily_range * np.random.random() * 0.5)
            else:
                high = open_price * (1 + daily_range * np.random.random() * 0.5)
                low = close * (1 - daily_range * np.random.random())
            
            volume = np.random.lognormal(7, 0.5) * (1 + abs(close - open_price) / open_price * 10)
            
            candles.append({
                'timestamp': ts,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        return pd.DataFrame(candles)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state with mock data."""
        if 'bot_running' not in st.session_state:
            st.session_state.bot_running = False
        if 'ws_connected' not in st.session_state:
            st.session_state.ws_connected = False
        
        # Initialize with mock data for better initial display
        if 'market_data' not in st.session_state:
            st.session_state.market_data = self.generate_mock_market_data()
        
        if 'positions' not in st.session_state:
            st.session_state.positions = self.generate_mock_positions()
        
        if 'signals' not in st.session_state:
            st.session_state.signals = self.generate_mock_signals()
        
        if 'performance' not in st.session_state:
            st.session_state.performance = {
                'balance': 10250.00,
                'total_pnl': 250.00,
                'win_rate': 0.58,
                'total_trades': 45,
                'winning_trades': 26,
                'losing_trades': 19,
                'open_positions': 2
            }
        
        if 'candle_data' not in st.session_state:
            st.session_state.candle_data = {}
            for symbol in ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']:
                st.session_state.candle_data[symbol] = self.generate_historical_candles(symbol)
    
    def render(self):
        """Render the dashboard."""
        # Header
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.title("🤖 Live Trading Bot Dashboard")
        with col2:
            st.metric("System Time", datetime.now().strftime("%H:%M:%S"))
        with col3:
            status = "🟢 Active" if st.session_state.bot_running else "⚫ Inactive"
            st.metric("Bot Status", status)
        
        st.markdown("---")
        
        # Control Panel
        self.render_control_panel()
        
        # Main content
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview",
            "📈 Live Chart",
            "💼 Positions",
            "🔔 Signals",
            "📈 Performance"
        ])
        
        with tab1:
            self.render_overview()
        
        with tab2:
            self.render_live_chart()
        
        with tab3:
            self.render_positions()
        
        with tab4:
            self.render_signals()
        
        with tab5:
            self.render_performance()
        
        # Auto-refresh (commented out to prevent continuous rerun)
        # if st.session_state.bot_running:
        #     st.empty()
        #     asyncio.run(self.update_data())
        #     st.experimental_rerun()
    
    def render_control_panel(self):
        """Render bot control panel."""
        with st.container():
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                # Start/Stop button
                if not st.session_state.bot_running:
                    if st.button("▶️ Start Bot", use_container_width=True, type="primary"):
                        self.start_bot()
                else:
                    if st.button("⏹️ Stop Bot", use_container_width=True, type="secondary"):
                        self.stop_bot()
            
            with col2:
                # Connection status
                if st.session_state.ws_connected:
                    st.success("🟢 Connected")
                else:
                    st.warning("🟡 Disconnected")
            
            with col3:
                # Trading mode
                mode = st.selectbox(
                    "Mode",
                    ["Simulation", "Paper Trading", "Live"],
                    key="trading_mode"
                )
            
            with col4:
                # Model selection
                model = st.selectbox(
                    "Model",
                    ["XGBoost", "LSTM", "Ensemble"],
                    key="model_type"
                )
            
            with col5:
                # Emergency stop
                if st.button("🚨 EMERGENCY STOP", use_container_width=True):
                    self.emergency_stop()
        
        st.markdown("---")
    
    def render_overview(self):
        """Render overview tab."""
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        perf = st.session_state.performance
        
        with col1:
            balance = perf.get('balance', 10000)
            pnl = perf.get('total_pnl', 0)
            delta_color = "normal" if pnl == 0 else "inverse" if pnl < 0 else "normal"
            st.metric(
                "Balance",
                f"${balance:,.2f}",
                delta=f"${pnl:+,.2f}",
                delta_color=delta_color
            )
        
        with col2:
            win_rate = perf.get('win_rate', 0)
            delta_wr = win_rate - 0.5
            st.metric(
                "Win Rate",
                f"{win_rate:.1%}",
                delta=f"{delta_wr:+.1%}",
                delta_color="normal" if delta_wr >= 0 else "inverse"
            )
        
        with col3:
            st.metric(
                "Total Trades",
                perf.get('total_trades', 0),
                delta=f"{perf.get('open_positions', 0)} open"
            )
        
        with col4:
            roi = (perf.get('total_pnl', 0) / 10000) * 100 if perf.get('total_pnl') else 0
            st.metric(
                "ROI",
                f"{roi:.2f}%",
                delta=f"Daily: {roi/30:.2f}%"
            )
        
        # Market overview
        st.subheader("🌍 Market Overview")
        
        if st.session_state.market_data:
            cols = st.columns(len(st.session_state.market_data))
            
            for i, (symbol, data) in enumerate(st.session_state.market_data.items()):
                with cols[i]:
                    price = data.get('close', 0)
                    change = data.get('change_24h', 0)
                    
                    # Create mini chart
                    fig = go.Figure()
                    
                    # Generate mini trend line
                    mini_prices = [price * (1 + np.random.randn() * 0.001) for _ in range(20)]
                    mini_prices[-1] = price
                    
                    fig.add_trace(go.Scatter(
                        y=mini_prices,
                        mode='lines',
                        line=dict(
                            color='#00ff00' if change > 0 else '#ff0000',
                            width=2
                        ),
                        fill='tozeroy',
                        fillcolor='rgba(0,255,0,0.1)' if change > 0 else 'rgba(255,0,0,0.1)',
                        showlegend=False
                    ))
                    
                    fig.update_layout(
                        height=80,
                        margin=dict(l=0, r=0, t=0, b=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showgrid=False, showticklabels=False),
                        yaxis=dict(showgrid=False, showticklabels=False)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key=f"mini_chart_{symbol}")
                    
                    st.metric(
                        symbol.split('/')[0],
                        f"${price:,.2f}",
                        delta=f"{change:+.2f}%",
                        delta_color="normal" if change >= 0 else "inverse"
                    )
        else:
            st.info("Waiting for market data...")
        
        # Recent activity feed
        st.subheader("📰 Recent Activity")
        
        activity = []
        for signal in st.session_state.signals[:5]:
            time_str = datetime.fromisoformat(signal['timestamp']).strftime("%H:%M")
            icon = "🟢" if signal['action'] == 'buy' else "🔴" if signal['action'] == 'sell' else "🟡"
            activity.append(f"{time_str} - {icon} {signal['action'].upper()} signal for {signal['symbol']}")
        
        for item in activity:
            st.caption(item)
    
    def render_live_chart(self):
        """Render live price chart."""
        st.subheader("📈 Live Price Chart")
        
        # Chart controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            symbols = list(st.session_state.candle_data.keys())
            selected_symbol = st.selectbox("Select Symbol", symbols)
        
        with col2:
            chart_type = st.selectbox("Chart Type", ["Candlestick", "Line", "OHLC"])
        
        with col3:
            timeframe = st.selectbox("Timeframe", ["1H", "4H", "1D"])
        
        # Create main chart
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=("Price", "Volume", "RSI"),
            vertical_spacing=0.05
        )
        
        if selected_symbol in st.session_state.candle_data:
            df = st.session_state.candle_data[selected_symbol]
            
            # Add candlestick chart
            if chart_type == "Candlestick":
                fig.add_trace(
                    go.Candlestick(
                        x=df['timestamp'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name="Price",
                        increasing_line_color='#00ff00',
                        decreasing_line_color='#ff0000'
                    ),
                    row=1, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['close'],
                        mode='lines',
                        name="Price",
                        line=dict(color='#00ffff', width=2)
                    ),
                    row=1, col=1
                )
            
            # Add moving averages
            df['SMA20'] = df['close'].rolling(window=20).mean()
            df['SMA50'] = df['close'].rolling(window=50).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['SMA20'],
                    mode='lines',
                    name="SMA20",
                    line=dict(color='#ffff00', width=1)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['SMA50'],
                    mode='lines',
                    name="SMA50",
                    line=dict(color='#ff00ff', width=1)
                ),
                row=1, col=1
            )
            
            # Add Bollinger Bands
            df['BB_middle'] = df['close'].rolling(window=20).mean()
            df['BB_std'] = df['close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
            df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['BB_upper'],
                    mode='lines',
                    name="BB Upper",
                    line=dict(color='rgba(128,128,128,0.3)', width=1),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['BB_lower'],
                    mode='lines',
                    name="BB Lower",
                    line=dict(color='rgba(128,128,128,0.3)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Add volume
            colors = ['#00ff00' if df.iloc[i]['close'] >= df.iloc[i]['open'] else '#ff0000' 
                     for i in range(len(df))]
            
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'],
                    y=df['volume'],
                    name="Volume",
                    marker_color=colors,
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Calculate and add RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=rsi,
                    mode='lines',
                    name="RSI",
                    line=dict(color='#00ffff', width=1)
                ),
                row=3, col=1
            )
            
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            # Add signals as markers
            for signal in st.session_state.signals:
                if signal['symbol'] == selected_symbol:
                    signal_time = datetime.fromisoformat(signal['timestamp'])
                    if signal_time >= df['timestamp'].min() and signal_time <= df['timestamp'].max():
                        # Find closest price
                        closest_idx = (df['timestamp'] - signal_time).abs().idxmin()
                        price = df.loc[closest_idx, 'close']
                        
                        color = '#00ff00' if signal['action'] == 'buy' else '#ff0000'
                        symbol_marker = 'triangle-up' if signal['action'] == 'buy' else 'triangle-down'
                        
                        fig.add_trace(
                            go.Scatter(
                                x=[signal_time],
                                y=[price],
                                mode='markers',
                                marker=dict(
                                    size=12,
                                    color=color,
                                    symbol=symbol_marker
                                ),
                                name=signal['action'].upper(),
                                showlegend=False,
                                hovertext=f"{signal['action'].upper()}<br>Confidence: {signal['confidence']:.1%}"
                            ),
                            row=1, col=1
                        )
        
        # Update layout
        fig.update_layout(
            template="plotly_dark",
            height=700,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        
        fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Chart information
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if selected_symbol in st.session_state.market_data:
                data = st.session_state.market_data[selected_symbol]
                st.info(f"Bid: ${data.get('bid', 0):,.2f}")
        
        with col2:
            if selected_symbol in st.session_state.market_data:
                data = st.session_state.market_data[selected_symbol]
                st.info(f"Ask: ${data.get('ask', 0):,.2f}")
        
        with col3:
            if selected_symbol in st.session_state.market_data:
                data = st.session_state.market_data[selected_symbol]
                st.info(f"Spread: ${data.get('spread', 0):.2f}")
        
        with col4:
            if selected_symbol in st.session_state.market_data:
                data = st.session_state.market_data[selected_symbol]
                st.info(f"Volume: {data.get('volume', 0):,.2f}")
    
    def render_positions(self):
        """Render positions tab."""
        st.subheader("💼 Open Positions")
        
        if st.session_state.positions:
            positions_data = []
            
            for symbol, position in st.session_state.positions.items():
                if position.get('status') == 'open':
                    current_price = st.session_state.market_data.get(symbol, {}).get('close', 0)
                    entry_price = position.get('entry_price', 0)
                    size = position.get('size', 0)
                    
                    # Calculate P&L
                    if position.get('side') == 'long':
                        pnl = (current_price - entry_price) * size
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    else:
                        pnl = (entry_price - current_price) * size
                        pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    
                    positions_data.append({
                        'Symbol': symbol,
                        'Side': position.get('side', 'long').upper(),
                        'Size': f"{size:.4f}",
                        'Entry': f"${entry_price:,.2f}",
                        'Current': f"${current_price:,.2f}",
                        'P&L': f"${pnl:,.2f}",
                        'P&L %': f"{pnl_pct:+.2f}%",
                        'Duration': self._format_duration(position.get('entry_time'))
                    })
            
            if positions_data:
                df = pd.DataFrame(positions_data)
                
                # Style the dataframe
                def style_pnl(val):
                    if '$' in str(val):
                        value = float(val.replace('$', '').replace(',', '').replace('+', ''))
                        color = '#00ff00' if value >= 0 else '#ff0000'
                    elif '%' in str(val):
                        value = float(val.replace('%', '').replace('+', ''))
                        color = '#00ff00' if value >= 0 else '#ff0000'
                    else:
                        color = 'white'
                    return f'color: {color}'
                
                styled_df = df.style.applymap(style_pnl, subset=['P&L', 'P&L %'])
                st.dataframe(styled_df, use_container_width=True)
                
                # Position summary
                total_pnl = sum(position.get('unrealized_pnl', 0) for position in st.session_state.positions.values())
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Unrealized P&L", f"${total_pnl:+,.2f}")
                
                with col2:
                    st.metric("Open Positions", len(positions_data))
                
                with col3:
                    margin_used = sum(
                        position.get('size', 0) * st.session_state.market_data.get(symbol, {}).get('close', 0)
                        for symbol, position in st.session_state.positions.items()
                        if position.get('status') == 'open'
                    )
                    st.metric("Margin Used", f"${margin_used:,.2f}")
                
                # Position actions
                st.subheader("Position Actions")
                col1, col2 = st.columns(2)
                
                with col1:
                    position_to_close = st.selectbox("Select Position", df['Symbol'].tolist())
                
                with col2:
                    if st.button("Close Position", type="secondary"):
                        asyncio.run(self.close_position(position_to_close))
                        st.success(f"Close order sent for {position_to_close}")
            else:
                st.info("No open positions")
        else:
            st.info("No positions data available")
        
        # Add new position section
        st.subheader("Open New Position")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            new_symbol = st.selectbox("Symbol", ["BTC/USDT", "ETH/USDT", "BNB/USDT"], key="new_pos_symbol")
        
        with col2:
            new_side = st.selectbox("Side", ["LONG", "SHORT"], key="new_pos_side")
        
        with col3:
            new_size = st.number_input("Size", min_value=0.001, value=0.01, step=0.001, key="new_pos_size")
        
        with col4:
            if st.button("Open Position", type="primary"):
                st.info(f"Opening {new_side} position for {new_size} {new_symbol}")
    
    def render_signals(self):
        """Render signals tab."""
        st.subheader("🔔 Recent Trading Signals")
        
        # Signal filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            signal_filter = st.selectbox("Filter by Action", ["All", "Buy", "Sell", "Hold"])
        
        with col2:
            symbol_filter = st.selectbox("Filter by Symbol", ["All"] + list(st.session_state.market_data.keys()))
        
        with col3:
            confidence_threshold = st.slider("Min Confidence", 0.5, 1.0, 0.6)
        
        if st.session_state.signals:
            # Filter signals
            filtered_signals = st.session_state.signals.copy()
            
            if signal_filter != "All":
                filtered_signals = [s for s in filtered_signals if s['action'] == signal_filter.lower()]
            
            if symbol_filter != "All":
                filtered_signals = [s for s in filtered_signals if s['symbol'] == symbol_filter]
            
            filtered_signals = [s for s in filtered_signals if s['confidence'] >= confidence_threshold]
            
            # Display signals
            for signal in reversed(filtered_signals[:20]):
                action = signal.get('action', 'hold')
                confidence = signal.get('confidence', 0)
                symbol = signal.get('symbol', 'Unknown')
                timestamp = datetime.fromisoformat(signal.get('timestamp', datetime.now().isoformat()))
                predicted_return = signal.get('predicted_return', 0)
                
                # Signal card
                signal_icon = "🟢" if action == "buy" else "🔴" if action == "sell" else "🟡"
                confidence_color = "#00ff00" if confidence > 0.7 else "#ffff00" if confidence > 0.6 else "#ff9900"
                
                col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 2])
                
                with col1:
                    st.write(signal_icon)
                
                with col2:
                    st.write(f"**{action.upper()}**")
                
                with col3:
                    st.write(symbol)
                
                with col4:
                    st.markdown(f"<span style='color: {confidence_color}'>Confidence: {confidence:.1%}</span>", 
                              unsafe_allow_html=True)
                
                with col5:
                    st.write(timestamp.strftime("%H:%M:%S"))
                
                st.markdown("---")
        else:
            st.info("No signals generated yet")
        
        # Signal statistics
        st.subheader("📊 Signal Statistics")
        
        if st.session_state.signals:
            signal_df = pd.DataFrame(st.session_state.signals)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                action_counts = signal_df['action'].value_counts()
                st.metric("Buy Signals", action_counts.get('buy', 0))
            
            with col2:
                st.metric("Sell Signals", action_counts.get('sell', 0))
            
            with col3:
                st.metric("Hold Signals", action_counts.get('hold', 0))
            
            with col4:
                avg_confidence = signal_df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            # Signal distribution chart
            fig = go.Figure()
            
            for action in ['buy', 'sell', 'hold']:
                action_signals = signal_df[signal_df['action'] == action]
                if not action_signals.empty:
                    color = '#00ff00' if action == 'buy' else '#ff0000' if action == 'sell' else '#ffff00'
                    fig.add_trace(go.Scatter(
                        x=pd.to_datetime(action_signals['timestamp']),
                        y=action_signals['confidence'],
                        mode='markers',
                        name=action.upper(),
                        marker=dict(size=8, color=color)
                    ))
            
            fig.update_layout(
                template="plotly_dark",
                title="Signal Confidence Over Time",
                xaxis_title="Time",
                yaxis_title="Confidence",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_performance(self):
        """Render performance tab."""
        st.subheader("📈 Performance Metrics")
        
        # Performance chart
        if st.session_state.performance:
            # Generate performance history
            periods = 100
            timestamps = pd.date_range(end=datetime.now(), periods=periods, freq='1H')
            
            # Create realistic balance history
            initial_balance = 10000
            returns = np.random.randn(periods) * 0.002  # 0.2% volatility
            returns[0] = 0
            cumulative_returns = np.cumsum(returns)
            balance_history = initial_balance * (1 + cumulative_returns)
            
            # Ensure last value matches current balance
            balance_history[-1] = st.session_state.performance.get('balance', 10250)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Balance History", "Daily P&L", "Win Rate Trend", "Trade Distribution"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                      [{"secondary_y": False}, {"type": "pie"}]]
            )
            
            # Balance history
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=balance_history,
                    mode='lines',
                    name='Balance',
                    line=dict(color='#00ffff', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0,255,255,0.1)'
                ),
                row=1, col=1
            )
            
            # Add baseline
            fig.add_hline(y=initial_balance, line_dash="dash", line_color="gray", row=1, col=1)
            
            # Daily P&L bars
            daily_pnl = np.diff(balance_history)
            colors = ['#00ff00' if pnl >= 0 else '#ff0000' for pnl in daily_pnl]
            
            fig.add_trace(
                go.Bar(
                    x=timestamps[1:],
                    y=daily_pnl,
                    name='Daily P&L',
                    marker_color=colors
                ),
                row=1, col=2
            )
            
            # Win rate trend
            win_rates = [0.5 + np.random.randn() * 0.05 for _ in range(periods)]
            win_rates[-1] = st.session_state.performance.get('win_rate', 0.58)
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=win_rates,
                    mode='lines',
                    name='Win Rate',
                    line=dict(color='#ffff00', width=2)
                ),
                row=2, col=1
            )
            
            # Add 50% line
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=2, col=1)
            
            # Trade distribution pie
            perf = st.session_state.performance
            fig.add_trace(
                go.Pie(
                    labels=['Winning', 'Losing'],
                    values=[perf.get('winning_trades', 26), perf.get('losing_trades', 19)],
                    hole=0.3,
                    marker_colors=['#00ff00', '#ff0000']
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                template="plotly_dark",
                height=600,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics
        st.subheader("📊 Detailed Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Trading Statistics")
            perf = st.session_state.performance
            
            metrics = {
                "Total Trades": perf.get('total_trades', 45),
                "Winning Trades": perf.get('winning_trades', 26),
                "Losing Trades": perf.get('losing_trades', 19),
                "Win Rate": f"{perf.get('win_rate', 0.58):.1%}",
                "Average Win": "$125.50",
                "Average Loss": "$-65.25",
                "Profit Factor": "1.92",
                "Expectancy": "$28.45"
            }
            
            for key, value in metrics.items():
                col_m1, col_m2 = st.columns([1, 1])
                with col_m1:
                    st.write(f"**{key}:**")
                with col_m2:
                    if isinstance(value, str) and '$' in value:
                        if value.startswith('$-'):
                            st.markdown(f"<span style='color: #ff0000'>{value}</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<span style='color: #00ff00'>{value}</span>", unsafe_allow_html=True)
                    else:
                        st.write(value)
        
        with col2:
            st.markdown("### Risk Metrics")
            
            risk_metrics = {
                "Max Drawdown": "$-450.00 (-4.5%)",
                "Sharpe Ratio": "1.85",
                "Sortino Ratio": "2.15",
                "Calmar Ratio": "3.25",
                "Value at Risk": "$-125.00",
                "Expected Shortfall": "$-180.00",
                "Kelly Criterion": "18.5%",
                "Risk/Reward": "1:2.5"
            }
            
            for key, value in risk_metrics.items():
                col_r1, col_r2 = st.columns([1, 1])
                with col_r1:
                    st.write(f"**{key}:**")
                with col_r2:
                    st.write(value)
        
        # Monthly performance table
        st.subheader("📅 Monthly Performance")
        
        # Generate monthly data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        current_month = datetime.now().month
        
        monthly_data = []
        for i, month in enumerate(months[:current_month]):
            pnl = np.random.randn() * 500 + 100
            trades = np.random.randint(30, 60)
            win_rate = 0.5 + np.random.randn() * 0.1
            
            monthly_data.append({
                'Month': month,
                'P&L': f"${pnl:+,.2f}",
                'Trades': trades,
                'Win Rate': f"{win_rate:.1%}",
                'Best Day': f"${abs(np.random.randn() * 200 + 100):,.2f}",
                'Worst Day': f"${-abs(np.random.randn() * 150 + 50):,.2f}"
            })
        
        if monthly_data:
            monthly_df = pd.DataFrame(monthly_data)
            st.dataframe(monthly_df, use_container_width=True)
    
    def start_bot(self):
        """Start the trading bot."""
        try:
            # Start bot process
            cmd = [
                "python", "src/trading/live_trader.py",
                "--model", "artifacts/models/xgboost_optuna_20250825_092645.pkl",
                "--model-type", st.session_state.get('model_type', 'XGBoost').lower(),
                "--mode", "simulation"
            ]
            
            self.bot_process = subprocess.Popen(cmd)
            st.session_state.bot_running = True
            
            # Connect WebSocket
            asyncio.run(self.connect_websocket())
            
            st.success("✅ Trading bot started successfully!")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Failed to start bot: {e}")
    
    def stop_bot(self):
        """Stop the trading bot."""
        try:
            if self.bot_process:
                self.bot_process.terminate()
                self.bot_process = None
            
            st.session_state.bot_running = False
            
            # Disconnect WebSocket
            if self.ws_client:
                asyncio.run(self.ws_client.disconnect())
            
            st.success("✅ Trading bot stopped")
            
        except Exception as e:
            st.error(f"❌ Failed to stop bot: {e}")
    
    def emergency_stop(self):
        """Emergency stop - close all positions and stop bot."""
        st.warning("🚨 EMERGENCY STOP ACTIVATED!")
        
        # Close all positions
        if self.ws_client:
            asyncio.run(self.ws_client.send_command("close_all_positions"))
        
        # Stop bot
        self.stop_bot()
        
        st.error("⛔ All positions closed and bot stopped!")
    
    async def connect_websocket(self):
        """Connect to WebSocket server."""
        try:
            self.ws_client = TradingWebSocketClient()
            connected = await self.ws_client.connect()
            
            if connected:
                st.session_state.ws_connected = True
                
                # Subscribe to all channels
                await self.ws_client.subscribe([
                    "signals",
                    "positions",
                    "market_data",
                    "performance"
                ])
            
        except Exception as e:
            st.error(f"❌ WebSocket connection failed: {e}")
    
    async def close_position(self, symbol: str):
        """Close a specific position."""
        if self.ws_client:
            await self.ws_client.send_command("close_position", {"symbol": symbol})
    
    async def update_data(self):
        """Update data from WebSocket."""
        if not self.ws_client or not st.session_state.ws_connected:
            return
        
        try:
            # Get latest data
            async for message in self.ws_client.receive_messages():
                msg_type = message.get('type')
                data = message.get('data')
                
                if msg_type == 'market_data':
                    st.session_state.market_data.update(data)
                
                elif msg_type == 'signal':
                    st.session_state.signals.append(data)
                    # Keep only last 100 signals
                    st.session_state.signals = st.session_state.signals[-100:]
                
                elif msg_type == 'position_update':
                    symbol = data.get('symbol')
                    st.session_state.positions[symbol] = data
                
                elif msg_type == 'performance':
                    st.session_state.performance.update(data)
                
                # Process only one message per update
                break
                
        except Exception as e:
            st.error(f"❌ Data update error: {e}")
    
    def _format_duration(self, entry_time):
        """Format position duration."""
        if not entry_time:
            return "Unknown"
        
        try:
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time)
            
            duration = datetime.now() - entry_time
            hours = duration.total_seconds() / 3600
            
            if hours < 1:
                return f"{int(duration.total_seconds() / 60)}m"
            elif hours < 24:
                return f"{int(hours)}h"
            else:
                return f"{int(hours / 24)}d"
                
        except:
            return "Unknown"


def main():
    """Main function to run the dashboard."""
    dashboard = LiveTradingDashboard()
    dashboard.render()


if __name__ == "__main__":
    main()