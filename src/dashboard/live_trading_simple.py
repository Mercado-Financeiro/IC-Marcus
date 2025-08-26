"""
Simplified Live Trading Dashboard
Focused on data visualization with mock data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="Live Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .stMetric {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 5px;
    }
    .profit { color: #00ff00; }
    .loss { color: #ff0000; }
</style>
""", unsafe_allow_html=True)


def generate_candlestick_data(symbol='BTC/USDT', periods=200):
    """Generate realistic candlestick data."""
    
    # Base prices for different symbols
    base_prices = {
        'BTC/USDT': 65000,
        'ETH/USDT': 3500,
        'BNB/USDT': 600,
        'SOL/USDT': 150,
        'ADA/USDT': 0.45
    }
    
    base_price = base_prices.get(symbol, 100)
    
    # Generate timestamps
    end_time = datetime.now()
    timestamps = pd.date_range(end=end_time, periods=periods, freq='1h')
    
    # Generate realistic OHLCV data
    data = []
    current_price = base_price
    
    for i, ts in enumerate(timestamps):
        # Random walk with trend
        trend = np.sin(i / 20) * 0.02  # Sinusoidal trend
        volatility = 0.005 + abs(np.sin(i / 10)) * 0.01  # Variable volatility
        
        change = np.random.randn() * volatility + trend * 0.1
        current_price = current_price * (1 + change)
        
        # Generate OHLC
        open_price = current_price
        close_price = current_price * (1 + np.random.randn() * volatility)
        high_price = max(open_price, close_price) * (1 + abs(np.random.randn() * volatility * 0.5))
        low_price = min(open_price, close_price) * (1 - abs(np.random.randn() * volatility * 0.5))
        
        volume = np.random.lognormal(10, 1) * (1 + abs(change) * 10)
        
        data.append({
            'timestamp': ts,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': round(volume, 2)
        })
        
        current_price = close_price
    
    return pd.DataFrame(data)


def calculate_indicators(df):
    """Calculate technical indicators."""
    # Simple Moving Averages
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    df['BB_Middle'] = df['close'].rolling(window=bb_period).mean()
    bb_std_dev = df['close'].rolling(window=bb_period).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std_dev * bb_std)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std_dev * bb_std)
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volume Moving Average
    df['Volume_MA'] = df['volume'].rolling(window=20).mean()
    
    return df


def create_candlestick_chart(df, symbol):
    """Create interactive candlestick chart with indicators."""
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{symbol} Price', 'Volume', 'RSI')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#00ff00',
            decreasing_line_color='#ff0000'
        ),
        row=1, col=1
    )
    
    # Add SMA lines
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['SMA_20'],
            name='SMA 20',
            line=dict(color='#ffff00', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['SMA_50'],
            name='SMA 50',
            line=dict(color='#00ffff', width=1)
        ),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['BB_Upper'],
            name='BB Upper',
            line=dict(color='rgba(250, 128, 114, 0.3)', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['BB_Lower'],
            name='BB Lower',
            line=dict(color='rgba(250, 128, 114, 0.3)', width=1),
            fill='tonexty',
            fillcolor='rgba(250, 128, 114, 0.1)'
        ),
        row=1, col=1
    )
    
    # Volume bars
    colors = ['#00ff00' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ff0000' 
              for i in range(len(df))]
    
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # Volume MA
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['Volume_MA'],
            name='Volume MA',
            line=dict(color='#ffff00', width=1)
        ),
        row=2, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['RSI'],
            name='RSI',
            line=dict(color='#ff00ff', width=2)
        ),
        row=3, col=1
    )
    
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        title=f"{symbol} - Live Trading Chart",
        yaxis_title="Price (USDT)",
        yaxis2_title="Volume",
        yaxis3_title="RSI"
    )
    
    fig.update_xaxes(title_text="Time", row=3, col=1)
    
    return fig


def generate_market_overview_data():
    """Generate market overview data."""
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']
    
    market_data = []
    for symbol in symbols:
        base_prices = {
            'BTC/USDT': 65000,
            'ETH/USDT': 3500,
            'BNB/USDT': 600,
            'SOL/USDT': 150,
            'ADA/USDT': 0.45
        }
        
        price = base_prices[symbol]
        change_24h = np.random.uniform(-5, 5)
        volume_24h = np.random.uniform(1000000, 10000000)
        
        market_data.append({
            'Symbol': symbol,
            'Price': f"${price:,.2f}",
            '24h Change': f"{change_24h:+.2f}%",
            '24h Volume': f"${volume_24h:,.0f}",
            'Market Cap': f"${price * volume_24h:,.0f}"
        })
    
    return pd.DataFrame(market_data)


def generate_positions():
    """Generate mock positions."""
    positions = [
        {
            'Symbol': 'BTC/USDT',
            'Side': 'LONG',
            'Entry': '$64,500',
            'Current': '$65,000',
            'Size': '0.15 BTC',
            'P&L': '+$75.00',
            'P&L %': '+0.78%',
            'Duration': '2h 15m'
        },
        {
            'Symbol': 'ETH/USDT',
            'Side': 'SHORT',
            'Entry': '$3,520',
            'Current': '$3,500',
            'Size': '2.5 ETH',
            'P&L': '+$50.00',
            'P&L %': '+0.57%',
            'Duration': '45m'
        },
        {
            'Symbol': 'BNB/USDT',
            'Side': 'LONG',
            'Entry': '$595',
            'Current': '$600',
            'Size': '5 BNB',
            'P&L': '+$25.00',
            'P&L %': '+0.84%',
            'Duration': '1h 30m'
        }
    ]
    
    return pd.DataFrame(positions)


def generate_signals():
    """Generate trading signals."""
    signals = []
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']
    
    for symbol in symbols:
        confidence = np.random.uniform(0.6, 0.95)
        signal_type = np.random.choice(['BUY', 'SELL', 'HOLD'])
        
        signals.append({
            'Time': datetime.now().strftime('%H:%M:%S'),
            'Symbol': symbol,
            'Signal': signal_type,
            'Confidence': f"{confidence:.1%}",
            'Price': f"${np.random.uniform(100, 65000):.2f}",
            'Reason': np.random.choice(['RSI Oversold', 'MA Crossover', 'BB Breakout', 'Volume Spike'])
        })
    
    return pd.DataFrame(signals)


def main():
    """Main dashboard function."""
    
    # Title and header
    st.title("📈 Live Trading Dashboard")
    st.markdown("Real-time cryptocurrency trading with ML predictions")
    
    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Portfolio Value", "$10,250", "+2.5%")
    with col2:
        st.metric("Today's P&L", "+$150", "+1.5%")
    with col3:
        st.metric("Open Positions", "3", "")
    with col4:
        st.metric("Win Rate", "58%", "+3%")
    with col5:
        st.metric("Bot Status", "🟢 Active", "")
    
    st.markdown("---")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Live Chart", "💹 Market Overview", "💼 Positions", "🔔 Signals", "📈 Performance"
    ])
    
    with tab1:
        st.subheader("Live Price Chart")
        
        # Symbol selector
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            symbol = st.selectbox(
                "Select Symbol",
                ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]
            )
        with col2:
            timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"])
        with col3:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        # Generate and display chart
        with st.spinner("Loading chart data..."):
            df = generate_candlestick_data(symbol)
            df = calculate_indicators(df)
            fig = create_candlestick_chart(df, symbol)
            st.plotly_chart(fig, use_container_width=True)
        
        # Latest price info
        latest = df.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Last Price", f"${latest['close']:,.2f}")
        with col2:
            change = ((latest['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close']) * 100
            st.metric("Change", f"{change:+.2f}%")
        with col3:
            st.metric("Volume", f"{latest['volume']:,.0f}")
        with col4:
            st.metric("RSI", f"{latest['RSI']:.1f}")
    
    with tab2:
        st.subheader("Market Overview")
        
        # Market data table
        market_df = generate_market_overview_data()
        st.dataframe(
            market_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "24h Change": st.column_config.TextColumn(
                    "24h Change",
                    help="24 hour price change"
                ),
            }
        )
        
        # Mini charts
        st.subheader("Price Trends")
        cols = st.columns(3)
        
        for i, symbol in enumerate(['BTC/USDT', 'ETH/USDT', 'BNB/USDT']):
            with cols[i]:
                df_mini = generate_candlestick_data(symbol, periods=50)
                
                fig_mini = go.Figure()
                fig_mini.add_trace(go.Scatter(
                    x=df_mini['timestamp'],
                    y=df_mini['close'],
                    mode='lines',
                    name=symbol,
                    line=dict(color='#00ff00' if df_mini['close'].iloc[-1] > df_mini['close'].iloc[0] else '#ff0000', width=2)
                ))
                
                fig_mini.update_layout(
                    title=symbol,
                    template="plotly_dark",
                    height=200,
                    showlegend=False,
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                
                st.plotly_chart(fig_mini, use_container_width=True)
    
    with tab3:
        st.subheader("Open Positions")
        
        positions_df = generate_positions()
        
        # Display positions with color coding
        for _, pos in positions_df.iterrows():
            color = "green" if pos['P&L'].startswith('+') else "red"
            with st.container():
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.write(f"**{pos['Symbol']}**")
                with col2:
                    st.write(f"{'🟢' if pos['Side'] == 'LONG' else '🔴'} {pos['Side']}")
                with col3:
                    st.write(f"Entry: {pos['Entry']}")
                with col4:
                    st.write(f"Current: {pos['Current']}")
                with col5:
                    st.markdown(f"<span style='color: {color}'>{pos['P&L']} ({pos['P&L %']})</span>", unsafe_allow_html=True)
                with col6:
                    if st.button(f"Close", key=f"close_{pos['Symbol']}"):
                        st.info(f"Closing position for {pos['Symbol']}...")
                st.markdown("---")
        
        # Position summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Positions", len(positions_df))
        with col2:
            st.metric("Total P&L", "+$150.00", "+1.5%")
        with col3:
            st.metric("Average Duration", "1h 30m")
    
    with tab4:
        st.subheader("Trading Signals")
        
        signals_df = generate_signals()
        
        # Display signals with color coding
        for _, signal in signals_df.iterrows():
            color = "#00ff00" if signal['Signal'] == 'BUY' else "#ff0000" if signal['Signal'] == 'SELL' else "#ffff00"
            icon = "🟢" if signal['Signal'] == 'BUY' else "🔴" if signal['Signal'] == 'SELL' else "🟡"
            
            with st.container():
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.write(signal['Time'])
                with col2:
                    st.write(f"**{signal['Symbol']}**")
                with col3:
                    st.markdown(f"<span style='color: {color}'>{icon} {signal['Signal']}</span>", unsafe_allow_html=True)
                with col4:
                    st.write(signal['Confidence'])
                with col5:
                    st.write(signal['Price'])
                with col6:
                    st.write(signal['Reason'])
                st.markdown("---")
        
        # Signal statistics
        st.subheader("Signal Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Signals Today", len(signals_df))
        with col2:
            st.metric("Accuracy", "72%")
        with col3:
            st.metric("Avg Confidence", "78%")
    
    with tab5:
        st.subheader("Performance Analytics")
        
        # Performance chart
        days = 30
        timestamps = pd.date_range(end=datetime.now(), periods=days, freq='1d')
        balances = [10000]
        
        for i in range(1, days):
            change = np.random.randn() * 100
            balances.append(balances[-1] + change)
        
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(
            x=timestamps,
            y=balances,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#00ff00', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)'
        ))
        
        fig_perf.update_layout(
            title="Portfolio Performance (30 Days)",
            template="plotly_dark",
            height=400,
            xaxis_title="Date",
            yaxis_title="Value (USD)"
        )
        
        st.plotly_chart(fig_perf, use_container_width=True)
        
        # Performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Trading Metrics")
            metrics = {
                "Total Return": "+$250.00 (+2.5%)",
                "Sharpe Ratio": "1.85",
                "Max Drawdown": "-4.5%",
                "Win Rate": "58%",
                "Profit Factor": "1.92"
            }
            for key, value in metrics.items():
                st.write(f"**{key}:** {value}")
        
        with col2:
            st.markdown("### Statistics")
            stats = {
                "Total Trades": "145",
                "Winning Trades": "84",
                "Losing Trades": "61",
                "Average Win": "$45.50",
                "Average Loss": "$-25.25"
            }
            for key, value in stats.items():
                st.write(f"**{key}:** {value}")
    
    # Auto-refresh
    time.sleep(5)
    st.rerun()


if __name__ == "__main__":
    main()