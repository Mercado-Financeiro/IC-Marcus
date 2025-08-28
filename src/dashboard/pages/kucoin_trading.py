"""
KuCoin Trading page for ML Trading Dashboard.
Real-time trading interface inspired by KuCoin.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Any
import asyncio
import json

# Import custom components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.dashboard.theme import DashboardTheme
from src.dashboard.components.charts import TradingCharts
from src.dashboard.components.kucoin_orderbook import render_kucoin_orderbook
from src.dashboard.components.kucoin_trades import render_kucoin_trades
from src.dashboard.components.kucoin_ticker import render_kucoin_ticker
from src.dashboard.components.kucoin_data_manager import get_kucoin_data_manager

class KuCoinTradingPage:
    """KuCoin Trading page manager."""
    
    def __init__(self):
        """Initialize KuCoin trading page."""
        self.available_symbols = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "XRP/USDT",
            "SOL/USDT", "DOT/USDT", "AVAX/USDT", "MATIC/USDT", "LINK/USDT",
            "UNI/USDT", "LTC/USDT", "BCH/USDT", "ETC/USDT", "ATOM/USDT"
        ]
        
        # Initialize session state
        if 'kucoin_selected_symbol' not in st.session_state:
            st.session_state.kucoin_selected_symbol = "BTC/USDT"
        
        if 'kucoin_auto_refresh' not in st.session_state:
            st.session_state.kucoin_auto_refresh = True
        
        if 'kucoin_refresh_interval' not in st.session_state:
            st.session_state.kucoin_refresh_interval = 2
    
    def render_header(self):
        """Render page header with symbol selection."""
        
        # Header ticker
        render_kucoin_ticker(
            symbol=st.session_state.kucoin_selected_symbol,
            header=True
        )
        
        # Controls row
        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
        
        with col1:
            # Symbol selector
            selected_symbol = st.selectbox(
                "Trading Pair",
                options=self.available_symbols,
                index=self.available_symbols.index(st.session_state.kucoin_selected_symbol),
                key="symbol_selector"
            )
            
            if selected_symbol != st.session_state.kucoin_selected_symbol:
                st.session_state.kucoin_selected_symbol = selected_symbol
                st.rerun()
        
        with col2:
            # Auto refresh toggle
            auto_refresh = st.checkbox(
                "Auto Refresh",
                value=st.session_state.kucoin_auto_refresh,
                help="Automatically refresh data"
            )
            
            if auto_refresh != st.session_state.kucoin_auto_refresh:
                st.session_state.kucoin_auto_refresh = auto_refresh
        
        with col3:
            # Refresh interval
            if st.session_state.kucoin_auto_refresh:
                interval = st.selectbox(
                    "Interval (s)",
                    options=[1, 2, 5, 10],
                    index=[1, 2, 5, 10].index(st.session_state.kucoin_refresh_interval),
                    key="refresh_interval"
                )
                st.session_state.kucoin_refresh_interval = interval
        
        with col4:
            # Manual refresh button
            if st.button("🔄 Refresh", help="Manually refresh data"):
                self.refresh_data()
        
        with col5:
            # Connection status (check if data manager is connected)
            data_manager = get_kucoin_data_manager()
            is_connected = data_manager.is_connected()
            if is_connected:
                st.markdown('<span class="status-connected">🟢 LIVE</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-disconnected">🟡 MOCK</span>', unsafe_allow_html=True)
    
    def refresh_data(self):
        """Refresh all data for current symbol."""
        # This would trigger data refresh from WebSocket
        # For now, we'll simulate with a placeholder
        st.session_state['last_refresh'] = datetime.now()
    
    def render_main_chart(self, symbol: str):
        """Render main trading chart."""
        try:
            # Get theme for chart styling
            theme = st.session_state.get('theme', 'kucoin')
            theme_config = DashboardTheme.THEMES[theme]
            
            # Get real chart data from KuCoin data manager
            data_manager = get_kucoin_data_manager()
            timeframe = st.session_state.get('timeframe', '1h')
            chart_data = data_manager.get_chart_data(symbol, timeframe=timeframe, limit=100)
            
            # Convert to DataFrame
            df_data = []
            for item in chart_data:
                df_data.append({
                    'datetime': datetime.fromtimestamp(item['timestamp'] / 1000),
                    'open': item['open'],
                    'high': item['high'], 
                    'low': item['low'],
                    'close': item['close'],
                    'volume': item['volume']
                })
            
            df = pd.DataFrame(df_data)
            
            # Create main chart with volume
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3],
                subplot_titles=(f'{symbol} Price Chart', 'Volume')
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df['datetime'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    increasing_line_color=theme_config['candle_up'],
                    decreasing_line_color=theme_config['candle_down'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Volume bars
            colors = ['rgba(3, 166, 109, 0.7)' if close >= open else 'rgba(246, 70, 93, 0.7)' 
                     for close, open in zip(df['close'], df['open'])]
            
            fig.add_trace(
                go.Bar(
                    x=df['datetime'],
                    y=df['volume'],
                    marker_color=colors,
                    name='Volume',
                    yaxis='y2'
                ),
                row=2, col=1
            )
            
            # Update layout with KuCoin theme
            fig.update_layout(
                title=f"{symbol} - Real-time Trading Chart",
                template="plotly_dark",
                paper_bgcolor=theme_config['chart_bg'],
                plot_bgcolor=theme_config['chart_bg'],
                font=dict(color=theme_config['text_primary'], size=12),
                showlegend=False,
                height=600,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            
            fig.update_xaxes(
                gridcolor=theme_config['chart_grid'],
                showgrid=True
            )
            fig.update_yaxes(
                gridcolor=theme_config['chart_grid'],
                showgrid=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error rendering chart: {e}")
            # Fallback simple chart
            st.markdown(f"📊 **{symbol} Chart** (Chart loading...)")
    
    def render_trading_controls(self, symbol: str):
        """Render trading control panel."""
        st.markdown("### 📋 Quick Trade")
        
        # Trading tabs
        buy_tab, sell_tab = st.tabs(["🟢 BUY", "🔴 SELL"])
        
        with buy_tab:
            self.render_buy_panel(symbol)
        
        with sell_tab:
            self.render_sell_panel(symbol)
    
    def render_buy_panel(self, symbol: str):
        """Render buy order panel."""
        col1, col2 = st.columns(2)
        
        with col1:
            order_type = st.selectbox(
                "Order Type",
                ["Market", "Limit", "Stop-Limit"],
                key="buy_order_type"
            )
        
        with col2:
            if order_type != "Market":
                # Get current market price as default
                data_manager = get_kucoin_data_manager()
                ticker_data = data_manager.get_ticker_data(symbol)
                market_price = float(ticker_data.get('last', 50000.0 if 'BTC' in symbol else 3000.0))
                
                price = st.number_input(
                    "Price",
                    min_value=0.0,
                    value=market_price,
                    format="%.8f",
                    key="buy_price"
                )
        
        amount = st.number_input(
            "Amount",
            min_value=0.0,
            value=0.001,
            format="%.8f",
            key="buy_amount"
        )
        
        if order_type != "Market":
            total = amount * price
            st.markdown(f"**Total: {total:.2f} USDT**")
        
        if st.button("🟢 Place Buy Order", use_container_width=True, type="primary"):
            st.success(f"Buy order placed for {amount} {symbol.split('/')[0]}")
    
    def render_sell_panel(self, symbol: str):
        """Render sell order panel."""
        col1, col2 = st.columns(2)
        
        with col1:
            order_type = st.selectbox(
                "Order Type",
                ["Market", "Limit", "Stop-Limit"],
                key="sell_order_type"
            )
        
        with col2:
            if order_type != "Market":
                # Get current market price as default
                data_manager = get_kucoin_data_manager()
                ticker_data = data_manager.get_ticker_data(symbol)
                market_price = float(ticker_data.get('last', 50000.0 if 'BTC' in symbol else 3000.0))
                
                price = st.number_input(
                    "Price",
                    min_value=0.0,
                    value=market_price,
                    format="%.8f",
                    key="sell_price"
                )
        
        amount = st.number_input(
            "Amount",
            min_value=0.0,
            value=0.001,
            format="%.8f",
            key="sell_amount"
        )
        
        if order_type != "Market":
            total = amount * price
            st.markdown(f"**Total: {total:.2f} USDT**")
        
        if st.button("🔴 Place Sell Order", use_container_width=True):
            st.success(f"Sell order placed for {amount} {symbol.split('/')[0]}")
    
    def render_portfolio_summary(self):
        """Render portfolio summary."""
        st.markdown("### 💼 Portfolio Summary")
        
        # Mock portfolio data
        portfolio_data = {
            'BTC': {'amount': 0.5, 'value': 25000, 'change_24h': 2.5},
            'ETH': {'amount': 5.0, 'value': 15000, 'change_24h': -1.2},
            'USDT': {'amount': 10000, 'value': 10000, 'change_24h': 0.0},
        }
        
        total_value = sum(asset['value'] for asset in portfolio_data.values())
        total_change = sum(asset['value'] * asset['change_24h'] / 100 for asset in portfolio_data.values())
        total_change_pct = (total_change / total_value) * 100
        
        # Total portfolio value
        change_color = "#03A66D" if total_change >= 0 else "#F6465D"
        change_prefix = "+" if total_change >= 0 else ""
        
        st.markdown(f"""
        <div class="ticker-container" style="text-align: center; margin-bottom: 1rem;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #EAECEF;">
                Total Portfolio Value
            </div>
            <div style="font-size: 2rem; font-weight: 700; color: #EAECEF; margin: 0.5rem 0;">
                ${total_value:,.2f}
            </div>
            <div style="color: {change_color}; font-size: 1rem; font-weight: 600;">
                {change_prefix}${total_change:.2f} ({change_prefix}{total_change_pct:.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Asset breakdown
        for asset, data in portfolio_data.items():
            col1, col2, col3, col4 = st.columns(4)
            
            asset_change_color = "#03A66D" if data['change_24h'] >= 0 else "#F6465D"
            asset_change_prefix = "+" if data['change_24h'] >= 0 else ""
            
            with col1:
                st.markdown(f"**{asset}**")
            with col2:
                st.markdown(f"{data['amount']:.4f}")
            with col3:
                st.markdown(f"${data['value']:,.2f}")
            with col4:
                st.markdown(f'<span style="color: {asset_change_color};">{asset_change_prefix}{data["change_24h"]:.2f}%</span>', 
                           unsafe_allow_html=True)
    
    def render_market_overview(self):
        """Render market overview section."""
        st.markdown("### 🌐 Market Overview")
        
        # Top gainers/losers
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🚀 Top Gainers (24h)**")
            gainers = [
                ("SOL/USDT", 15.2),
                ("AVAX/USDT", 8.7),
                ("MATIC/USDT", 6.4)
            ]
            
            for symbol, change in gainers:
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 0.25rem 0;">
                    <span>{symbol}</span>
                    <span style="color: #03A66D; font-weight: 600;">+{change:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**📉 Top Losers (24h)**")
            losers = [
                ("ATOM/USDT", -4.2),
                ("LTC/USDT", -3.8),
                ("BCH/USDT", -2.1)
            ]
            
            for symbol, change in losers:
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 0.25rem 0;">
                    <span>{symbol}</span>
                    <span style="color: #F6465D; font-weight: 600;">{change:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
    
    def render(self):
        """Render the complete KuCoin trading page."""
        
        # Page header
        self.render_header()
        
        st.markdown("---")
        
        # Main content layout - KuCoin style grid
        col1, col2 = st.columns([7, 3])  # 70:30 split like KuCoin
        
        with col1:
            # Main trading chart
            self.render_main_chart(st.session_state.kucoin_selected_symbol)
            
            # Chart controls/indicators (placeholder)
            st.markdown("**📊 Chart Tools**")
            
            chart_col1, chart_col2, chart_col3, chart_col4 = st.columns(4)
            with chart_col1:
                st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3, key="timeframe")
            with chart_col2:
                st.selectbox("Chart Type", ["Candlestick", "Line", "Area"], key="chart_type")
            with chart_col3:
                st.multiselect("Indicators", ["SMA", "EMA", "RSI", "MACD", "BB"], key="indicators")
            with chart_col4:
                st.checkbox("Volume", value=True, key="show_volume")
        
        with col2:
            # Right sidebar - Order book, trades, and controls
            
            # Ticker summary (compact)
            render_kucoin_ticker(
                symbol=st.session_state.kucoin_selected_symbol,
                compact=True
            )
            
            st.markdown("---")
            
            # Order Book
            orderbook = render_kucoin_orderbook(
                symbol=st.session_state.kucoin_selected_symbol,
                height=300
            )
            
            st.markdown("---")
            
            # Recent Trades
            trades = render_kucoin_trades(
                symbol=st.session_state.kucoin_selected_symbol,
                height=300
            )
            
            st.markdown("---")
            
            # Trading controls
            self.render_trading_controls(st.session_state.kucoin_selected_symbol)
        
        # Bottom section - Portfolio and Market Overview
        st.markdown("---")
        
        bottom_col1, bottom_col2 = st.columns(2)
        
        with bottom_col1:
            self.render_portfolio_summary()
        
        with bottom_col2:
            self.render_market_overview()
        
        # Auto-refresh mechanism
        if st.session_state.kucoin_auto_refresh:
            time.sleep(st.session_state.kucoin_refresh_interval)
            st.rerun()


def render_kucoin_trading_page():
    """Render KuCoin trading page."""
    if 'kucoin_trading_page' not in st.session_state:
        st.session_state.kucoin_trading_page = KuCoinTradingPage()
    
    page = st.session_state.kucoin_trading_page
    page.render()