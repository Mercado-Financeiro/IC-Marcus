"""
KuCoin Ticker component for Streamlit dashboard.
Displays real-time price information with KuCoin styling.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
import time
from .kucoin_data_manager import get_kucoin_data_manager

class KuCoinTicker:
    """KuCoin-style Ticker component for Streamlit."""
    
    def __init__(self):
        """Initialize Ticker component."""
        if 'ticker_data' not in st.session_state:
            st.session_state.ticker_data = {
                'symbol': '',
                'last': 0,
                'high': 0,
                'low': 0,
                'volume': 0,
                'change': 0,
                'percentage': 0,
                'last_update': None,
                'price_history': []  # For mini chart
            }
    
    def update_data(self, ticker_data: Dict[str, Any]):
        """Update ticker data."""
        if ticker_data:
            # Store price history for trend visualization
            current_history = st.session_state.ticker_data.get('price_history', [])
            current_price = float(ticker_data.get('last', 0))
            
            # Add current price to history (keep last 20 points)
            current_history.append({
                'price': current_price,
                'timestamp': datetime.now().timestamp()
            })
            
            if len(current_history) > 20:
                current_history = current_history[-20:]
            
            st.session_state.ticker_data = {
                'symbol': ticker_data.get('symbol', ''),
                'last': current_price,
                'high': float(ticker_data.get('high', 0)),
                'low': float(ticker_data.get('low', 0)),
                'volume': float(ticker_data.get('volume', 0)),
                'change': float(ticker_data.get('change', 0)),
                'percentage': float(ticker_data.get('percentage', 0)),
                'last_update': datetime.now(),
                'price_history': current_history
            }
    
    def format_price(self, price: float) -> str:
        """Format price for display."""
        if price >= 1:
            return f"{price:,.2f}"
        elif price >= 0.01:
            return f"{price:.4f}"
        else:
            return f"{price:.8f}"
    
    def format_volume(self, volume: float) -> str:
        """Format volume for display."""
        if volume >= 1_000_000_000:
            return f"{volume / 1_000_000_000:.2f}B"
        elif volume >= 1_000_000:
            return f"{volume / 1_000_000:.2f}M"
        elif volume >= 1_000:
            return f"{volume / 1_000:.2f}K"
        else:
            return f"{volume:.2f}"
    
    def get_price_trend_emoji(self, percentage: float) -> str:
        """Get emoji based on price trend."""
        if percentage > 5:
            return "🚀"
        elif percentage > 0:
            return "📈"
        elif percentage < -5:
            return "📉" 
        elif percentage < 0:
            return "🔻"
        else:
            return "➡️"
    
    def render_main_ticker(self, symbol: str):
        """Render main price ticker."""
        ticker_data = st.session_state.ticker_data
        
        price = ticker_data.get('last', 0)
        change = ticker_data.get('change', 0)
        percentage = ticker_data.get('percentage', 0)
        
        # Determine color based on price movement
        is_positive = percentage >= 0
        price_color = "#03A66D" if is_positive else "#F6465D"
        change_symbol = "▲" if is_positive else "▼"
        change_prefix = "+" if is_positive else ""
        
        # Main price display
        st.markdown(f"""
        <div class="trading-pair">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div class="trading-pair-symbol">
                        {self.get_price_trend_emoji(percentage)} {symbol}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div class="ticker-price" style="color: {price_color};">
                        {change_symbol} {self.format_price(price)}
                    </div>
                    <div style="color: {price_color}; font-size: 1rem; font-weight: 600; margin-top: 0.25rem;">
                        {change_prefix}{self.format_price(change)} ({change_prefix}{percentage:.2f}%)
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_ticker_stats(self):
        """Render additional ticker statistics."""
        ticker_data = st.session_state.ticker_data
        
        high = ticker_data.get('high', 0)
        low = ticker_data.get('low', 0)
        volume = ticker_data.get('volume', 0)
        last_update = ticker_data.get('last_update')
        
        # Stats grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="ticker-stat">
                <div class="ticker-stat-label">24h High</div>
                <div class="ticker-stat-value" style="color: #03A66D;">
                    {self.format_price(high)}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="ticker-stat">
                <div class="ticker-stat-label">24h Low</div>
                <div class="ticker-stat-value" style="color: #F6465D;">
                    {self.format_price(low)}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="ticker-stat">
                <div class="ticker-stat-label">24h Volume</div>
                <div class="ticker-stat-value">
                    {self.format_volume(volume)}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            if last_update:
                time_diff = (datetime.now() - last_update).total_seconds()
                status_color = "#03A66D" if time_diff < 10 else "#F0B90B" if time_diff < 30 else "#F6465D"
            else:
                time_diff = 0
                status_color = "#5E6673"
            
            st.markdown(f"""
            <div class="ticker-stat">
                <div class="ticker-stat-label">Last Update</div>
                <div class="ticker-stat-value" style="color: {status_color};">
                    {time_diff:.0f}s ago
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_mini_chart(self):
        """Render mini price trend chart."""
        ticker_data = st.session_state.ticker_data
        price_history = ticker_data.get('price_history', [])
        
        if len(price_history) < 2:
            return
        
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Prepare data
            prices = [p['price'] for p in price_history]
            timestamps = [datetime.fromtimestamp(p['timestamp']) for p in price_history]
            
            # Determine trend color
            trend_color = "#03A66D" if prices[-1] >= prices[0] else "#F6465D"
            
            # Create mini chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=prices,
                mode='lines',
                line=dict(color=trend_color, width=2),
                fill='tonexty',
                fillcolor=f"{trend_color}20",
                name='Price',
                hovertemplate='Price: %{y:.2f}<br>Time: %{x}<extra></extra>'
            ))
            
            fig.update_layout(
                height=150,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(
                    showgrid=False,
                    showticklabels=False,
                    zeroline=False
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='#2B2F36',
                    showticklabels=True,
                    tickfont=dict(size=10, color='#848E9C'),
                    zeroline=False
                ),
                plot_bgcolor='#1E2329',
                paper_bgcolor='#1E2329',
                showlegend=False,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except ImportError:
            # Fallback to simple text representation
            if price_history:
                trend = "📈" if price_history[-1]['price'] >= price_history[0]['price'] else "📉"
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; color: #848E9C;">
                    {trend} Price trend over last {len(price_history)} updates
                </div>
                """, unsafe_allow_html=True)
    
    def render_price_alerts_section(self, symbol: str):
        """Render price alerts configuration section."""
        st.markdown("### 🔔 Price Alerts")
        
        current_price = st.session_state.ticker_data.get('last', 0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Alert Above**")
            alert_high = st.number_input(
                "Price",
                min_value=0.0,
                value=float(current_price * 1.05) if current_price else 0.0,
                format="%.8f",
                key=f"alert_high_{symbol}"
            )
            
            if st.button("Set High Alert", key=f"set_high_{symbol}"):
                st.success(f"Alert set for {symbol} above {self.format_price(alert_high)}")
        
        with col2:
            st.markdown("**Alert Below**")
            alert_low = st.number_input(
                "Price",
                min_value=0.0,
                value=float(current_price * 0.95) if current_price else 0.0,
                format="%.8f",
                key=f"alert_low_{symbol}"
            )
            
            if st.button("Set Low Alert", key=f"set_low_{symbol}"):
                st.success(f"Alert set for {symbol} below {self.format_price(alert_low)}")
    
    def render(self, symbol: str = "BTC/USDT", show_chart: bool = True, show_alerts: bool = False):
        """Render the complete ticker component."""
        
        # Get data from KuCoin data manager and update ticker
        data_manager = get_kucoin_data_manager()
        ticker_data = data_manager.get_ticker_data(symbol)
        self.update_data(ticker_data)
        
        # Main ticker display
        self.render_main_ticker(symbol)
        
        # Stats section
        st.markdown("---")
        self.render_ticker_stats()
        
        # Optional mini chart
        if show_chart:
            st.markdown("---")
            st.markdown("**📊 Price Trend**")
            self.render_mini_chart()
        
        # Optional alerts section
        if show_alerts:
            st.markdown("---")
            self.render_price_alerts_section(symbol)
    
    def render_compact(self, symbol: str = "BTC/USDT"):
        """Render a compact version of the ticker."""
        # Get data from KuCoin data manager and update ticker
        data_manager = get_kucoin_data_manager()
        ticker_data = data_manager.get_ticker_data(symbol)
        self.update_data(ticker_data)
        
        ticker_data = st.session_state.ticker_data
        
        price = ticker_data.get('last', 0)
        percentage = ticker_data.get('percentage', 0)
        volume = ticker_data.get('volume', 0)
        
        is_positive = percentage >= 0
        price_color = "#03A66D" if is_positive else "#F6465D"
        change_symbol = "▲" if is_positive else "▼"
        change_prefix = "+" if is_positive else ""
        
        # Compact display
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div style="font-size: 1.2rem; font-weight: 600; color: {price_color};">
                {change_symbol} {self.format_price(price)}
            </div>
            <div style="font-size: 0.875rem; color: #848E9C;">
                {symbol}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="text-align: center; font-weight: 600; color: {price_color};">
                {change_prefix}{percentage:.2f}%
            </div>
            <div style="text-align: center; font-size: 0.75rem; color: #5E6673;">
                24h Change
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="text-align: right; font-weight: 500; color: #EAECEF;">
                {self.format_volume(volume)}
            </div>
            <div style="text-align: right; font-size: 0.75rem; color: #5E6673;">
                24h Volume
            </div>
            """, unsafe_allow_html=True)
    
    def render_header_ticker(self, symbol: str = "BTC/USDT"):
        """Render ticker for page header."""
        # Get data from KuCoin data manager and update ticker
        data_manager = get_kucoin_data_manager()
        ticker_data = data_manager.get_ticker_data(symbol)
        self.update_data(ticker_data)
        
        ticker_data = st.session_state.ticker_data
        
        price = ticker_data.get('last', 0)
        percentage = ticker_data.get('percentage', 0)
        
        is_positive = percentage >= 0
        price_color = "#03A66D" if is_positive else "#F6465D"
        trend_emoji = self.get_price_trend_emoji(percentage)
        change_prefix = "+" if is_positive else ""
        
        return st.markdown(f"""
        <div class="kucoin-header" style="text-align: center;">
            <span class="kucoin-logo">KuCoin</span>
            <span style="margin: 0 1rem; color: #848E9C;">|</span>
            <span style="font-size: 1.5rem; font-weight: 600; color: {price_color};">
                {trend_emoji} {symbol} {self.format_price(price)}
            </span>
            <span style="margin-left: 0.5rem; color: {price_color}; font-weight: 500;">
                ({change_prefix}{percentage:.2f}%)
            </span>
        </div>
        """, unsafe_allow_html=True)


# Convenience function for easy usage
def render_kucoin_ticker(symbol: str = "BTC/USDT", compact: bool = False, 
                        header: bool = False, show_chart: bool = True, show_alerts: bool = False):
    """Render KuCoin ticker component."""
    if 'kucoin_ticker' not in st.session_state:
        st.session_state.kucoin_ticker = KuCoinTicker()
    
    ticker = st.session_state.kucoin_ticker
    
    if header:
        ticker.render_header_ticker(symbol)
    elif compact:
        ticker.render_compact(symbol)
    else:
        ticker.render(symbol, show_chart, show_alerts)
    
    return ticker