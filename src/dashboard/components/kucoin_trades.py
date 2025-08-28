"""
KuCoin Recent Trades component for Streamlit dashboard.
Displays real-time trade history with KuCoin styling.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import time
from .kucoin_data_manager import get_kucoin_data_manager

class KuCoinRecentTrades:
    """KuCoin-style Recent Trades component for Streamlit."""
    
    def __init__(self):
        """Initialize Recent Trades component."""
        if 'trades_data' not in st.session_state:
            st.session_state.trades_data = {
                'trades': [],
                'last_update': None
            }
    
    def update_data(self, trades_data: List[Dict[str, Any]]):
        """Update trades data."""
        if trades_data:
            # Sort trades by timestamp (newest first)
            sorted_trades = sorted(trades_data, key=lambda x: x.get('timestamp', 0), reverse=True)
            
            st.session_state.trades_data = {
                'trades': sorted_trades,
                'last_update': datetime.now()
            }
    
    def format_price(self, price: float) -> str:
        """Format price for display."""
        if price >= 1:
            return f"{price:,.2f}"
        elif price >= 0.01:
            return f"{price:.4f}"
        else:
            return f"{price:.8f}"
    
    def format_amount(self, amount: float) -> str:
        """Format amount for display."""
        if amount >= 1000:
            return f"{amount:,.1f}"
        elif amount >= 10:
            return f"{amount:.2f}"
        else:
            return f"{amount:.4f}"
    
    def format_time(self, timestamp: int) -> str:
        """Format timestamp for display."""
        try:
            dt = datetime.fromtimestamp(timestamp / 1000)
            now = datetime.now()
            
            # If today, show time only
            if dt.date() == now.date():
                return dt.strftime("%H:%M:%S")
            # If yesterday, show "Yesterday HH:MM"
            elif dt.date() == (now - timedelta(days=1)).date():
                return f"Yesterday {dt.strftime('%H:%M')}"
            # Otherwise show date
            else:
                return dt.strftime("%m-%d %H:%M")
        except:
            return "N/A"
    
    def get_trade_volume(self, price: float, amount: float) -> float:
        """Calculate trade volume (price * amount)."""
        return price * amount
    
    def render_trades_header(self):
        """Render trades header."""
        col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 1])
        
        with col1:
            st.markdown("**Price (USDT)**", help="Trade execution price")
        with col2:
            st.markdown("**Amount**", help="Trade amount")
        with col3:
            st.markdown("**Volume**", help="Trade volume (price × amount)")
        with col4:
            st.markdown("**Time**", help="Trade execution time")
    
    def render_trade_row(self, trade: Dict[str, Any], index: int):
        """Render individual trade row using native Streamlit components."""
        try:
            price = float(trade.get('price', 0))
            amount = float(trade.get('amount', 0))
            side = trade.get('side', 'buy').lower()
            timestamp = trade.get('timestamp', int(datetime.now().timestamp() * 1000))
            
            volume = self.get_trade_volume(price, amount)
            
            # Determine colors and symbols based on side
            side_symbol = "🟢▲" if side == "buy" else "🔴▼"
            price_color = "#03A66D" if side == "buy" else "#F6465D"
            
            # Use native Streamlit columns for layout
            col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 1])
            
            with col1:
                st.markdown(
                    f'<span style="color: {price_color}; font-weight: 600;">'
                    f'{side_symbol} {self.format_price(price)}</span>',
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f'<div style="text-align: right; color: #EAECEF;">'
                    f'{self.format_amount(amount)}</div>',
                    unsafe_allow_html=True
                )
            
            with col3:
                st.markdown(
                    f'<div style="text-align: right; color: #848E9C;">'
                    f'{self.format_amount(volume)}</div>',
                    unsafe_allow_html=True
                )
            
            with col4:
                st.markdown(
                    f'<div style="text-align: right; color: #5E6673; font-size: 0.75rem;">'
                    f'{self.format_time(timestamp)}</div>',
                    unsafe_allow_html=True
                )
            
            # Add subtle separator
            if index < 19:  # Don't add separator after last row
                st.markdown(
                    '<hr style="margin: 0.2rem 0; border: 0; height: 1px; background: #2B2F36;">',
                    unsafe_allow_html=True
                )
            
        except Exception as e:
            st.error(f"Error rendering trade row: {e}")
    
    def calculate_trade_stats(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trading statistics."""
        if not trades:
            return {}
        
        try:
            buy_trades = [t for t in trades if t.get('side', '').lower() == 'buy']
            sell_trades = [t for t in trades if t.get('side', '').lower() == 'sell']
            
            total_buy_volume = sum(float(t.get('price', 0)) * float(t.get('amount', 0)) for t in buy_trades)
            total_sell_volume = sum(float(t.get('price', 0)) * float(t.get('amount', 0)) for t in sell_trades)
            total_volume = total_buy_volume + total_sell_volume
            
            avg_price = sum(float(t.get('price', 0)) for t in trades) / len(trades)
            
            return {
                'total_trades': len(trades),
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'total_volume': total_volume,
                'buy_volume': total_buy_volume,
                'sell_volume': total_sell_volume,
                'avg_price': avg_price,
                'buy_percentage': (len(buy_trades) / len(trades)) * 100 if trades else 0
            }
        except Exception:
            return {}
    
    def render_stats_summary(self, trades: List[Dict[str, Any]]):
        """Render trading statistics summary."""
        stats = self.calculate_trade_stats(trades)
        
        if not stats:
            return
        
        # Quick stats bar
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Trades",
                value=stats['total_trades'],
                help="Total number of recent trades"
            )
        
        with col2:
            buy_pct = stats['buy_percentage']
            st.metric(
                label="Buy/Sell Ratio",
                value=f"{buy_pct:.1f}%",
                delta=f"{buy_pct - 50:.1f}%" if buy_pct != 50 else None,
                help="Percentage of buy orders vs sell orders"
            )
        
        with col3:
            st.metric(
                label="Volume (USDT)",
                value=f"{stats['total_volume']:,.0f}",
                help="Total trading volume"
            )
        
        with col4:
            st.metric(
                label="Avg Price",
                value=f"{self.format_price(stats['avg_price'])}",
                help="Average trade price"
            )
    
    def render(self, symbol: str = "BTC/USDT", height: int = 400, max_trades: int = 50):
        """Render the complete recent trades component."""
        
        # Container for trades
        trades_container = st.container()
        
        with trades_container:
            # Header
            st.markdown(f"""
            <div class="orderbook-header">
                📈 Recent Trades - {symbol}
            </div>
            """, unsafe_allow_html=True)
            
            # Get data from KuCoin data manager
            data_manager = get_kucoin_data_manager()
            trades = data_manager.get_trades_data(symbol, limit=max_trades)
            last_update = datetime.now()
            
            # Show last update time
            if last_update:
                time_diff = (datetime.now() - last_update).total_seconds()
                status_color = "#03A66D" if time_diff < 10 else "#F0B90B" if time_diff < 30 else "#F6465D"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 0.5rem; font-size: 0.75rem;">
                    <span style="color: {status_color};">
                        Last update: {time_diff:.0f}s ago
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            # Show quick statistics
            if trades:
                self.render_stats_summary(trades)
                st.markdown("---")
            
            # Create container for trades with native Streamlit
            trades_container = st.container()
            
            with trades_container:
                # Header row
                self.render_trades_header()
                st.markdown("---")
                
                # Render trades
                if trades:
                    for i, trade in enumerate(trades):
                        self.render_trade_row(trade, i)
                else:
                    st.info("📈 No recent trades available")
    
    def render_compact(self, symbol: str = "BTC/USDT", max_trades: int = 10):
        """Render a compact version of recent trades."""
        
        st.markdown(f"**📈 {symbol} Recent Trades**")
        
        # Get data from KuCoin data manager
        data_manager = get_kucoin_data_manager()
        trades = data_manager.get_trades_data(symbol, limit=max_trades)
        
        if not trades:
            st.markdown("*No recent trades*")
            return
        
        # Compact display using native Streamlit
        for trade in trades:
            try:
                price = float(trade.get('price', 0))
                amount = float(trade.get('amount', 0))
                side = trade.get('side', 'buy').lower()
                timestamp = trade.get('timestamp', int(datetime.now().timestamp() * 1000))
                
                side_symbol = "🟢▲" if side == "buy" else "🔴▼"
                price_color = "#03A66D" if side == "buy" else "#F6465D"
                
                # Use columns for compact layout
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(
                        f'<span style="color: {price_color}; font-size: 0.8rem;">'
                        f'{side_symbol} {self.format_price(price)}</span>',
                        unsafe_allow_html=True
                    )
                
                with col2:
                    st.markdown(
                        f'<div style="text-align: right; color: #848E9C; font-size: 0.8rem;">'
                        f'{self.format_amount(amount)}</div>',
                        unsafe_allow_html=True
                    )
                
                with col3:
                    st.markdown(
                        f'<div style="text-align: right; color: #5E6673; font-size: 0.7rem;">'
                        f'{self.format_time(timestamp)}</div>',
                        unsafe_allow_html=True
                    )
                
            except Exception:
                continue
    
    def render_mini_ticker(self, symbol: str = "BTC/USDT", max_trades: int = 5):
        """Render a mini ticker showing recent price action."""
        
        # Get data from KuCoin data manager
        data_manager = get_kucoin_data_manager()
        trades = data_manager.get_trades_data(symbol, limit=max_trades)
        
        if not trades:
            return
        
        try:
            latest_trade = trades[0]
            latest_price = float(latest_trade.get('price', 0))
            latest_side = latest_trade.get('side', 'buy').lower()
            
            # Calculate price trend from recent trades
            if len(trades) > 1:
                older_price = float(trades[-1].get('price', latest_price))
                price_change = latest_price - older_price
                price_change_pct = (price_change / older_price) * 100 if older_price > 0 else 0
            else:
                price_change = 0
                price_change_pct = 0
            
            # Determine colors
            trend_color = "#03A66D" if price_change >= 0 else "#F6465D"
            trend_symbol = "▲" if price_change >= 0 else "▼"
            
            st.markdown(f"""
            <div class="ticker-container" style="text-align: center; padding: 1rem;">
                <div class="ticker-price" style="font-size: 1.5rem; font-weight: 700; color: {trend_color};">
                    {trend_symbol} {self.format_price(latest_price)}
                </div>
                <div style="color: {trend_color}; font-size: 0.875rem; margin-top: 0.25rem;">
                    {'+' if price_change >= 0 else ''}{self.format_price(price_change)} 
                    ({'+' if price_change_pct >= 0 else ''}{price_change_pct:.2f}%)
                </div>
                <div style="color: #5E6673; font-size: 0.75rem; margin-top: 0.5rem;">
                    Last: {latest_side.upper()} @ {self.format_time(latest_trade.get('timestamp', 0))}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception:
            pass


# Convenience function for easy usage
def render_kucoin_trades(symbol: str = "BTC/USDT", height: int = 400, compact: bool = False, mini: bool = False):
    """Render KuCoin recent trades component."""
    if 'kucoin_trades' not in st.session_state:
        st.session_state.kucoin_trades = KuCoinRecentTrades()
    
    trades = st.session_state.kucoin_trades
    
    if mini:
        trades.render_mini_ticker(symbol)
    elif compact:
        trades.render_compact(symbol)
    else:
        trades.render(symbol, height)
    
    return trades