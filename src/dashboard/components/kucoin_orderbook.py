"""
KuCoin OrderBook component for Streamlit dashboard.
Displays real-time bid/ask orders with KuCoin styling.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
from .kucoin_data_manager import get_kucoin_data_manager

class KuCoinOrderBook:
    """KuCoin-style OrderBook component for Streamlit."""
    
    def __init__(self):
        """Initialize OrderBook component."""
        if 'orderbook_data' not in st.session_state:
            st.session_state.orderbook_data = {
                'bids': [],
                'asks': [],
                'last_update': None
            }
    
    def update_data(self, orderbook_data: Dict[str, Any]):
        """Update orderbook data."""
        if orderbook_data and 'bids' in orderbook_data and 'asks' in orderbook_data:
            st.session_state.orderbook_data = {
                'bids': orderbook_data.get('bids', []),
                'asks': orderbook_data.get('asks', []),
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
    
    def calculate_depth_percentage(self, orders: List[List[float]], max_amount: float) -> List[float]:
        """Calculate depth percentage for visual representation."""
        if not orders or max_amount == 0:
            return []
        
        return [(order[1] / max_amount) * 100 for order in orders]
    
    def render_orderbook_header(self, symbol: str):
        """Render orderbook header."""
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.markdown("**Price (USDT)**", help="Bid/Ask prices")
        with col2:
            st.markdown("**Size**", help="Order amounts")
        with col3:
            st.markdown("**Total**", help="Cumulative amounts")
    
    def render_asks_section(self, asks: List[List[float]], max_amount: float):
        """Render asks (sell orders) section using native Streamlit."""
        if not asks:
            st.info("No ask orders")
            return
        
        # Show asks in reverse order (highest price first)
        asks_display = list(reversed(asks[:10]))
        
        cumulative_amount = 0
        
        for i, (ask_price, ask_amount) in enumerate(asks_display):
            cumulative_amount += ask_amount
            
            # Use native Streamlit columns
            col1, col2, col3 = st.columns([2, 1.5, 1.5])
            
            with col1:
                st.markdown(
                    f'<span style="color: #F6465D; font-weight: 600;">'
                    f'{self.format_price(ask_price)}</span>',
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f'<div style="text-align: right; color: #EAECEF;">'
                    f'{self.format_amount(ask_amount)}</div>',
                    unsafe_allow_html=True
                )
            
            with col3:
                st.markdown(
                    f'<div style="text-align: right; color: #848E9C;">'
                    f'{self.format_amount(cumulative_amount)}</div>',
                    unsafe_allow_html=True
                )
    
    def render_spread_section(self, bids: List[List[float]], asks: List[List[float]]):
        """Render spread information."""
        if bids and asks:
            highest_bid = bids[0][0] if bids else 0
            lowest_ask = asks[0][0] if asks else 0
            spread = lowest_ask - highest_bid
            spread_pct = (spread / lowest_ask) * 100 if lowest_ask > 0 else 0
            
            spread_html = f"""
            <div style="
                background-color: rgba(132, 142, 156, 0.1);
                border: 1px solid #2B2F36;
                border-radius: 4px;
                padding: 0.5rem;
                margin: 0.5rem 0;
                text-align: center;
                font-size: 0.875rem;
            ">
                <span style="color: #848E9C;">Spread: </span>
                <span style="color: #EAECEF; font-weight: 600;">
                    {self.format_price(spread)} ({spread_pct:.3f}%)
                </span>
            </div>
            """
            
            st.markdown(spread_html, unsafe_allow_html=True)
    
    def render_bids_section(self, bids: List[List[float]], max_amount: float):
        """Render bids (buy orders) section using native Streamlit."""
        if not bids:
            st.info("No bid orders")
            return
        
        bids_display = bids[:10]
        
        cumulative_amount = 0
        
        for i, (bid_price, bid_amount) in enumerate(bids_display):
            cumulative_amount += bid_amount
            
            # Use native Streamlit columns
            col1, col2, col3 = st.columns([2, 1.5, 1.5])
            
            with col1:
                st.markdown(
                    f'<span style="color: #03A66D; font-weight: 600;">'
                    f'{self.format_price(bid_price)}</span>',
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f'<div style="text-align: right; color: #EAECEF;">'
                    f'{self.format_amount(bid_amount)}</div>',
                    unsafe_allow_html=True
                )
            
            with col3:
                st.markdown(
                    f'<div style="text-align: right; color: #848E9C;">'
                    f'{self.format_amount(cumulative_amount)}</div>',
                    unsafe_allow_html=True
                )
    
    def render(self, symbol: str = "BTC/USDT", height: int = 400):
        """Render the complete orderbook component."""
        
        # Container for orderbook
        orderbook_container = st.container()
        
        with orderbook_container:
            # Header
            st.markdown(f"""
            <div class="orderbook-header">
                📊 Order Book - {symbol}
            </div>
            """, unsafe_allow_html=True)
            
            # Get data from KuCoin data manager
            data_manager = get_kucoin_data_manager()
            orderbook_data = data_manager.get_orderbook_data(symbol, limit=20)
            
            bids = orderbook_data.get('bids', [])
            asks = orderbook_data.get('asks', [])
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
            
            # Calculate max amount for depth visualization
            all_amounts = []
            if bids:
                all_amounts.extend([bid[1] for bid in bids])
            if asks:
                all_amounts.extend([ask[1] for ask in asks])
            
            max_amount = max(all_amounts) if all_amounts else 1
            
            # Create container for orderbook with native Streamlit
            orderbook_container = st.container()
            
            with orderbook_container:
                # Header row
                self.render_orderbook_header(symbol)
                
                # Asks section (sell orders) - top half
                st.markdown("**🔴 SELL ORDERS**", help="Ask orders (sellers)")
                self.render_asks_section(asks, max_amount)
                
                # Spread section
                self.render_spread_section(bids, asks)
                
                # Bids section (buy orders) - bottom half  
                st.markdown("**🟢 BUY ORDERS**", help="Bid orders (buyers)")
                self.render_bids_section(bids, max_amount)
            
            # Summary stats
            if bids and asks:
                total_bid_volume = sum(bid[1] for bid in bids)
                total_ask_volume = sum(ask[1] for ask in asks)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Total Bids",
                        value=f"{self.format_amount(total_bid_volume)}",
                        help="Total volume of buy orders"
                    )
                with col2:
                    st.metric(
                        label="Total Asks", 
                        value=f"{self.format_amount(total_ask_volume)}",
                        help="Total volume of sell orders"
                    )
    
    def render_compact(self, symbol: str = "BTC/USDT", max_orders: int = 5):
        """Render a compact version of the orderbook."""
        
        st.markdown(f"**📊 {symbol} Order Book**")
        
        # Get data from KuCoin data manager
        data_manager = get_kucoin_data_manager()
        orderbook_data = data_manager.get_orderbook_data(symbol, limit=max_orders)
        
        bids = orderbook_data.get('bids', [])[:max_orders]
        asks = orderbook_data.get('asks', [])[:max_orders]
        
        # Create compact display
        if asks:
            st.markdown("**Asks (Sell)**")
            for ask_price, ask_amount in reversed(asks):
                st.markdown(f"""
                <div class="orderbook-row ask-row" style="font-size: 0.8rem; padding: 0.2rem 0.5rem;">
                    <span class="ask-price">{self.format_price(ask_price)}</span>
                    <span class="order-amount" style="float: right;">{self.format_amount(ask_amount)}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Spread
        if bids and asks:
            spread = asks[0][0] - bids[0][0] 
            st.markdown(f"""
            <div style="text-align: center; padding: 0.25rem; font-size: 0.75rem; color: #848E9C;">
                Spread: {self.format_price(spread)}
            </div>
            """, unsafe_allow_html=True)
        
        if bids:
            st.markdown("**Bids (Buy)**")
            for bid_price, bid_amount in bids:
                st.markdown(f"""
                <div class="orderbook-row bid-row" style="font-size: 0.8rem; padding: 0.2rem 0.5rem;">
                    <span class="bid-price">{self.format_price(bid_price)}</span>
                    <span class="order-amount" style="float: right;">{self.format_amount(bid_amount)}</span>
                </div>
                """, unsafe_allow_html=True)


# Convenience function for easy usage
def render_kucoin_orderbook(symbol: str = "BTC/USDT", height: int = 400, compact: bool = False):
    """Render KuCoin orderbook component."""
    if 'kucoin_orderbook' not in st.session_state:
        st.session_state.kucoin_orderbook = KuCoinOrderBook()
    
    orderbook = st.session_state.kucoin_orderbook
    
    if compact:
        orderbook.render_compact(symbol)
    else:
        orderbook.render(symbol, height)
    
    return orderbook