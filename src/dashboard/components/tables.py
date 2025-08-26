"""
Interactive table components for ML Trading Dashboard.
Professional data tables with sorting, filtering, and export capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable
from datetime import datetime
import plotly.graph_objects as go

class InteractiveTables:
    """Interactive table components with advanced features."""
    
    @staticmethod
    def render_trading_table(
        df: pd.DataFrame,
        title: str = "Trading Data",
        theme: Dict = None,
        show_filters: bool = True,
        show_export: bool = True,
        row_height: int = 35,
        max_rows: int = 20
    ):
        """
        Render an interactive trading data table.
        
        Args:
            df: DataFrame to display
            title: Table title
            theme: Theme configuration
            show_filters: Show column filters
            show_export: Show export buttons
            row_height: Height of each row in pixels
            max_rows: Maximum rows to display
        """
        if df.empty:
            st.warning(f"No data available for {title}")
            return
        
        # Table header
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"### {title}")
        
        with col2:
            if show_filters:
                # Search box
                search_term = st.text_input("🔍 Search", key=f"search_{title}")
                if search_term:
                    mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
                    df = df[mask]
        
        with col3:
            if show_export:
                # Export buttons
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 CSV",
                    data=csv,
                    file_name=f"{title.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        # Column filters
        if show_filters and not df.empty:
            st.markdown("#### Filters")
            filter_cols = st.columns(min(len(df.columns), 4))
            
            filtered_df = df.copy()
            
            for idx, col in enumerate(df.columns[:4]):
                with filter_cols[idx]:
                    if df[col].dtype in ['int64', 'float64']:
                        min_val = float(df[col].min())
                        max_val = float(df[col].max())
                        range_val = st.slider(
                            f"{col}",
                            min_val,
                            max_val,
                            (min_val, max_val),
                            key=f"filter_{title}_{col}"
                        )
                        filtered_df = filtered_df[
                            (filtered_df[col] >= range_val[0]) & 
                            (filtered_df[col] <= range_val[1])
                        ]
                    else:
                        unique_vals = df[col].unique()
                        if len(unique_vals) <= 10:
                            selected = st.multiselect(
                                f"{col}",
                                unique_vals,
                                default=unique_vals,
                                key=f"filter_{title}_{col}"
                            )
                            filtered_df = filtered_df[filtered_df[col].isin(selected)]
        else:
            filtered_df = df
        
        # Pagination
        rows_per_page = st.slider(
            "Rows per page",
            min_value=5,
            max_value=50,
            value=max_rows,
            step=5,
            key=f"rows_{title}"
        )
        
        total_pages = max(1, len(filtered_df) // rows_per_page + (1 if len(filtered_df) % rows_per_page else 0))
        page_num = st.number_input(
            f"Page (1-{total_pages})",
            min_value=1,
            max_value=total_pages,
            value=1,
            key=f"page_{title}"
        )
        
        start_idx = (page_num - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, len(filtered_df))
        
        # Display table with custom styling
        display_df = filtered_df.iloc[start_idx:end_idx]
        
        # Apply conditional formatting
        styled_df = InteractiveTables.apply_table_styling(display_df, theme)
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=min(len(display_df) * row_height + 50, 600)
        )
        
        # Table footer with statistics
        st.markdown(f"Showing {start_idx + 1}-{end_idx} of {len(filtered_df)} rows")
    
    @staticmethod
    def apply_table_styling(df: pd.DataFrame, theme: Dict = None) -> pd.DataFrame:
        """Apply conditional formatting to DataFrame."""
        if theme is None:
            theme = {'success': '#10B981', 'danger': '#EF4444', 'warning': '#F59E0B'}
        
        def color_negative_red(val):
            """Color negative values red, positive green."""
            if isinstance(val, (int, float)):
                color = theme['danger'] if val < 0 else theme['success'] if val > 0 else 'inherit'
                return f'color: {color}'
            return ''
        
        def highlight_max(s):
            """Highlight maximum value in series."""
            if s.dtype in ['int64', 'float64']:
                is_max = s == s.max()
                return ['background-color: rgba(16, 185, 129, 0.2)' if v else '' for v in is_max]
            return [''] * len(s)
        
        # Apply styles based on column names
        styled = df.style
        
        # Color P&L columns
        pnl_cols = [col for col in df.columns if 'pnl' in col.lower() or 'return' in col.lower()]
        if pnl_cols:
            styled = styled.applymap(color_negative_red, subset=pnl_cols)
        
        # Highlight max values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            styled = styled.apply(highlight_max, subset=numeric_cols)
        
        return styled
    
    @staticmethod
    def render_order_book(
        bids: pd.DataFrame,
        asks: pd.DataFrame,
        theme: Dict,
        depth: int = 20
    ):
        """
        Render order book visualization.
        
        Args:
            bids: DataFrame with bid orders (price, size, total)
            asks: DataFrame with ask orders (price, size, total)
            theme: Theme configuration
            depth: Number of levels to show
        """
        st.markdown("### 📊 Order Book")
        
        # Create order book visualization
        fig = go.Figure()
        
        # Add bid bars
        if not bids.empty:
            fig.add_trace(go.Bar(
                x=bids['size'][:depth],
                y=bids['price'][:depth],
                orientation='h',
                name='Bids',
                marker_color=theme['success'],
                text=[f"${p:,.2f}" for p in bids['price'][:depth]],
                textposition='inside',
                hovertemplate='Price: $%{y:,.2f}<br>Size: %{x:,.4f}<br>Total: %{customdata:,.4f}',
                customdata=bids['total'][:depth]
            ))
        
        # Add ask bars
        if not asks.empty:
            fig.add_trace(go.Bar(
                x=-asks['size'][:depth],
                y=asks['price'][:depth],
                orientation='h',
                name='Asks',
                marker_color=theme['danger'],
                text=[f"${p:,.2f}" for p in asks['price'][:depth]],
                textposition='inside',
                hovertemplate='Price: $%{y:,.2f}<br>Size: %{x:,.4f}<br>Total: %{customdata:,.4f}',
                customdata=asks['total'][:depth]
            ))
        
        # Update layout
        fig.update_layout(
            height=400,
            paper_bgcolor=theme['chart_bg'],
            plot_bgcolor=theme['chart_bg'],
            font={'color': theme['text_primary']},
            xaxis={
                'title': 'Size',
                'gridcolor': theme['chart_grid'],
                'zeroline': True,
                'zerolinecolor': theme['border'],
                'zerolinewidth': 2
            },
            yaxis={
                'title': 'Price',
                'gridcolor': theme['chart_grid']
            },
            barmode='overlay',
            bargap=0.1,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Order book imbalance
        if not bids.empty and not asks.empty:
            total_bid_vol = bids['size'][:depth].sum()
            total_ask_vol = asks['size'][:depth].sum()
            imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Bid Volume", f"{total_bid_vol:,.2f}")
            
            with col2:
                st.metric("Ask Volume", f"{total_ask_vol:,.2f}")
            
            with col3:
                color = theme['success'] if imbalance > 0 else theme['danger']
                st.metric(
                    "Imbalance",
                    f"{imbalance:.2%}",
                    delta="Bullish" if imbalance > 0 else "Bearish"
                )
    
    @staticmethod
    def render_trade_history(
        trades: pd.DataFrame,
        theme: Dict,
        show_chart: bool = True
    ):
        """
        Render trade history table with optional chart.
        
        Args:
            trades: DataFrame with trade history
            theme: Theme configuration
            show_chart: Whether to show trade chart
        """
        if trades.empty:
            st.info("No trade history available")
            return
        
        st.markdown("### 📜 Trade History")
        
        # Summary metrics
        total_trades = len(trades)
        winning_trades = len(trades[trades['pnl'] > 0])
        losing_trades = len(trades[trades['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", total_trades)
        
        with col2:
            st.metric("Win Rate", f"{win_rate:.1%}")
        
        with col3:
            st.metric("Winners", winning_trades)
        
        with col4:
            st.metric("Losers", losing_trades)
        
        # Trade chart
        if show_chart and 'timestamp' in trades.columns:
            fig = go.Figure()
            
            # Cumulative P&L
            trades['cumulative_pnl'] = trades['pnl'].cumsum()
            
            fig.add_trace(go.Scatter(
                x=trades['timestamp'],
                y=trades['cumulative_pnl'],
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color=theme['primary'], width=2),
                marker=dict(
                    size=8,
                    color=['green' if p > 0 else 'red' for p in trades['pnl']],
                    line=dict(width=1, color='white')
                ),
                hovertemplate='Time: %{x}<br>Cum P&L: $%{y:,.2f}<br>Trade P&L: %{customdata:,.2f}',
                customdata=trades['pnl']
            ))
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color=theme['text_muted'], opacity=0.5)
            
            fig.update_layout(
                height=300,
                paper_bgcolor=theme['chart_bg'],
                plot_bgcolor=theme['chart_bg'],
                font={'color': theme['text_primary']},
                xaxis={'gridcolor': theme['chart_grid']},
                yaxis={'gridcolor': theme['chart_grid'], 'title': 'Cumulative P&L ($)'},
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Trade table
        display_cols = ['timestamp', 'symbol', 'side', 'entry_price', 'exit_price', 
                       'quantity', 'pnl', 'pnl_pct', 'duration']
        
        available_cols = [col for col in display_cols if col in trades.columns]
        display_df = trades[available_cols].copy()
        
        # Format columns
        if 'pnl' in display_df.columns:
            display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:,.2f}")
        if 'pnl_pct' in display_df.columns:
            display_df['pnl_pct'] = display_df['pnl_pct'].apply(lambda x: f"{x:.2%}")
        if 'entry_price' in display_df.columns:
            display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:,.2f}")
        if 'exit_price' in display_df.columns:
            display_df['exit_price'] = display_df['exit_price'].apply(lambda x: f"${x:,.2f}")
        
        InteractiveTables.render_trading_table(
            display_df,
            title="",
            theme=theme,
            show_filters=False,
            max_rows=10
        )