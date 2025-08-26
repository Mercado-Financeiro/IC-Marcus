"""
Advanced chart components for ML Trading Dashboard.
Professional trading charts with technical indicators.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class TradingCharts:
    """Advanced trading chart components."""
    
    @staticmethod
    def create_candlestick_chart(
        df: pd.DataFrame,
        theme: Dict,
        indicators: Optional[List[str]] = None,
        height: int = 600
    ) -> go.Figure:
        """
        Create professional candlestick chart with volume and indicators.
        
        Args:
            df: DataFrame with OHLCV data
            theme: Theme configuration dict
            indicators: List of indicators to plot ['sma', 'ema', 'bollinger', 'rsi', 'macd']
            height: Chart height in pixels
        """
        indicators = indicators or []
        
        # Calculate number of subplots needed
        n_subplots = 2  # Main chart + volume
        if 'rsi' in indicators:
            n_subplots += 1
        if 'macd' in indicators:
            n_subplots += 1
        
        # Create subplot heights
        heights = [0.6, 0.2]  # Main chart and volume
        if 'rsi' in indicators:
            heights.append(0.1)
        if 'macd' in indicators:
            heights.append(0.1)
        
        # Normalize heights
        total_height = sum(heights)
        heights = [h/total_height for h in heights]
        
        # Create subplots
        subplot_titles = ['Price', 'Volume']
        if 'rsi' in indicators:
            subplot_titles.append('RSI')
        if 'macd' in indicators:
            subplot_titles.append('MACD')
        
        fig = make_subplots(
            rows=n_subplots,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=subplot_titles,
            row_heights=heights
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color=theme['candle_up'],
                decreasing_line_color=theme['candle_down'],
                increasing_fillcolor=theme['candle_up'],
                decreasing_fillcolor=theme['candle_down']
            ),
            row=1, col=1
        )
        
        # Add SMA if requested
        if 'sma' in indicators and 'sma_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['sma_20'],
                    name='SMA 20',
                    line=dict(color=theme['primary'], width=1),
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            if 'sma_50' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['sma_50'],
                        name='SMA 50',
                        line=dict(color=theme['secondary'], width=1),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
        
        # Add EMA if requested
        if 'ema' in indicators and 'ema_12' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['ema_12'],
                    name='EMA 12',
                    line=dict(color=theme['info'], width=1, dash='dot'),
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            if 'ema_26' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['ema_26'],
                        name='EMA 26',
                        line=dict(color=theme['warning'], width=1, dash='dot'),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
        
        # Add Bollinger Bands if requested
        if 'bollinger' in indicators:
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['bb_upper'],
                        name='BB Upper',
                        line=dict(color=theme['text_muted'], width=1),
                        opacity=0.3
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['bb_lower'],
                        name='BB Lower',
                        line=dict(color=theme['text_muted'], width=1),
                        fill='tonexty',
                        fillcolor=f"rgba(128, 128, 128, 0.1)",
                        opacity=0.3
                    ),
                    row=1, col=1
                )
        
        # Volume bars
        colors = [theme['candle_up'] if close >= open_ else theme['candle_down'] 
                  for close, open_ in zip(df['close'], df['open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.5
            ),
            row=2, col=1
        )
        
        # RSI if requested
        current_row = 3
        if 'rsi' in indicators and 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['rsi'],
                    name='RSI',
                    line=dict(color=theme['primary'], width=1.5)
                ),
                row=current_row, col=1
            )
            
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color=theme['danger'], 
                         opacity=0.3, row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color=theme['success'], 
                         opacity=0.3, row=current_row, col=1)
            current_row += 1
        
        # MACD if requested
        if 'macd' in indicators:
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['macd'],
                        name='MACD',
                        line=dict(color=theme['primary'], width=1.5)
                    ),
                    row=current_row, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['macd_signal'],
                        name='Signal',
                        line=dict(color=theme['secondary'], width=1.5)
                    ),
                    row=current_row, col=1
                )
                
                if 'macd_hist' in df.columns:
                    colors = [theme['candle_up'] if h >= 0 else theme['candle_down'] 
                             for h in df['macd_hist']]
                    fig.add_trace(
                        go.Bar(
                            x=df.index,
                            y=df['macd_hist'],
                            name='Histogram',
                            marker_color=colors,
                            opacity=0.3
                        ),
                        row=current_row, col=1
                    )
        
        # Update layout
        fig.update_layout(
            height=height,
            template="plotly_dark" if theme == "dark" else "plotly_white",
            paper_bgcolor=theme['chart_bg'],
            plot_bgcolor=theme['chart_bg'],
            font=dict(family="Inter, sans-serif", color=theme['text_primary']),
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor=theme['bg_card'],
                bordercolor=theme['border'],
                borderwidth=1
            )
        )
        
        # Update axes
        fig.update_xaxes(
            gridcolor=theme['chart_grid'],
            linecolor=theme['border'],
            tickfont=dict(color=theme['text_secondary'])
        )
        
        fig.update_yaxes(
            gridcolor=theme['chart_grid'],
            linecolor=theme['border'],
            tickfont=dict(color=theme['text_secondary'])
        )
        
        return fig
    
    @staticmethod
    def create_performance_chart(
        equity_curve: pd.Series,
        benchmark: Optional[pd.Series],
        theme: Dict,
        height: int = 400
    ) -> go.Figure:
        """
        Create performance chart with equity curve and drawdown.
        
        Args:
            equity_curve: Series with equity values
            benchmark: Optional benchmark series
            theme: Theme configuration
            height: Chart height
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Equity Curve', 'Drawdown'),
            row_heights=[0.7, 0.3]
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                name='Strategy',
                line=dict(color=theme['primary'], width=2),
                fill='tozeroy',
                fillcolor=f"{theme['primary']}20"
            ),
            row=1, col=1
        )
        
        # Benchmark if provided
        if benchmark is not None:
            fig.add_trace(
                go.Scatter(
                    x=benchmark.index,
                    y=benchmark.values,
                    name='Benchmark',
                    line=dict(color=theme['text_muted'], width=1, dash='dot')
                ),
                row=1, col=1
            )
        
        # Calculate drawdown
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax * 100
        
        # Drawdown chart
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                name='Drawdown',
                line=dict(color=theme['danger'], width=1),
                fill='tozeroy',
                fillcolor=f"{theme['danger']}20"
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=height,
            template="plotly_dark" if theme == "dark" else "plotly_white",
            paper_bgcolor=theme['chart_bg'],
            plot_bgcolor=theme['chart_bg'],
            font=dict(family="Inter, sans-serif", color=theme['text_primary']),
            hovermode='x unified',
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(
            gridcolor=theme['chart_grid'],
            linecolor=theme['border']
        )
        
        fig.update_yaxes(
            gridcolor=theme['chart_grid'],
            linecolor=theme['border']
        )
        
        fig.update_yaxes(title="Value", row=1, col=1)
        fig.update_yaxes(title="Drawdown %", row=2, col=1)
        
        return fig
    
    @staticmethod
    def create_heatmap(
        data: pd.DataFrame,
        theme: Dict,
        title: str = "Correlation Matrix",
        height: int = 500
    ) -> go.Figure:
        """Create correlation heatmap."""
        fig = go.Figure(data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(data.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(
                title="Correlation",
                titleside="right",
                tickmode="linear",
                tick0=-1,
                dtick=0.5
            )
        ))
        
        fig.update_layout(
            title=title,
            height=height,
            template="plotly_dark" if theme == "dark" else "plotly_white",
            paper_bgcolor=theme['chart_bg'],
            plot_bgcolor=theme['chart_bg'],
            font=dict(family="Inter, sans-serif", color=theme['text_primary']),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        return fig
    
    @staticmethod
    def create_pie_chart(
        labels: List[str],
        values: List[float],
        theme: Dict,
        title: str = "Portfolio Allocation",
        height: int = 400
    ) -> go.Figure:
        """Create pie chart for portfolio allocation."""
        colors = [theme['primary'], theme['secondary'], theme['success'], 
                 theme['warning'], theme['info'], theme['danger']]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(colors=colors[:len(labels)]),
            textposition='auto',
            textinfo='label+percent'
        )])
        
        fig.update_layout(
            title=title,
            height=height,
            template="plotly_dark" if theme == "dark" else "plotly_white",
            paper_bgcolor=theme['chart_bg'],
            plot_bgcolor=theme['chart_bg'],
            font=dict(family="Inter, sans-serif", color=theme['text_primary']),
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05
            )
        )
        
        return fig