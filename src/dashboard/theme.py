"""
Modern theme configuration for ML Trading Dashboard.
Provides dark/light themes with professional trading interface aesthetics.
"""

import streamlit as st
from typing import Dict, Literal

class DashboardTheme:
    """Dashboard theme manager with dark/light mode support."""
    
    # Color palettes
    THEMES = {
        "dark": {
            # Primary colors
            "primary": "#00D4FF",
            "secondary": "#FF6B6B",
            "success": "#4ECDC4",
            "warning": "#FFD93D",
            "danger": "#FF4757",
            "info": "#54A0FF",
            
            # Background colors
            "bg_primary": "#0E1117",
            "bg_secondary": "#1A1D29",
            "bg_card": "#262730",
            "bg_hover": "#2E3140",
            
            # Text colors
            "text_primary": "#FAFAFA",
            "text_secondary": "#B8BCC8",
            "text_muted": "#6C7293",
            
            # Chart colors
            "chart_bg": "#1A1D29",
            "chart_grid": "#2E3140",
            "candle_up": "#26A69A",
            "candle_down": "#EF5350",
            "volume": "#546E7A",
            
            # Border colors
            "border": "#2E3140",
            "border_focus": "#00D4FF",
        },
        "light": {
            # Primary colors
            "primary": "#0066CC",
            "secondary": "#FF4458",
            "success": "#10B981",
            "warning": "#F59E0B",
            "danger": "#EF4444",
            "info": "#3B82F6",
            
            # Background colors
            "bg_primary": "#FFFFFF",
            "bg_secondary": "#F8F9FA",
            "bg_card": "#FFFFFF",
            "bg_hover": "#F1F3F5",
            
            # Text colors
            "text_primary": "#1F2937",
            "text_secondary": "#6B7280",
            "text_muted": "#9CA3AF",
            
            # Chart colors
            "chart_bg": "#FFFFFF",
            "chart_grid": "#E5E7EB",
            "candle_up": "#10B981",
            "candle_down": "#EF4444",
            "volume": "#9CA3AF",
            
            # Border colors
            "border": "#E5E7EB",
            "border_focus": "#0066CC",
        }
    }
    
    @staticmethod
    def load_theme(theme_name: Literal["dark", "light"] = "dark") -> Dict:
        """Load and apply theme to Streamlit."""
        theme = DashboardTheme.THEMES[theme_name]
        
        # Custom CSS for Streamlit
        css = f"""
        <style>
            /* Import Google Fonts */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            
            /* Global styles */
            .stApp {{
                background-color: {theme['bg_primary']};
                font-family: 'Inter', sans-serif;
            }}
            
            /* Sidebar styling */
            .css-1d391kg, [data-testid="stSidebar"] {{
                background-color: {theme['bg_secondary']};
                border-right: 1px solid {theme['border']};
            }}
            
            /* Headers */
            h1, h2, h3, h4, h5, h6 {{
                color: {theme['text_primary']} !important;
                font-weight: 600;
            }}
            
            /* Text */
            p, span, div {{
                color: {theme['text_secondary']};
            }}
            
            /* Metric cards */
            [data-testid="metric-container"] {{
                background-color: {theme['bg_card']};
                border: 1px solid {theme['border']};
                padding: 1rem;
                border-radius: 0.5rem;
                box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
            }}
            
            [data-testid="metric-container"]:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                border-color: {theme['border_focus']};
            }}
            
            /* Buttons */
            .stButton > button {{
                background-color: {theme['primary']};
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 0.375rem;
                font-weight: 500;
                transition: all 0.2s ease;
            }}
            
            .stButton > button:hover {{
                background-color: {theme['primary']}dd;
                transform: translateY(-1px);
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }}
            
            /* Select boxes and inputs */
            .stSelectbox > div > div, .stTextInput > div > div > input {{
                background-color: {theme['bg_card']};
                color: {theme['text_primary']};
                border: 1px solid {theme['border']};
                border-radius: 0.375rem;
            }}
            
            .stSelectbox > div > div:focus, .stTextInput > div > div > input:focus {{
                border-color: {theme['border_focus']};
                box-shadow: 0 0 0 3px {theme['border_focus']}20;
            }}
            
            /* Data frames */
            .dataframe {{
                background-color: {theme['bg_card']} !important;
                color: {theme['text_primary']} !important;
            }}
            
            .dataframe th {{
                background-color: {theme['bg_secondary']} !important;
                color: {theme['text_primary']} !important;
                font-weight: 600;
            }}
            
            .dataframe td {{
                background-color: {theme['bg_card']} !important;
                color: {theme['text_secondary']} !important;
            }}
            
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {{
                background-color: {theme['bg_secondary']};
                border-radius: 0.5rem;
                padding: 0.25rem;
            }}
            
            .stTabs [data-baseweb="tab"] {{
                color: {theme['text_secondary']};
                background-color: transparent;
                border-radius: 0.375rem;
                padding: 0.5rem 1rem;
                font-weight: 500;
            }}
            
            .stTabs [aria-selected="true"] {{
                background-color: {theme['bg_card']};
                color: {theme['primary']};
            }}
            
            /* Expander */
            .streamlit-expanderHeader {{
                background-color: {theme['bg_card']};
                border: 1px solid {theme['border']};
                border-radius: 0.375rem;
                color: {theme['text_primary']};
            }}
            
            /* Success/Error/Warning/Info boxes */
            .stAlert {{
                background-color: {theme['bg_card']};
                border: 1px solid {theme['border']};
                border-radius: 0.375rem;
            }}
            
            /* Progress bar */
            .stProgress > div > div > div > div {{
                background-color: {theme['primary']};
            }}
            
            /* Plotly charts */
            .js-plotly-plot .plotly .modebar {{
                background-color: {theme['chart_bg']} !important;
            }}
            
            /* Custom card class */
            .dashboard-card {{
                background-color: {theme['bg_card']};
                border: 1px solid {theme['border']};
                border-radius: 0.5rem;
                padding: 1.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            }}
            
            /* Trading specific styles */
            .price-up {{
                color: {theme['candle_up']} !important;
                font-weight: 600;
            }}
            
            .price-down {{
                color: {theme['candle_down']} !important;
                font-weight: 600;
            }}
            
            /* Animations */
            @keyframes pulse {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
                100% {{ opacity: 1; }}
            }}
            
            .loading {{
                animation: pulse 2s infinite;
            }}
            
            /* Responsive design */
            @media (max-width: 768px) {{
                .dashboard-card {{
                    padding: 1rem;
                }}
                
                [data-testid="metric-container"] {{
                    padding: 0.75rem;
                }}
            }}
        </style>
        """
        
        return {"theme": theme, "css": css}
    
    @staticmethod
    def apply_theme(theme_name: Literal["dark", "light"] = "dark"):
        """Apply theme to current Streamlit app."""
        theme_config = DashboardTheme.load_theme(theme_name)
        st.markdown(theme_config["css"], unsafe_allow_html=True)
        return theme_config["theme"]
    
    @staticmethod
    def get_chart_layout(theme: Dict) -> Dict:
        """Get Plotly chart layout configuration for the theme."""
        return {
            "template": "plotly_dark" if theme == DashboardTheme.THEMES["dark"] else "plotly_white",
            "paper_bgcolor": theme["chart_bg"],
            "plot_bgcolor": theme["chart_bg"],
            "font": {
                "family": "Inter, sans-serif",
                "color": theme["text_primary"],
                "size": 12
            },
            "xaxis": {
                "gridcolor": theme["chart_grid"],
                "linecolor": theme["border"],
                "tickfont": {"color": theme["text_secondary"]}
            },
            "yaxis": {
                "gridcolor": theme["chart_grid"],
                "linecolor": theme["border"],
                "tickfont": {"color": theme["text_secondary"]}
            },
            "hoverlabel": {
                "bgcolor": theme["bg_card"],
                "bordercolor": theme["border"],
                "font": {"color": theme["text_primary"]}
            },
            "margin": {"l": 10, "r": 10, "t": 30, "b": 10}
        }
    
    @staticmethod
    def render_theme_toggle():
        """Render theme toggle in sidebar."""
        if "theme" not in st.session_state:
            st.session_state.theme = "dark"
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🌙 Dark", use_container_width=True):
                st.session_state.theme = "dark"
                st.rerun()
        with col2:
            if st.button("☀️ Light", use_container_width=True):
                st.session_state.theme = "light"
                st.rerun()
        
        return st.session_state.theme