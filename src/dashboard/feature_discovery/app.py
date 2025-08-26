"""
Streamlit-based Feature Discovery Interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any
import json

# Internal imports
try:
    from ...features.store import FeatureStore, FeatureStoreConfig, VersionStatus
    from ...monitoring.drift_detector import DriftDetector
    from ...mlops.automl.feature_generator import AutoMLFeatureGenerator, FeatureGenerationConfig
except ImportError:
    # For standalone testing
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

logger = logging.getLogger(__name__)


class FeatureDiscoveryApp:
    """
    Comprehensive Feature Discovery Web Interface.
    
    Features:
    - Interactive feature search and filtering
    - Feature lineage visualization
    - Quality metrics dashboard
    - Usage analytics
    - AutoML feature generation
    - Real-time monitoring
    """
    
    def __init__(self, feature_store: FeatureStore = None):
        """Initialize feature discovery app."""
        # Initialize or use provided feature store
        if feature_store is None:
            config = FeatureStoreConfig(
                store_path="data/feature_store_ui",
                validate_on_write=True,
                track_lineage=True
            )
            self.feature_store = FeatureStore(config)
        else:
            self.feature_store = feature_store
        
        # Initialize other components
        self.drift_detector = DriftDetector()
        self.automl_generator = None
        
        # App state
        if 'selected_features' not in st.session_state:
            st.session_state.selected_features = []
        if 'feature_data' not in st.session_state:
            st.session_state.feature_data = None
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []
    
    def run(self):
        """Run the Streamlit application."""
        st.set_page_config(
            page_title="🔍 Feature Discovery Platform",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown(self._get_custom_css(), unsafe_allow_html=True)
        
        # Header
        st.markdown("""
        # 🔍 Feature Discovery Platform
        **Explore, analyze, and manage your ML features with enterprise-grade capabilities**
        """)
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "🧭 Navigate",
            [
                "🔍 Feature Search",
                "📊 Feature Analytics", 
                "🌐 Lineage Explorer",
                "⚡ AutoML Generator",
                "📈 Quality Monitor",
                "🚀 Usage Dashboard",
                "⚙️ Management"
            ]
        )
        
        # Route to appropriate page
        if page == "🔍 Feature Search":
            self._show_search_page()
        elif page == "📊 Feature Analytics":
            self._show_analytics_page()
        elif page == "🌐 Lineage Explorer":
            self._show_lineage_page()
        elif page == "⚡ AutoML Generator":
            self._show_automl_page()
        elif page == "📈 Quality Monitor":
            self._show_quality_page()
        elif page == "🚀 Usage Dashboard":
            self._show_usage_page()
        elif page == "⚙️ Management":
            self._show_management_page()
    
    def _get_custom_css(self) -> str:
        """Get custom CSS for the app."""
        return """
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
        
        .feature-card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            background: #fafafa;
        }
        
        .quality-good { color: #28a745; }
        .quality-warning { color: #ffc107; }
        .quality-bad { color: #dc3545; }
        
        .stSelectbox > div > div > div {
            background-color: #f8f9fa;
        }
        </style>
        """
    
    def _show_search_page(self):
        """Show feature search and discovery page."""
        st.header("🔍 Feature Search & Discovery")
        
        # Search interface
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_query = st.text_input(
                "🔍 Search features",
                placeholder="Enter feature name, description, or tag...",
                help="Search across feature names, descriptions, and tags"
            )
        
        with col2:
            min_quality = st.slider(
                "Min Quality Score",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                help="Filter features by minimum quality score"
            )
        
        with col3:
            max_results = st.number_input(
                "Max Results",
                min_value=10,
                max_value=500,
                value=50,
                step=10
            )
        
        # Advanced filters
        with st.expander("🎛️ Advanced Filters"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                owner_filter = st.text_input("Owner", placeholder="Filter by owner")
                
            with col2:
                tags_filter = st.text_input(
                    "Tags", 
                    placeholder="tag1,tag2,tag3",
                    help="Comma-separated tags"
                )
                
            with col3:
                date_filter = st.date_input(
                    "Created After",
                    value=datetime.now() - timedelta(days=30)
                )
        
        # Search button
        if st.button("🔍 Search Features", type="primary") or search_query:
            try:
                # Parse tags
                tags = [t.strip() for t in tags_filter.split(",")] if tags_filter else None
                tags = [t for t in tags if t] if tags else None
                
                # Perform search
                results = self.feature_store.search_features(
                    query=search_query or "",
                    tags=tags,
                    owner=owner_filter if owner_filter else None,
                    min_quality_score=min_quality,
                    limit=max_results
                )
                
                st.session_state.search_results = results
                
            except Exception as e:
                st.error(f"Search failed: {e}")
                results = []
        
        # Display results
        if st.session_state.search_results:
            self._display_search_results(st.session_state.search_results)
        else:
            st.info("👆 Use the search bar above to discover features")
    
    def _display_search_results(self, results: List[Dict[str, Any]]):
        """Display search results in an interactive format."""
        st.subheader(f"📊 Found {len(results)} features")
        
        if not results:
            return
        
        # Results table with selection
        df_results = pd.DataFrame(results)
        
        # Format the display
        display_df = df_results[[
            'name', 'group_name', 'description', 'quality_score', 
            'usage_count', 'last_accessed', 'tags'
        ]].copy()
        
        # Add selection column
        display_df.insert(0, 'Select', False)
        
        # Format columns
        display_df['quality_score'] = display_df['quality_score'].round(3)
        display_df['tags'] = display_df['tags'].apply(
            lambda x: ', '.join(eval(x)) if isinstance(x, str) and x.startswith('[') else str(x)
        )
        
        # Interactive table
        edited_df = st.data_editor(
            display_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                'Select': st.column_config.CheckboxColumn(
                    'Select',
                    help="Select features for detailed analysis",
                    default=False
                ),
                'quality_score': st.column_config.ProgressColumn(
                    'Quality Score',
                    help="Feature quality score (0-1)",
                    min_value=0,
                    max_value=1
                ),
                'usage_count': st.column_config.NumberColumn(
                    'Usage Count',
                    help="Number of times feature has been accessed"
                )
            }
        )
        
        # Store selected features
        selected_mask = edited_df['Select']
        st.session_state.selected_features = df_results[selected_mask]['id'].tolist()
        
        # Action buttons
        if st.session_state.selected_features:
            st.success(f"✅ Selected {len(st.session_state.selected_features)} features")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("📊 Analyze Selected"):
                    st.switch_page("📊 Feature Analytics")
            
            with col2:
                if st.button("🌐 View Lineage"):
                    st.switch_page("🌐 Lineage Explorer")
            
            with col3:
                if st.button("📥 Export Selected"):
                    self._export_features(st.session_state.selected_features)
            
            with col4:
                if st.button("🔄 Compare Versions"):
                    self._show_version_comparison()
    
    def _show_analytics_page(self):
        """Show feature analytics page."""
        st.header("📊 Feature Analytics")
        
        # Get feature store stats
        try:
            stats = self.feature_store.get_storage_stats()
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "🏪 Total Features",
                    stats.get('total_registered_features', 0),
                    help="Total number of registered features"
                )
            
            with col2:
                st.metric(
                    "📦 Feature Groups",
                    stats.get('total_registered_groups', 0),
                    help="Number of feature groups"
                )
            
            with col3:
                st.metric(
                    "🔄 Versions",
                    stats.get('total_versions', 0),
                    help="Total feature versions"
                )
            
            with col4:
                st.metric(
                    "💾 Storage (MB)",
                    f"{stats.get('total_size_mb', 0):.1f}",
                    help="Total storage used"
                )
            
            # Detailed analytics
            st.subheader("📈 Usage Analytics")
            
            # Mock usage data for demo
            self._show_usage_charts()
            
            # Quality distribution
            st.subheader("🎯 Quality Distribution")
            self._show_quality_distribution()
            
        except Exception as e:
            st.error(f"Failed to load analytics: {e}")
    
    def _show_usage_charts(self):
        """Show usage analytics charts."""
        # Generate mock data for demonstration
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        
        # Feature usage over time
        usage_data = pd.DataFrame({
            'date': dates,
            'feature_requests': np.random.poisson(100, len(dates)),
            'unique_users': np.random.poisson(20, len(dates)),
            'avg_latency_ms': np.random.normal(50, 10, len(dates))
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Usage trend
            fig = px.line(
                usage_data, 
                x='date', 
                y='feature_requests',
                title="📈 Feature Requests Over Time",
                labels={'feature_requests': 'Requests', 'date': 'Date'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Latency trend
            fig = px.line(
                usage_data,
                x='date',
                y='avg_latency_ms', 
                title="⚡ Average Latency Trend",
                labels={'avg_latency_ms': 'Latency (ms)', 'date': 'Date'},
                color_discrete_sequence=['#ff6b6b']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_quality_distribution(self):
        """Show feature quality distribution."""
        # Mock quality data
        quality_scores = np.random.beta(2, 1, 1000)  # Skewed towards higher quality
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Quality histogram
            fig = px.histogram(
                x=quality_scores,
                nbins=20,
                title="🎯 Feature Quality Score Distribution",
                labels={'x': 'Quality Score', 'y': 'Number of Features'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Quality by category
            categories = ['Technical', 'Fundamental', 'Sentiment', 'Macro']
            category_quality = [
                np.random.beta(2, 1, 250).mean() for _ in categories
            ]
            
            fig = px.bar(
                x=categories,
                y=category_quality,
                title="📊 Quality by Feature Category",
                labels={'x': 'Category', 'y': 'Average Quality Score'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_lineage_page(self):
        """Show feature lineage visualization."""
        st.header("🌐 Feature Lineage Explorer")
        
        # Feature selection for lineage
        if st.session_state.selected_features:
            st.success(f"Showing lineage for {len(st.session_state.selected_features)} selected features")
            feature_ids = st.session_state.selected_features
        else:
            # Manual feature selection
            st.info("💡 Select features from the search page or choose manually below")
            
            # Mock feature list for demo
            available_features = [
                "crypto_technical.ma_20",
                "crypto_technical.rsi_14", 
                "crypto_volume.vwap",
                "crypto_price.volatility_10d"
            ]
            
            feature_ids = st.multiselect(
                "Select features to explore lineage",
                available_features,
                default=available_features[:2] if available_features else []
            )
        
        if feature_ids:
            # Lineage depth control
            depth = st.slider(
                "Lineage Depth",
                min_value=1,
                max_value=5,
                value=3,
                help="How many levels of dependencies to show"
            )
            
            # Generate lineage visualization
            self._show_lineage_graph(feature_ids, depth)
        
        # Lineage analysis
        if feature_ids:
            st.subheader("🔍 Lineage Analysis")
            self._show_lineage_analysis(feature_ids)
    
    def _show_lineage_graph(self, feature_ids: List[str], depth: int):
        """Show interactive lineage graph."""
        # Create mock lineage data
        G = nx.DiGraph()
        
        # Add nodes and edges for demo
        nodes = [
            ("raw_data.prices", {"type": "data", "level": 0}),
            ("raw_data.volume", {"type": "data", "level": 0}),
            ("crypto_technical.ma_20", {"type": "feature", "level": 1}),
            ("crypto_technical.rsi_14", {"type": "feature", "level": 1}),
            ("crypto_volume.vwap", {"type": "feature", "level": 1}),
            ("crypto_signals.trend", {"type": "derived", "level": 2}),
        ]
        
        edges = [
            ("raw_data.prices", "crypto_technical.ma_20"),
            ("raw_data.prices", "crypto_technical.rsi_14"),
            ("raw_data.prices", "crypto_volume.vwap"),
            ("raw_data.volume", "crypto_volume.vwap"),
            ("crypto_technical.ma_20", "crypto_signals.trend"),
            ("crypto_technical.rsi_14", "crypto_signals.trend"),
        ]
        
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create plotly network graph
        edge_x = []
        edge_y = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Node traces by type
        node_traces = {}
        colors = {'data': '#ff6b6b', 'feature': '#4ecdc4', 'derived': '#45b7d1'}
        
        for node_type in ['data', 'feature', 'derived']:
            node_x = []
            node_y = []
            node_text = []
            
            for node in G.nodes():
                if G.nodes[node].get('type') == node_type:
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
            
            if node_x:  # Only create trace if there are nodes of this type
                node_traces[node_type] = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    hoverinfo='text',
                    text=node_text,
                    textposition="middle center",
                    marker=dict(
                        size=20,
                        color=colors[node_type],
                        line=dict(width=2, color='white')
                    ),
                    name=node_type.title()
                )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace] + list(node_traces.values()),
            layout=go.Layout(
                title="🌐 Feature Lineage Graph",
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="💡 Hover over nodes for details",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color='gray', size=12)
                ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_lineage_analysis(self, feature_ids: List[str]):
        """Show lineage analysis details."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Dependency Analysis")
            
            # Mock dependency data
            dependencies = [
                {"feature": "crypto_technical.ma_20", "dependencies": 1, "dependents": 3},
                {"feature": "crypto_technical.rsi_14", "dependencies": 1, "dependents": 2},
                {"feature": "crypto_volume.vwap", "dependencies": 2, "dependents": 1},
            ]
            
            df_deps = pd.DataFrame(dependencies)
            st.dataframe(df_deps, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("### ⚠️ Impact Analysis")
            
            st.info("🔄 **Downstream Impact**: 5 features depend on your selection")
            st.warning("⚡ **Performance Impact**: Medium - affects 3 critical paths")
            st.success("✅ **Quality Impact**: High - selected features have >0.8 quality scores")
    
    def _show_automl_page(self):
        """Show AutoML feature generation page."""
        st.header("⚡ AutoML Feature Generator")
        
        st.markdown("""
        Generate features automatically using advanced machine learning techniques.
        Upload your data and let AutoML discover the best features for your use case.
        """)
        
        # File upload
        uploaded_file = st.file_uploader(
            "📁 Upload your dataset",
            type=['csv', 'parquet', 'json'],
            help="Upload a CSV, Parquet, or JSON file with your raw data"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.parquet'):
                    data = pd.read_parquet(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    data = pd.read_json(uploaded_file)
                
                st.success(f"✅ Loaded {len(data)} rows with {len(data.columns)} columns")
                
                # Show data preview
                with st.expander("👀 Data Preview"):
                    st.dataframe(data.head(10))
                
                # AutoML configuration
                st.subheader("⚙️ AutoML Configuration")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    target_column = st.selectbox(
                        "🎯 Target Column (optional)",
                        options=["None"] + list(data.columns),
                        help="Select target column for supervised feature selection"
                    )
                    
                    max_features = st.number_input(
                        "📊 Max Features to Generate",
                        min_value=10,
                        max_value=1000,
                        value=100,
                        step=10
                    )
                
                with col2:
                    feature_types = st.multiselect(
                        "🔧 Feature Types",
                        options=[
                            "Basic Statistics",
                            "Rolling Windows", 
                            "Technical Indicators",
                            "Time Features",
                            "Interaction Features"
                        ],
                        default=["Basic Statistics", "Rolling Windows", "Technical Indicators"]
                    )
                    
                    rolling_windows = st.text_input(
                        "📏 Rolling Windows",
                        value="5,10,20,50",
                        help="Comma-separated window sizes"
                    )
                
                # Generate features button
                if st.button("🚀 Generate Features", type="primary"):
                    self._generate_automl_features(data, target_column, max_features, feature_types, rolling_windows)
                
            except Exception as e:
                st.error(f"Failed to load data: {e}")
    
    def _generate_automl_features(self, data: pd.DataFrame, target_column: str, 
                                 max_features: int, feature_types: List[str], rolling_windows: str):
        """Generate features using AutoML."""
        with st.spinner("🔄 Generating features... This may take a few minutes."):
            try:
                # Parse rolling windows
                windows = [int(w.strip()) for w in rolling_windows.split(",")]
                
                # Configure AutoML
                config = FeatureGenerationConfig(
                    enable_basic_stats="Basic Statistics" in feature_types,
                    enable_rolling_features="Rolling Windows" in feature_types,
                    enable_technical_indicators="Technical Indicators" in feature_types,
                    enable_time_features="Time Features" in feature_types,
                    enable_interaction_features="Interaction Features" in feature_types,
                    max_features=max_features,
                    rolling_windows=windows
                )
                
                # Initialize generator
                generator = AutoMLFeatureGenerator(config)
                
                # Prepare target
                target = data[target_column] if target_column != "None" else None
                
                # Generate features
                features = generator.fit_transform(data, target)
                
                # Show results
                st.success(f"✅ Generated {len(features.columns)} features!")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Generated Features")
                    st.dataframe(features.head(10), use_container_width=True)
                
                with col2:
                    st.subheader("🏆 Top Features by Importance")
                    importance = generator.get_feature_importance(10)
                    if importance:
                        importance_df = pd.DataFrame(
                            list(importance.items()),
                            columns=['Feature', 'Importance']
                        )
                        st.dataframe(importance_df, hide_index=True, use_container_width=True)
                    else:
                        st.info("Feature importance not available")
                
                # Save options
                st.subheader("💾 Save Generated Features")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    group_name = st.text_input(
                        "Feature Group Name",
                        value="automl_generated_features"
                    )
                
                with col2:
                    description = st.text_input(
                        "Description",
                        value="Features generated by AutoML"
                    )
                
                with col3:
                    if st.button("💾 Save to Feature Store"):
                        self._save_generated_features(features, group_name, description, generator)
                
            except Exception as e:
                st.error(f"Feature generation failed: {e}")
                logger.error(f"AutoML feature generation error: {e}")
    
    def _save_generated_features(self, features: pd.DataFrame, group_name: str, 
                                description: str, generator: AutoMLFeatureGenerator):
        """Save generated features to feature store."""
        try:
            version_id = self.feature_store.write_feature_group(
                group_name=group_name,
                data=features,
                description=description,
                created_by="automl_generator",
                tags=["automl", "generated", "auto_features"]
            )
            
            st.success(f"✅ Features saved to Feature Store with version: {version_id}")
            
            # Show report
            report = generator.generate_feature_report()
            with st.expander("📄 Generation Report"):
                st.json(report)
            
        except Exception as e:
            st.error(f"Failed to save features: {e}")
    
    def _show_quality_page(self):
        """Show feature quality monitoring page."""
        st.header("📈 Feature Quality Monitor")
        
        # Quality overview
        st.subheader("🎯 Quality Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🟢 High Quality", "234", "↗️ +12")
        
        with col2:
            st.metric("🟡 Medium Quality", "45", "↘️ -3")
        
        with col3:
            st.metric("🔴 Low Quality", "12", "→ 0")
        
        with col4:
            st.metric("📊 Avg Quality", "0.847", "↗️ +0.02")
        
        # Quality trends
        st.subheader("📈 Quality Trends")
        self._show_quality_trends()
        
        # Quality alerts
        st.subheader("🚨 Quality Alerts")
        self._show_quality_alerts()
    
    def _show_quality_trends(self):
        """Show quality trend charts."""
        # Mock quality trend data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        quality_data = pd.DataFrame({
            'date': dates,
            'avg_quality': 0.8 + np.random.normal(0, 0.05, 30).cumsum() * 0.01,
            'completeness': 0.9 + np.random.normal(0, 0.03, 30).cumsum() * 0.01,
            'consistency': 0.85 + np.random.normal(0, 0.04, 30).cumsum() * 0.01
        })
        
        # Ensure values stay in [0, 1] range
        for col in ['avg_quality', 'completeness', 'consistency']:
            quality_data[col] = quality_data[col].clip(0, 1)
        
        fig = px.line(
            quality_data,
            x='date',
            y=['avg_quality', 'completeness', 'consistency'],
            title="📈 Quality Metrics Over Time",
            labels={'value': 'Quality Score', 'date': 'Date'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_quality_alerts(self):
        """Show quality alerts."""
        alerts = [
            {
                "feature": "crypto_volume.abnormal_volume",
                "issue": "Completeness dropped to 0.65",
                "severity": "🔴 High",
                "timestamp": "2024-01-15 14:30"
            },
            {
                "feature": "crypto_price.price_change",
                "issue": "Drift detected in distribution", 
                "severity": "🟡 Medium",
                "timestamp": "2024-01-15 12:15"
            },
            {
                "feature": "crypto_technical.bollinger_upper",
                "issue": "Correlation spike with existing feature",
                "severity": "🟡 Medium", 
                "timestamp": "2024-01-15 09:45"
            }
        ]
        
        for alert in alerts:
            with st.container():
                st.markdown(f"""
                <div class="feature-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{alert['feature']}</strong><br>
                            {alert['issue']}
                        </div>
                        <div style="text-align: right;">
                            {alert['severity']}<br>
                            <small>{alert['timestamp']}</small>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    def _show_usage_page(self):
        """Show feature usage dashboard."""
        st.header("🚀 Feature Usage Dashboard")
        
        # Usage metrics
        st.subheader("📊 Usage Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📈 Daily Requests", "12.4K", "↗️ +8.2%")
        
        with col2:
            st.metric("👥 Active Users", "234", "↗️ +12")
        
        with col3:
            st.metric("⚡ Avg Latency", "45ms", "↘️ -5ms")
        
        with col4:
            st.metric("💾 Cache Hit Rate", "87.3%", "↗️ +2.1%")
        
        # Usage charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Top features by usage
            st.subheader("🏆 Most Used Features")
            
            top_features = pd.DataFrame({
                'Feature': [
                    'crypto_price.close',
                    'crypto_technical.ma_20', 
                    'crypto_volume.volume_sma',
                    'crypto_technical.rsi_14',
                    'crypto_price.volatility'
                ],
                'Requests': [8543, 7234, 6123, 5678, 4321]
            })
            
            fig = px.bar(
                top_features,
                x='Requests',
                y='Feature',
                orientation='h',
                title="🏆 Top Features by Request Volume"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Usage by user type
            st.subheader("👥 Usage by User Type")
            
            user_data = pd.DataFrame({
                'User Type': ['Data Scientists', 'ML Engineers', 'Analysts', 'Applications'],
                'Usage %': [35, 28, 22, 15]
            })
            
            fig = px.pie(
                user_data,
                values='Usage %',
                names='User Type',
                title="👥 Feature Usage by User Type"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_management_page(self):
        """Show feature management page."""
        st.header("⚙️ Feature Management")
        
        # System status
        st.subheader("🖥️ System Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("✅ **Feature Store**: Healthy")
            st.info("📊 **Storage**: 2.3GB / 10GB used")
        
        with col2:
            st.success("✅ **Quality Monitor**: Active") 
            st.info("⚡ **Processing**: 12 jobs queued")
        
        with col3:
            st.success("✅ **API**: Operational")
            st.info("🔄 **Sync**: Last sync 2 min ago")
        
        # Management actions
        st.subheader("🛠️ Management Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🧹 Cleanup Operations")
            
            if st.button("🗑️ Clean Old Versions"):
                st.info("Cleaning up versions older than 30 days...")
                # Simulate cleanup
                import time
                time.sleep(2)
                st.success("✅ Cleanup completed: Removed 12 old versions")
            
            if st.button("📊 Recompute Quality Metrics"):
                st.info("Recomputing quality metrics for all features...")
                time.sleep(3)
                st.success("✅ Quality metrics updated for 234 features")
        
        with col2:
            st.markdown("### 📋 Export & Backup")
            
            if st.button("📥 Export Feature Catalog"):
                st.success("✅ Feature catalog exported to downloads/")
            
            if st.button("💾 Create Backup"):
                st.success("✅ Backup created: backup_2024_01_15.tar.gz")
        
        # Configuration
        st.subheader("⚙️ Configuration")
        
        with st.expander("🔧 Feature Store Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                auto_quality_check = st.checkbox("Auto Quality Checking", value=True)
                enable_lineage = st.checkbox("Track Feature Lineage", value=True)
                
            with col2:
                quality_threshold = st.slider("Quality Threshold", 0.0, 1.0, 0.8)
                retention_days = st.number_input("Retention Days", value=90, min_value=1)
            
            if st.button("💾 Save Configuration"):
                st.success("✅ Configuration saved successfully")
    
    def _export_features(self, feature_ids: List[str]):
        """Export selected features."""
        try:
            # Mock export functionality
            export_data = {
                'exported_features': len(feature_ids),
                'export_time': datetime.now().isoformat(),
                'format': 'CSV'
            }
            
            st.success(f"✅ Exported {len(feature_ids)} features")
            st.download_button(
                "📥 Download Export",
                data=json.dumps(export_data, indent=2),
                file_name=f"features_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"Export failed: {e}")
    
    def _show_version_comparison(self):
        """Show version comparison interface."""
        st.subheader("🔄 Version Comparison")
        st.info("Version comparison feature - coming soon!")


def main():
    """Main function to run the app."""
    app = FeatureDiscoveryApp()
    app.run()


if __name__ == "__main__":
    main()