"""
Alert and notification components for ML Trading Dashboard.
Toast notifications, alert banners, and notification management.
"""

import streamlit as st
from typing import Dict, List, Optional, Literal
from datetime import datetime, timedelta
import time
from enum import Enum

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertManager:
    """Manage alerts and notifications in the dashboard."""
    
    def __init__(self):
        """Initialize alert manager."""
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        if 'alert_settings' not in st.session_state:
            st.session_state.alert_settings = {
                'max_alerts': 50,
                'auto_dismiss': True,
                'dismiss_time': 5,
                'sound_enabled': False,
                'desktop_notifications': False
            }
    
    @staticmethod
    def add_alert(
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        title: Optional[str] = None,
        duration: Optional[int] = None,
        dismissible: bool = True,
        action: Optional[Dict] = None
    ):
        """
        Add a new alert to the system.
        
        Args:
            message: Alert message
            severity: Alert severity level
            title: Optional alert title
            duration: Duration in seconds (None for persistent)
            dismissible: Whether alert can be dismissed
            action: Optional action button config {"label": str, "callback": callable}
        """
        alert = {
            'id': f"alert_{datetime.now().timestamp()}",
            'message': message,
            'severity': severity.value,
            'title': title,
            'timestamp': datetime.now(),
            'duration': duration,
            'dismissible': dismissible,
            'action': action,
            'dismissed': False
        }
        
        st.session_state.alerts.append(alert)
        
        # Limit alerts
        max_alerts = st.session_state.alert_settings['max_alerts']
        if len(st.session_state.alerts) > max_alerts:
            st.session_state.alerts = st.session_state.alerts[-max_alerts:]
    
    @staticmethod
    def dismiss_alert(alert_id: str):
        """Dismiss an alert by ID."""
        for alert in st.session_state.alerts:
            if alert['id'] == alert_id:
                alert['dismissed'] = True
                break
    
    @staticmethod
    def clear_alerts(severity: Optional[AlertSeverity] = None):
        """Clear all alerts or alerts of specific severity."""
        if severity:
            st.session_state.alerts = [
                a for a in st.session_state.alerts
                if a['severity'] != severity.value
            ]
        else:
            st.session_state.alerts = []
    
    @staticmethod
    def render_alert_center(theme: Dict):
        """Render the alert notification center."""
        alerts = [a for a in st.session_state.alerts if not a['dismissed']]
        
        if not alerts:
            return
        
        # Alert container
        with st.container():
            st.markdown(
                f"""
                <div style="position: fixed; top: 70px; right: 20px; z-index: 999; 
                            max-width: 400px; max-height: 500px; overflow-y: auto;">
                """,
                unsafe_allow_html=True
            )
            
            for alert in reversed(alerts[-5:]):  # Show last 5 alerts
                AlertComponents.render_toast(
                    alert['message'],
                    alert['severity'],
                    alert['title'],
                    alert['id'],
                    theme
                )
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    @staticmethod
    def get_active_alerts(severity: Optional[AlertSeverity] = None) -> List[Dict]:
        """Get active (non-dismissed) alerts."""
        alerts = [a for a in st.session_state.alerts if not a['dismissed']]
        
        if severity:
            alerts = [a for a in alerts if a['severity'] == severity.value]
        
        return alerts


class AlertComponents:
    """Alert UI components."""
    
    @staticmethod
    def render_toast(
        message: str,
        severity: str,
        title: Optional[str],
        alert_id: str,
        theme: Dict
    ):
        """Render a toast notification."""
        colors = {
            'info': theme['info'],
            'success': theme['success'],
            'warning': theme['warning'],
            'error': theme['danger'],
            'critical': theme['danger']
        }
        
        icons = {
            'info': '💡',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌',
            'critical': '🚨'
        }
        
        color = colors.get(severity, theme['info'])
        icon = icons.get(severity, '📢')
        
        st.markdown(
            f"""
            <div style="background-color: {theme['bg_card']}; 
                        border-left: 4px solid {color};
                        border-radius: 5px;
                        padding: 12px;
                        margin-bottom: 10px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        animation: slideIn 0.3s ease-out;">
                <div style="display: flex; align-items: start; justify-content: space-between;">
                    <div style="display: flex; align-items: start;">
                        <span style="font-size: 1.5em; margin-right: 10px;">{icon}</span>
                        <div>
                            {f'<strong style="color: {theme["text_primary"]};">{title}</strong><br>' if title else ''}
                            <span style="color: {theme['text_secondary']};">{message}</span>
                        </div>
                    </div>
                    <button onclick="this.parentElement.parentElement.style.display='none'" 
                            style="background: none; border: none; color: {theme['text_muted']}; 
                                   cursor: pointer; font-size: 1.2em;">×</button>
                </div>
            </div>
            
            <style>
                @keyframes slideIn {{
                    from {{ transform: translateX(100%); opacity: 0; }}
                    to {{ transform: translateX(0); opacity: 1; }}
                }}
            </style>
            """,
            unsafe_allow_html=True
        )
    
    @staticmethod
    def render_alert_banner(
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        icon: Optional[str] = None,
        dismissible: bool = True,
        theme: Optional[Dict] = None
    ):
        """Render an alert banner."""
        if theme is None:
            theme = {'info': '#3B82F6', 'success': '#10B981', 'warning': '#F59E0B', 'danger': '#EF4444'}
        
        severity_configs = {
            AlertSeverity.INFO: {
                'color': theme.get('info', '#3B82F6'),
                'bg_color': f"{theme.get('info', '#3B82F6')}20",
                'icon': icon or '💡'
            },
            AlertSeverity.SUCCESS: {
                'color': theme.get('success', '#10B981'),
                'bg_color': f"{theme.get('success', '#10B981')}20",
                'icon': icon or '✅'
            },
            AlertSeverity.WARNING: {
                'color': theme.get('warning', '#F59E0B'),
                'bg_color': f"{theme.get('warning', '#F59E0B')}20",
                'icon': icon or '⚠️'
            },
            AlertSeverity.ERROR: {
                'color': theme.get('danger', '#EF4444'),
                'bg_color': f"{theme.get('danger', '#EF4444')}20",
                'icon': icon or '❌'
            },
            AlertSeverity.CRITICAL: {
                'color': theme.get('danger', '#EF4444'),
                'bg_color': f"{theme.get('danger', '#EF4444')}30",
                'icon': icon or '🚨'
            }
        }
        
        config = severity_configs[severity]
        
        alert_html = f"""
        <div style="background-color: {config['bg_color']}; 
                    border-left: 4px solid {config['color']};
                    padding: 12px 16px;
                    margin: 10px 0;
                    border-radius: 5px;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 1.5em; margin-right: 12px;">{config['icon']}</span>
                <span>{message}</span>
            </div>
        """
        
        if dismissible:
            alert_html += """
            <button onclick="this.parentElement.style.display='none'" 
                    style="background: none; border: none; cursor: pointer; 
                           font-size: 1.5em; color: inherit; opacity: 0.7;">×</button>
            """
        
        alert_html += "</div>"
        
        st.markdown(alert_html, unsafe_allow_html=True)
    
    @staticmethod
    def render_alert_list(theme: Dict, limit: int = 10):
        """Render a list of recent alerts."""
        st.markdown("### 🔔 Recent Alerts")
        
        alerts = AlertManager.get_active_alerts()
        
        if not alerts:
            st.info("No active alerts")
            return
        
        # Alert filters
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            severity_filter = st.multiselect(
                "Filter by severity",
                [s.value for s in AlertSeverity],
                default=[s.value for s in AlertSeverity]
            )
        
        with col2:
            if st.button("Clear All"):
                AlertManager.clear_alerts()
                st.rerun()
        
        with col3:
            if st.button("Mark All Read"):
                for alert in alerts:
                    alert['dismissed'] = True
                st.rerun()
        
        # Display alerts
        filtered_alerts = [a for a in alerts if a['severity'] in severity_filter]
        
        for alert in filtered_alerts[:limit]:
            severity_colors = {
                'info': theme['info'],
                'success': theme['success'],
                'warning': theme['warning'],
                'error': theme['danger'],
                'critical': theme['danger']
            }
            
            severity_icons = {
                'info': '💡',
                'success': '✅',
                'warning': '⚠️',
                'error': '❌',
                'critical': '🚨'
            }
            
            with st.container():
                col1, col2, col3 = st.columns([1, 8, 1])
                
                with col1:
                    st.markdown(
                        f"<span style='font-size: 1.5em;'>{severity_icons[alert['severity']]}</span>",
                        unsafe_allow_html=True
                    )
                
                with col2:
                    if alert.get('title'):
                        st.markdown(f"**{alert['title']}**")
                    st.markdown(alert['message'])
                    st.caption(f"{alert['timestamp'].strftime('%H:%M:%S')} - {alert['severity'].upper()}")
                
                with col3:
                    if alert.get('dismissible', True):
                        if st.button("×", key=alert['id']):
                            AlertManager.dismiss_alert(alert['id'])
                            st.rerun()
                
                st.markdown("---")
    
    @staticmethod
    def render_alert_settings(theme: Dict):
        """Render alert settings panel."""
        st.markdown("### ⚙️ Alert Settings")
        
        settings = st.session_state.alert_settings
        
        col1, col2 = st.columns(2)
        
        with col1:
            settings['auto_dismiss'] = st.checkbox(
                "Auto-dismiss alerts",
                value=settings['auto_dismiss']
            )
            
            if settings['auto_dismiss']:
                settings['dismiss_time'] = st.slider(
                    "Dismiss after (seconds)",
                    min_value=1,
                    max_value=30,
                    value=settings['dismiss_time']
                )
            
            settings['max_alerts'] = st.number_input(
                "Maximum alerts to keep",
                min_value=10,
                max_value=100,
                value=settings['max_alerts']
            )
        
        with col2:
            settings['sound_enabled'] = st.checkbox(
                "🔊 Sound notifications",
                value=settings['sound_enabled']
            )
            
            settings['desktop_notifications'] = st.checkbox(
                "💻 Desktop notifications",
                value=settings['desktop_notifications']
            )
            
            alert_types = st.multiselect(
                "Alert types to show",
                [s.value for s in AlertSeverity],
                default=[s.value for s in AlertSeverity]
            )
        
        if st.button("Save Settings"):
            st.session_state.alert_settings = settings
            AlertManager.add_alert(
                "Settings saved successfully",
                AlertSeverity.SUCCESS,
                "Settings Updated"
            )
            st.rerun()