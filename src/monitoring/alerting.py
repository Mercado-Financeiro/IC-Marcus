"""
Advanced Alerting System for Data Quality Monitoring.
Supports multiple channels and intelligent alert management.
"""
import smtplib
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
import hashlib
import time
from collections import deque
import requests
import asyncio
from jinja2 import Template

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels."""
    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    LOG = "log"


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    source: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'metrics': self.metrics,
            'tags': self.tags,
            'metadata': self.metadata
        }


class AlertDeduplicator:
    """Deduplicates alerts to prevent spam."""
    
    def __init__(self, window_seconds: int = 300):
        """
        Initialize deduplicator.
        
        Args:
            window_seconds: Time window for deduplication
        """
        self.window_seconds = window_seconds
        self.alert_cache = {}
        
    def is_duplicate(self, alert: Alert) -> bool:
        """Check if alert is duplicate within window."""
        # Create hash of alert content
        alert_hash = self._hash_alert(alert)
        
        # Check cache
        if alert_hash in self.alert_cache:
            last_sent = self.alert_cache[alert_hash]
            if (alert.timestamp - last_sent).total_seconds() < self.window_seconds:
                return True
        
        # Update cache
        self.alert_cache[alert_hash] = alert.timestamp
        
        # Clean old entries
        self._clean_cache(alert.timestamp)
        
        return False
    
    def _hash_alert(self, alert: Alert) -> str:
        """Create hash of alert for deduplication."""
        content = f"{alert.severity.value}:{alert.title}:{alert.source}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _clean_cache(self, current_time: datetime):
        """Remove old entries from cache."""
        cutoff_time = current_time - timedelta(seconds=self.window_seconds * 2)
        self.alert_cache = {
            k: v for k, v in self.alert_cache.items()
            if v > cutoff_time
        }


class AlertThrottler:
    """Throttles alerts to prevent overwhelming recipients."""
    
    def __init__(self, 
                 max_alerts_per_minute: int = 10,
                 max_alerts_per_hour: int = 100):
        """
        Initialize throttler.
        
        Args:
            max_alerts_per_minute: Maximum alerts per minute
            max_alerts_per_hour: Maximum alerts per hour
        """
        self.max_per_minute = max_alerts_per_minute
        self.max_per_hour = max_alerts_per_hour
        self.minute_window = deque(maxlen=max_alerts_per_minute)
        self.hour_window = deque(maxlen=max_alerts_per_hour)
        
    def should_send(self, timestamp: datetime) -> bool:
        """Check if alert should be sent based on throttling."""
        # Check minute limit
        minute_cutoff = timestamp - timedelta(minutes=1)
        recent_minute = sum(1 for t in self.minute_window if t > minute_cutoff)
        
        if recent_minute >= self.max_per_minute:
            logger.warning(f"Throttling: {recent_minute} alerts in last minute")
            return False
        
        # Check hour limit
        hour_cutoff = timestamp - timedelta(hours=1)
        recent_hour = sum(1 for t in self.hour_window if t > hour_cutoff)
        
        if recent_hour >= self.max_per_hour:
            logger.warning(f"Throttling: {recent_hour} alerts in last hour")
            return False
        
        # Add to windows
        self.minute_window.append(timestamp)
        self.hour_window.append(timestamp)
        
        return True


class EmailChannel:
    """Email alert channel."""
    
    def __init__(self,
                 smtp_host: str,
                 smtp_port: int,
                 username: str,
                 password: str,
                 from_email: str,
                 to_emails: List[str],
                 use_tls: bool = True):
        """Initialize email channel."""
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
        self.use_tls = use_tls
        
    def send(self, alert: Alert) -> bool:
        """Send alert via email."""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            
            # Create HTML content
            html_content = self._create_html_content(alert)
            
            # Attach HTML
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _create_html_content(self, alert: Alert) -> str:
        """Create HTML email content."""
        template = """
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="background-color: {{ color }}; color: white; padding: 20px;">
                <h2>{{ alert.title }}</h2>
            </div>
            <div style="padding: 20px;">
                <p><strong>Severity:</strong> {{ alert.severity.value }}</p>
                <p><strong>Time:</strong> {{ alert.timestamp }}</p>
                <p><strong>Source:</strong> {{ alert.source }}</p>
                
                <h3>Message</h3>
                <p>{{ alert.message }}</p>
                
                {% if alert.metrics %}
                <h3>Metrics</h3>
                <table border="1" cellpadding="5">
                    {% for key, value in alert.metrics.items() %}
                    <tr>
                        <td>{{ key }}</td>
                        <td>{{ value }}</td>
                    </tr>
                    {% endfor %}
                </table>
                {% endif %}
                
                {% if alert.tags %}
                <p><strong>Tags:</strong> {{ ', '.join(alert.tags) }}</p>
                {% endif %}
            </div>
        </body>
        </html>
        """
        
        # Choose color based on severity
        colors = {
            AlertSeverity.INFO: '#17a2b8',
            AlertSeverity.WARNING: '#ffc107',
            AlertSeverity.ERROR: '#dc3545',
            AlertSeverity.CRITICAL: '#721c24'
        }
        
        tmpl = Template(template)
        return tmpl.render(alert=alert, color=colors.get(alert.severity, '#6c757d'))


class SlackChannel:
    """Slack alert channel."""
    
    def __init__(self, webhook_url: str):
        """Initialize Slack channel."""
        self.webhook_url = webhook_url
        
    def send(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        try:
            # Create Slack message
            payload = self._create_slack_payload(alert)
            
            # Send to Slack
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def _create_slack_payload(self, alert: Alert) -> Dict:
        """Create Slack message payload."""
        # Emoji based on severity
        emojis = {
            AlertSeverity.INFO: ":information_source:",
            AlertSeverity.WARNING: ":warning:",
            AlertSeverity.ERROR: ":x:",
            AlertSeverity.CRITICAL: ":rotating_light:"
        }
        
        # Color based on severity
        colors = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9900",
            AlertSeverity.ERROR: "#ff0000",
            AlertSeverity.CRITICAL: "#990000"
        }
        
        # Build fields
        fields = [
            {
                "title": "Source",
                "value": alert.source,
                "short": True
            },
            {
                "title": "Time",
                "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "short": True
            }
        ]
        
        # Add metrics as fields
        for key, value in alert.metrics.items():
            fields.append({
                "title": key,
                "value": str(value),
                "short": True
            })
        
        return {
            "attachments": [
                {
                    "fallback": f"{alert.severity.value.upper()}: {alert.title}",
                    "color": colors.get(alert.severity, "#808080"),
                    "pretext": f"{emojis.get(alert.severity, ':bell:')} *{alert.severity.value.upper()} Alert*",
                    "title": alert.title,
                    "text": alert.message,
                    "fields": fields,
                    "footer": "Data Quality Monitor",
                    "ts": int(alert.timestamp.timestamp())
                }
            ]
        }


class DiscordChannel:
    """Discord alert channel."""
    
    def __init__(self, webhook_url: str):
        """Initialize Discord channel."""
        self.webhook_url = webhook_url
        
    def send(self, alert: Alert) -> bool:
        """Send alert to Discord."""
        try:
            # Create Discord embed
            embed = self._create_discord_embed(alert)
            
            # Send to Discord
            response = requests.post(self.webhook_url, json={"embeds": [embed]})
            response.raise_for_status()
            
            logger.info(f"Discord alert sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False
    
    def _create_discord_embed(self, alert: Alert) -> Dict:
        """Create Discord embed."""
        # Color based on severity
        colors = {
            AlertSeverity.INFO: 0x17a2b8,
            AlertSeverity.WARNING: 0xffc107,
            AlertSeverity.ERROR: 0xdc3545,
            AlertSeverity.CRITICAL: 0x721c24
        }
        
        embed = {
            "title": f"{alert.severity.value.upper()}: {alert.title}",
            "description": alert.message,
            "color": colors.get(alert.severity, 0x808080),
            "timestamp": alert.timestamp.isoformat(),
            "footer": {
                "text": f"Source: {alert.source}"
            },
            "fields": []
        }
        
        # Add metrics as fields
        for key, value in alert.metrics.items():
            embed["fields"].append({
                "name": key,
                "value": str(value),
                "inline": True
            })
        
        return embed


class AlertManager:
    """Central alert management system."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize alert manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.channels = {}
        self.deduplicator = AlertDeduplicator()
        self.throttler = AlertThrottler()
        self.alert_history = deque(maxlen=1000)
        
        # Escalation rules
        self.escalation_rules = {
            AlertSeverity.INFO: [],
            AlertSeverity.WARNING: [AlertChannel.LOG, AlertChannel.SLACK],
            AlertSeverity.ERROR: [AlertChannel.SLACK, AlertChannel.EMAIL],
            AlertSeverity.CRITICAL: [AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.PAGERDUTY]
        }
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: Path):
        """Load configuration from file."""
        with open(config_path) as f:
            config = json.load(f)
        
        # Configure channels
        if 'email' in config:
            self.add_email_channel(**config['email'])
        
        if 'slack' in config:
            self.add_slack_channel(config['slack']['webhook_url'])
        
        if 'discord' in config:
            self.add_discord_channel(config['discord']['webhook_url'])
        
        # Configure escalation rules
        if 'escalation' in config:
            for severity, channels in config['escalation'].items():
                self.escalation_rules[AlertSeverity[severity.upper()]] = [
                    AlertChannel[ch.upper()] for ch in channels
                ]
    
    def add_email_channel(self, **kwargs):
        """Add email channel."""
        self.channels[AlertChannel.EMAIL] = EmailChannel(**kwargs)
    
    def add_slack_channel(self, webhook_url: str):
        """Add Slack channel."""
        self.channels[AlertChannel.SLACK] = SlackChannel(webhook_url)
    
    def add_discord_channel(self, webhook_url: str):
        """Add Discord channel."""
        self.channels[AlertChannel.DISCORD] = DiscordChannel(webhook_url)
    
    def send_alert(self, alert: Alert) -> bool:
        """
        Send alert through appropriate channels.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if alert was sent successfully
        """
        # Check for duplicates
        if self.deduplicator.is_duplicate(alert):
            logger.info(f"Alert deduplicated: {alert.title}")
            return False
        
        # Check throttling
        if not self.throttler.should_send(alert.timestamp):
            logger.warning(f"Alert throttled: {alert.title}")
            return False
        
        # Store in history
        self.alert_history.append(alert)
        
        # Get channels for severity
        channels_to_use = self.escalation_rules.get(alert.severity, [])
        
        # Send to each channel
        success = False
        for channel_type in channels_to_use:
            if channel_type == AlertChannel.LOG:
                self._log_alert(alert)
                success = True
            elif channel_type in self.channels:
                try:
                    if self.channels[channel_type].send(alert):
                        success = True
                except Exception as e:
                    logger.error(f"Failed to send alert via {channel_type}: {e}")
        
        return success
    
    def _log_alert(self, alert: Alert):
        """Log alert to file/console."""
        log_levels = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }
        
        level = log_levels.get(alert.severity, logging.INFO)
        logger.log(level, f"ALERT: {alert.title} - {alert.message}")
    
    def create_alert(self,
                    severity: Union[AlertSeverity, str],
                    title: str,
                    message: str,
                    source: str = "DataQualityMonitor",
                    metrics: Optional[Dict] = None,
                    tags: Optional[List[str]] = None) -> Alert:
        """
        Create and send an alert.
        
        Args:
            severity: Alert severity
            title: Alert title
            message: Alert message
            source: Alert source
            metrics: Associated metrics
            tags: Alert tags
            
        Returns:
            Created alert
        """
        # Convert severity if string
        if isinstance(severity, str):
            severity = AlertSeverity[severity.upper()]
        
        # Generate alert ID
        alert_id = hashlib.md5(
            f"{datetime.now().isoformat()}{title}{source}".encode()
        ).hexdigest()[:8]
        
        # Create alert
        alert = Alert(
            id=alert_id,
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.now(),
            source=source,
            metrics=metrics or {},
            tags=tags or []
        )
        
        # Send alert
        self.send_alert(alert)
        
        return alert
    
    def get_alert_history(self, 
                         limit: int = 100,
                         severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get alert history."""
        alerts = list(self.alert_history)
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return alerts[-limit:]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        if not self.alert_history:
            return {}
        
        # Count by severity
        severity_counts = {}
        for alert in self.alert_history:
            severity_counts[alert.severity.value] = severity_counts.get(
                alert.severity.value, 0
            ) + 1
        
        # Recent activity
        now = datetime.now()
        last_hour = [a for a in self.alert_history 
                     if (now - a.timestamp).total_seconds() < 3600]
        last_day = [a for a in self.alert_history 
                    if (now - a.timestamp).total_seconds() < 86400]
        
        return {
            'total_alerts': len(self.alert_history),
            'by_severity': severity_counts,
            'last_hour': len(last_hour),
            'last_day': len(last_day),
            'oldest_alert': self.alert_history[0].timestamp.isoformat() if self.alert_history else None,
            'newest_alert': self.alert_history[-1].timestamp.isoformat() if self.alert_history else None
        }


# Singleton instance
_alert_manager = None


def get_alert_manager(config_path: Optional[Path] = None) -> AlertManager:
    """Get or create alert manager singleton."""
    global _alert_manager
    
    if _alert_manager is None:
        _alert_manager = AlertManager(config_path)
    
    return _alert_manager


def send_alert(severity: Union[AlertSeverity, str],
              title: str,
              message: str,
              **kwargs) -> Alert:
    """Convenience function to send alert."""
    manager = get_alert_manager()
    return manager.create_alert(severity, title, message, **kwargs)