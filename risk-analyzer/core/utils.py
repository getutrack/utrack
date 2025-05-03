import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Clean and normalize text for processing."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def format_datetime(dt: Optional[datetime]) -> Optional[str]:
    """Format datetime to ISO format."""
    if not dt:
        return None
    return dt.isoformat()


def parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse datetime from ISO format."""
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str)
    except ValueError:
        logger.warning(f"Could not parse datetime: {dt_str}")
        return None


def get_date_range(days: int) -> Dict[str, str]:
    """Get start and end dates for a period."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat()
    }


def calculate_risk_level(factors: Dict[str, Any]) -> str:
    """Calculate overall risk level based on multiple factors."""
    # Simple algorithm: if any factor is HIGH, overall is HIGH
    # If majority are MEDIUM, overall is MEDIUM, otherwise LOW
    risk_values = [v for k, v in factors.items() if k.endswith('_risk')]
    
    if 'HIGH' in risk_values:
        return 'HIGH'
    elif risk_values.count('MEDIUM') > len(risk_values) / 2:
        return 'MEDIUM'
    else:
        return 'LOW'


def create_recommendations(
    risk_factors: Dict[str, str], 
    metrics: Dict[str, Any]
) -> List[str]:
    """Create recommendation list based on risk factors and metrics."""
    recommendations = []
    
    # Example recommendations based on capacity
    if risk_factors.get('capacity_risk') == 'HIGH':
        recommendations.append("Consider adding more team members")
    
    # Example recommendations based on timeline
    if risk_factors.get('timeline_risk') == 'HIGH':
        recommendations.append("Review project deadlines and task estimation")
    
    # Example recommendations based on transitions
    if metrics.get('state_changes', 0) > 50:
        recommendations.append("Review the state transition process")
    
    # Always include team analysis
    recommendations.append("Analyze issue distribution across team members")
    
    return [r for r in recommendations if r] 