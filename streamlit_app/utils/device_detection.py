"""
Device Detection Utility
"""
import streamlit as st


def is_mobile():
    """
    Detect if user is on mobile device
    
    Returns:
        bool: True if mobile, False otherwise
    """
    try:
        # Check if user agent is available
        session_state = st.session_state
        
        # Mobile keywords in user agent
        mobile_keywords = [
            'mobile', 'android', 'iphone', 'ipad', 
            'ipod', 'blackberry', 'windows phone'
        ]
        
        # Try to get user agent from query params or session
        # Note: Streamlit doesn't expose user agent directly
        # This is a placeholder for future enhancement
        
        return False  # Default to desktop for now
    
    except Exception:
        return False


def get_device_class():
    """
    Get CSS class for current device
    
    Returns:
        str: 'mobile', 'tablet', or 'desktop'
    """
    if is_mobile():
        return 'mobile'
    return 'desktop'


def show_mobile_warning():
    """Show info message for mobile users"""
    st.info("""
    📱 **Mobile User?** 
    For the best experience, try rotating your device to landscape mode 
    or use a larger screen for data-heavy sections.
    """)