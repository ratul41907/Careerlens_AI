"""
Confirmation dialog utilities
"""
import streamlit as st


def confirm_action(action_name: str, message: str, key: str) -> bool:
    """
    Show confirmation dialog for destructive actions
    
    Args:
        action_name: Name of action (e.g., "Clear All Data")
        message: Confirmation message
        key: Unique key for session state
    
    Returns:
        True if confirmed, False otherwise
    """
    confirm_key = f"confirm_{key}"
    
    if confirm_key not in st.session_state:
        st.session_state[confirm_key] = False
    
    if not st.session_state[confirm_key]:
        st.warning(f"⚠️ {message}")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button(f"✅ Yes, {action_name}", key=f"yes_{key}"):
                st.session_state[confirm_key] = True
                st.rerun()
        with col2:
            if st.button(f"❌ Cancel", key=f"no_{key}"):
                return False
        
        return False
    else:
        # Reset confirmation state
        st.session_state[confirm_key] = False
        return True


def show_data_warning(data_size: int):
    """Show warning about data size"""
    if data_size > 1000000:  # 1MB
        st.warning(f"""
        ⚠️ **Large Data Warning**
        
        File size: {data_size / 1000000:.1f}MB
        
        Processing may take longer than usual.
        Consider using a smaller file or text paste.
        """)