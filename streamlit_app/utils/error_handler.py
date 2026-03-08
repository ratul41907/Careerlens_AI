"""
Error handling utilities for better user experience
"""
import streamlit as st
import traceback
from typing import Optional


class ErrorHandler:
    """Centralized error handling for the application"""
    
    @staticmethod
    def handle_file_upload_error(e: Exception, file_type: str = "file"):
        """Handle file upload errors"""
        st.error(f"❌ **Error uploading {file_type}**")
        
        error_msg = str(e).lower()
        
        if "pdf" in error_msg:
            st.warning("""
            **PDF Upload Issue:**
            - Ensure PDF is not password-protected
            - Check file is not corrupted
            - Try converting to DOCX or use text paste
            """)
        elif "docx" in error_msg or "word" in error_msg:
            st.warning("""
            **DOCX Upload Issue:**
            - Ensure file is a valid Word document
            - Check file is not corrupted
            - Try saving a new copy
            """)
        elif "size" in error_msg or "large" in error_msg:
            st.warning("""
            **File Size Issue:**
            - File may be too large (limit: 10MB)
            - Try compressing the file
            - Use text paste instead
            """)
        else:
            st.warning(f"""
            **Unexpected Error:**
            {str(e)}
            
            **Try:**
            - Using a different file format
            - Pasting text instead of uploading
            - Restarting the application
            """)
    
    @staticmethod
    def handle_parsing_error(e: Exception, doc_type: str = "document"):
        """Handle document parsing errors"""
        st.error(f"❌ **Error parsing {doc_type}**")
        
        st.warning(f"""
        **Parsing Failed:**
        The {doc_type} could not be parsed correctly.
        
        **Common causes:**
        - File is corrupted or in unsupported format
        - File contains only images (no text)
        - File encoding is not UTF-8
        
        **Solutions:**
        - Try converting to plain text format
        - Copy and paste content manually
        - Ensure file contains extractable text
        
        **Technical details:** {str(e)}
        """)
    
    @staticmethod
    def handle_generation_error(e: Exception, output_type: str = "output"):
        """Handle generation errors"""
        st.error(f"❌ **Error generating {output_type}**")
        
        st.warning(f"""
        **Generation Failed:**
        
        **Possible causes:**
        - Missing required information
        - Invalid data format
        - System resource limitations
        
        **Solutions:**
        - Check all required fields are filled
        - Try with simpler input
        - Restart the application
        
        **Technical details:** {str(e)}
        """)
    
    @staticmethod
    def handle_model_error(e: Exception):
        """Handle ML model errors"""
        st.error("❌ **AI Model Error**")
        
        st.warning(f"""
        **Model Processing Failed:**
        
        **Common causes:**
        - Models not downloaded
        - Insufficient memory
        - Input text too long
        
        **Solutions:**
        1. Restart the application
        2. Check system has 8GB+ RAM
        3. Try with shorter input text
        4. Reinstall dependencies:
```
           pip install -r requirements.txt --force-reinstall
```
        
        **Technical details:** {str(e)}
        """)
    
    @staticmethod
    def show_debug_info(e: Exception):
        """Show detailed debug information (for developers)"""
        with st.expander("🔧 Debug Information (for developers)"):
            st.code(f"""
Exception Type: {type(e).__name__}
Exception Message: {str(e)}

Traceback:
{traceback.format_exc()}
            """)
    
    @staticmethod
    def safe_execute(func, error_handler=None, *args, **kwargs):
        """
        Execute function with error handling
        
        Args:
            func: Function to execute
            error_handler: Custom error handler function
            *args, **kwargs: Arguments for func
        
        Returns:
            Function result or None on error
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if error_handler:
                error_handler(e)
            else:
                st.error(f"❌ An error occurred: {str(e)}")
            ErrorHandler.show_debug_info(e)
            return None


# Convenience functions
def show_success(message: str, icon: str = "✅"):
    """Show success message with icon"""
    st.success(f"{icon} {message}")

def show_info(message: str, icon: str = "ℹ️"):
    """Show info message with icon"""
    st.info(f"{icon} {message}")

def show_warning(message: str, icon: str = "⚠️"):
    """Show warning message with icon"""
    st.warning(f"{icon} {message}")

def show_error(message: str, icon: str = "❌"):
    """Show error message with icon"""
    st.error(f"{icon} {message}")