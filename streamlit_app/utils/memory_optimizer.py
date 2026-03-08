"""
Memory optimization utilities
"""
import gc
import sys
from typing import Any


class MemoryOptimizer:
    """Optimize memory usage in Streamlit app"""
    
    @staticmethod
    def cleanup_large_objects(*objects):
        """Delete large objects and force garbage collection"""
        for obj in objects:
            try:
                del obj
            except:
                pass
        gc.collect()
    
    @staticmethod
    def get_object_size(obj: Any) -> int:
        """Get size of object in bytes"""
        return sys.getsizeof(obj)
    
    @staticmethod
    def get_session_state_size() -> dict:
        """Get size of session state objects"""
        import streamlit as st
        
        sizes = {}
        total_size = 0
        
        for key, value in st.session_state.items():
            size = sys.getsizeof(value)
            sizes[key] = size
            total_size += size
        
        return {
            'items': sizes,
            'total': total_size,
            'total_mb': total_size / (1024 * 1024)
        }
    
    @staticmethod
    def cleanup_session_state(keys_to_remove: list = None):
        """Remove specific keys from session state"""
        import streamlit as st
        
        if keys_to_remove is None:
            # Remove large temporary objects
            keys_to_remove = [
                'temp_cv_data',
                'temp_jd_data',
                'temp_embeddings'
            ]
        
        for key in keys_to_remove:
            if key in st.session_state:
                del st.session_state[key]
        
        gc.collect()
    
    @staticmethod
    def optimize_dataframe(df):
        """Optimize pandas DataFrame memory usage"""
        import pandas as pd
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > -128 and c_max < 127:
                        df[col] = df[col].astype('int8')
                    elif c_min > -32768 and c_max < 32767:
                        df[col] = df[col].astype('int16')
                    elif c_min > -2147483648 and c_max < 2147483647:
                        df[col] = df[col].astype('int32')
                
                elif str(col_type)[:5] == 'float':
                    df[col] = df[col].astype('float32')
        
        return df


def show_memory_usage():
    """Display current memory usage (for debugging)"""
    import streamlit as st
    
    optimizer = MemoryOptimizer()
    state_info = optimizer.get_session_state_size()
    
    with st.expander("🔍 Memory Usage (Debug)"):
        st.write(f"**Total Session State:** {state_info['total_mb']:.2f} MB")
        st.write("**By Item:**")
        for key, size in state_info['items'].items():
            st.write(f"- `{key}`: {size / 1024:.2f} KB")