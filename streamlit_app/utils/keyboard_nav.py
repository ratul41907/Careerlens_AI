"""
Keyboard Navigation Enhancement
"""
import streamlit as st


def inject_keyboard_navigation():
    """Add keyboard navigation support"""
    st.markdown("""
    <script>
    // Keyboard navigation - Day 26
    document.addEventListener('DOMContentLoaded', function() {
        // Tab trap for modals
        const modals = document.querySelectorAll('[role="dialog"]');
        
        modals.forEach(modal => {
            const focusableElements = modal.querySelectorAll(
                'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
            );
            
            if (focusableElements.length > 0) {
                const firstElement = focusableElements[0];
                const lastElement = focusableElements[focusableElements.length - 1];
                
                modal.addEventListener('keydown', function(e) {
                    if (e.key === 'Tab') {
                        if (e.shiftKey) {
                            if (document.activeElement === firstElement) {
                                lastElement.focus();
                                e.preventDefault();
                            }
                        } else {
                            if (document.activeElement === lastElement) {
                                firstElement.focus();
                                e.preventDefault();
                            }
                        }
                    }
                    
                    // ESC to close
                    if (e.key === 'Escape') {
                        modal.querySelector('[aria-label="Close"]')?.click();
                    }
                });
            }
        });
        
        // Skip link functionality
        const skipLink = document.querySelector('.skip-link');
        if (skipLink) {
            skipLink.addEventListener('click', function(e) {
                e.preventDefault();
                const target = document.querySelector('#main-content');
                if (target) {
                    target.setAttribute('tabindex', '-1');
                    target.focus();
                }
            });
        }
    });
    </script>
    """, unsafe_allow_html=True)