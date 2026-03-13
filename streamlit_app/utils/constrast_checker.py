"""
WCAG Color Contrast Checker
"""
from typing import Tuple


class ContrastChecker:
    """
    Check WCAG 2.1 color contrast ratios
    """
    
    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    @staticmethod
    def relative_luminance(rgb: Tuple[int, int, int]) -> float:
        """Calculate relative luminance"""
        r, g, b = [x / 255.0 for x in rgb]
        
        def adjust(color):
            return color / 12.92 if color <= 0.03928 else ((color + 0.055) / 1.055) ** 2.4
        
        r, g, b = adjust(r), adjust(g), adjust(b)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    @staticmethod
    def contrast_ratio(color1: str, color2: str) -> float:
        """
        Calculate contrast ratio between two colors
        
        Args:
            color1: Hex color (e.g., "#60a5fa")
            color2: Hex color (e.g., "#0f172a")
            
        Returns:
            Contrast ratio (1:1 to 21:1)
        """
        rgb1 = ContrastChecker.hex_to_rgb(color1)
        rgb2 = ContrastChecker.hex_to_rgb(color2)
        
        lum1 = ContrastChecker.relative_luminance(rgb1)
        lum2 = ContrastChecker.relative_luminance(rgb2)
        
        lighter = max(lum1, lum2)
        darker = min(lum1, lum2)
        
        return (lighter + 0.05) / (darker + 0.05)
    
    @staticmethod
    def meets_wcag_aa(foreground: str, background: str, large_text: bool = False) -> bool:
        """
        Check if color combination meets WCAG AA
        
        Args:
            foreground: Foreground color hex
            background: Background color hex
            large_text: True for large text (18pt+ or 14pt+ bold)
            
        Returns:
            True if meets WCAG AA (4.5:1 normal, 3:1 large)
        """
        ratio = ContrastChecker.contrast_ratio(foreground, background)
        required_ratio = 3.0 if large_text else 4.5
        return ratio >= required_ratio
    
    @staticmethod
    def meets_wcag_aaa(foreground: str, background: str, large_text: bool = False) -> bool:
        """
        Check if color combination meets WCAG AAA
        
        Args:
            foreground: Foreground color hex
            background: Background color hex
            large_text: True for large text
            
        Returns:
            True if meets WCAG AAA (7:1 normal, 4.5:1 large)
        """
        ratio = ContrastChecker.contrast_ratio(foreground, background)
        required_ratio = 4.5 if large_text else 7.0
        return ratio >= required_ratio


# Test our colors
if __name__ == "__main__":
    checker = ContrastChecker()
    
    # Test CareerLens AI colors
    colors = {
        "Primary text on dark bg": ("#e2e8f0", "#0f172a"),
        "Link color on dark bg": ("#60a5fa", "#0f172a"),
        "Button text on blue": ("#ffffff", "#3b82f6"),
        "Error text": ("#fca5a5", "#0f172a"),
        "Success text": ("#86efac", "#0f172a"),
    }
    
    print("WCAG Contrast Check:")
    print("=" * 60)
    
    for name, (fg, bg) in colors.items():
        ratio = checker.contrast_ratio(fg, bg)
        aa = checker.meets_wcag_aa(fg, bg)
        aaa = checker.meets_wcag_aaa(fg, bg)
        
        print(f"{name}:")
        print(f"  Ratio: {ratio:.2f}:1")
        print(f"  WCAG AA: {'✅ PASS' if aa else '❌ FAIL'}")
        print(f"  WCAG AAA: {'✅ PASS' if aaa else '❌ FAIL'}")
        print()