"""
Test Input Validation
"""
import pytest
from src.security.input_validator import InputValidator


class TestInputValidator:
    """Test input validation"""
    
    def test_valid_email(self):
        """Test valid email validation"""
        email = InputValidator.validate_email("test@example.com")
        assert email == "test@example.com"
    
    def test_invalid_email(self):
        """Test invalid email detection"""
        with pytest.raises(ValueError):
            InputValidator.validate_email("not-an-email")
    
    def test_email_with_xss(self):
        """Test XSS in email"""
        with pytest.raises(ValueError):
            InputValidator.validate_email("<script>alert('xss')</script>@test.com")
    
    def test_valid_phone(self):
        """Test valid phone validation"""
        phone = InputValidator.validate_phone("+1-234-567-8900")
        assert "234" in phone
    
    def test_invalid_phone(self):
        """Test invalid phone detection"""
        with pytest.raises(ValueError):
            InputValidator.validate_phone("123")  # Too short
    
    def test_sanitize_text(self):
        """Test text sanitization"""
        text = InputValidator.sanitize_text("Hello <b>World</b>")
        assert "<b>" not in text
        assert "Hello" in text
    
    def test_dangerous_script(self):
        """Test script tag detection"""
        with pytest.raises(ValueError):
            InputValidator.sanitize_text("<script>alert('xss')</script>")
    
    def test_valid_url(self):
        """Test valid URL validation"""
        url = InputValidator.validate_url("https://example.com")
        assert url == "https://example.com"
    
    def test_javascript_url(self):
        """Test JavaScript URL blocking"""
        with pytest.raises(ValueError):
            InputValidator.validate_url("javascript:alert('xss')")
    
    def test_valid_skill(self):
        """Test valid skill validation"""
        skill = InputValidator.validate_skill("Python 3.9")
        assert skill == "Python 3.9"
    
    def test_skill_with_special_chars(self):
        """Test skill with allowed special characters"""
        skill = InputValidator.validate_skill("C++")
        assert skill == "C++"
        
        skill = InputValidator.validate_skill("ASP.NET")
        assert skill == "ASP.NET"
    
    def test_long_text_rejection(self):
        """Test text length limits"""
        long_text = "A" * 100000
        with pytest.raises(ValueError):
            InputValidator.sanitize_text(long_text, input_type='text_short')
    
    def test_validate_skills_list(self):
        """Test skills list validation"""
        skills = ["Python", "JavaScript", "Docker"]
        validated = InputValidator.validate_skills_list(skills)
        assert len(validated) == 3
        assert "Python" in validated
    
    def test_duplicate_skills_removal(self):
        """Test duplicate removal"""
        skills = ["Python", "python", "PYTHON", "Java"]
        validated = InputValidator.validate_skills_list(skills)
        assert len(validated) == 2  # Python once, Java once
    
    def test_personal_info_validation(self):
        """Test personal info validation"""
        personal_info = {
            'name': 'John Doe',
            'email': 'john@example.com',
            'phone': '+1-234-567-8900'
        }
        validated = InputValidator.validate_personal_info(personal_info)
        assert validated['name'] == 'John Doe'
        assert validated['email'] == 'john@example.com'
    
    def test_path_traversal_attempt(self):
        """Test path traversal detection"""
        with pytest.raises(ValueError):
            InputValidator.sanitize_text("../../etc/passwd")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])