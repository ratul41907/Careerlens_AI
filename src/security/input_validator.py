"""
Input Validator - Sanitize and validate all user inputs
Prevents injection attacks, XSS, and malicious content
"""
import re
import html
from typing import Optional, Dict, List
import bleach


class InputValidator:
    """
    Validate and sanitize user inputs
    """
    
    # Maximum lengths for different input types
    MAX_LENGTHS = {
        'name': 100,
        'email': 254,
        'phone': 20,
        'text_short': 500,
        'text_medium': 2000,
        'text_long': 10000,
        'cv_text': 50000,
        'jd_text': 20000,
        'skill': 50,
        'url': 2048
    }
    
    # Allowed HTML tags for rich text (very restricted)
    ALLOWED_TAGS = ['p', 'br', 'strong', 'em', 'ul', 'ol', 'li']
    ALLOWED_ATTRIBUTES = {}
    
    # Dangerous patterns
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',                 # JavaScript protocol
        r'on\w+\s*=',                  # Event handlers
        r'eval\s*\(',                  # eval() calls
        r'expression\s*\(',            # CSS expressions
        r'import\s+',                  # Python imports in text
        r'__\w+__',                    # Python magic methods
        r'\.\./',                      # Path traversal
        r'\.\./\.\.',                  # Path traversal
    ]
    
    @classmethod
    def sanitize_text(cls, text: str, max_length: Optional[int] = None, 
                     input_type: str = 'text_medium') -> str:
        """
        Sanitize text input
        
        Args:
            text: Raw text input
            max_length: Maximum allowed length (overrides input_type)
            input_type: Type of input for length validation
            
        Returns:
            Sanitized text
            
        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Strip whitespace
        text = text.strip()
        
        # Check length
        max_len = max_length or cls.MAX_LENGTHS.get(input_type, cls.MAX_LENGTHS['text_medium'])
        if len(text) > max_len:
            raise ValueError(f"Input exceeds maximum length of {max_len} characters")
        
        # HTML escape (prevents XSS)
        text = html.escape(text)
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                raise ValueError(f"Input contains potentially dangerous content")
        
        return text
    
    @classmethod
    def sanitize_html(cls, html_text: str) -> str:
        """
        Sanitize HTML input (for rich text editors)
        
        Args:
            html_text: Raw HTML input
            
        Returns:
            Sanitized HTML
        """
        # Use bleach to sanitize HTML
        cleaned = bleach.clean(
            html_text,
            tags=cls.ALLOWED_TAGS,
            attributes=cls.ALLOWED_ATTRIBUTES,
            strip=True
        )
        
        return cleaned
    
    @classmethod
    def validate_email(cls, email: str) -> str:
        """
        Validate and sanitize email address
        
        Args:
            email: Email address
            
        Returns:
            Sanitized email
            
        Raises:
            ValueError: If email is invalid
        """
        email = email.strip().lower()
        
        # Check length
        if len(email) > cls.MAX_LENGTHS['email']:
            raise ValueError("Email address too long")
        
        # Basic email regex
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            raise ValueError("Invalid email format")
        
        # Check for dangerous characters
        if any(char in email for char in ['<', '>', '"', "'", '\\', '\x00']):
            raise ValueError("Email contains invalid characters")
        
        return email
    
    @classmethod
    def validate_phone(cls, phone: str) -> str:
        """
        Validate and sanitize phone number
        
        Args:
            phone: Phone number
            
        Returns:
            Sanitized phone number
            
        Raises:
            ValueError: If phone is invalid
        """
        # Remove common formatting
        phone = re.sub(r'[^\d+\-\(\)\s]', '', phone)
        phone = phone.strip()
        
        # Check length
        if len(phone) > cls.MAX_LENGTHS['phone']:
            raise ValueError("Phone number too long")
        
        # Must contain at least 7 digits
        digits = re.sub(r'\D', '', phone)
        if len(digits) < 7:
            raise ValueError("Phone number must contain at least 7 digits")
        
        return phone
    
    @classmethod
    def validate_url(cls, url: str) -> str:
        """
        Validate and sanitize URL
        
        Args:
            url: URL to validate
            
        Returns:
            Sanitized URL
            
        Raises:
            ValueError: If URL is invalid
        """
        url = url.strip()
        
        # Check length
        if len(url) > cls.MAX_LENGTHS['url']:
            raise ValueError("URL too long")
        
        # Must start with http:// or https://
        if not re.match(r'^https?://', url, re.IGNORECASE):
            raise ValueError("URL must start with http:// or https://")
        
        # Check for dangerous patterns
        if 'javascript:' in url.lower() or 'data:' in url.lower():
            raise ValueError("Invalid URL protocol")
        
        # Basic URL validation
        url_pattern = r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$'
        if not re.match(url_pattern, url, re.IGNORECASE):
            raise ValueError("Invalid URL format")
        
        return url
    
    @classmethod
    def validate_skill(cls, skill: str) -> str:
        """
        Validate skill name
        
        Args:
            skill: Skill name
            
        Returns:
            Sanitized skill name
            
        Raises:
            ValueError: If skill is invalid
        """
        skill = skill.strip()
        
        # Check length
        if len(skill) > cls.MAX_LENGTHS['skill']:
            raise ValueError("Skill name too long")
        
        # Must be alphanumeric with limited special chars
        if not re.match(r'^[a-zA-Z0-9\s\.\-\+\#\/]+$', skill):
            raise ValueError("Skill contains invalid characters")
        
        # Must not be empty
        if not skill:
            raise ValueError("Skill name cannot be empty")
        
        return skill
    
    @classmethod
    def validate_cv_text(cls, cv_text: str) -> str:
        """
        Validate CV text input
        
        Args:
            cv_text: CV text
            
        Returns:
            Sanitized CV text
            
        Raises:
            ValueError: If CV text is invalid
        """
        return cls.sanitize_text(cv_text, input_type='cv_text')
    
    @classmethod
    def validate_jd_text(cls, jd_text: str) -> str:
        """
        Validate job description text
        
        Args:
            jd_text: JD text
            
        Returns:
            Sanitized JD text
            
        Raises:
            ValueError: If JD text is invalid
        """
        return cls.sanitize_text(jd_text, input_type='jd_text')
    
    @classmethod
    def validate_personal_info(cls, personal_info: Dict) -> Dict:
        """
        Validate personal information dictionary
        
        Args:
            personal_info: Dictionary with personal information
            
        Returns:
            Sanitized personal info
            
        Raises:
            ValueError: If any field is invalid
        """
        validated = {}
        
        # Name
        if 'name' in personal_info:
            validated['name'] = cls.sanitize_text(
                personal_info['name'], 
                input_type='name'
            )
        
        # Email
        if 'email' in personal_info:
            validated['email'] = cls.validate_email(personal_info['email'])
        
        # Phone
        if 'phone' in personal_info:
            validated['phone'] = cls.validate_phone(personal_info['phone'])
        
        # Location
        if 'location' in personal_info:
            validated['location'] = cls.sanitize_text(
                personal_info['location'],
                input_type='text_short'
            )
        
        # URLs
        for url_field in ['linkedin', 'github', 'portfolio']:
            if url_field in personal_info and personal_info[url_field]:
                try:
                    validated[url_field] = cls.validate_url(personal_info[url_field])
                except ValueError:
                    # Optional fields - skip if invalid
                    pass
        
        # Summary
        if 'summary' in personal_info and personal_info['summary']:
            validated['summary'] = cls.sanitize_text(
                personal_info['summary'],
                input_type='text_long'
            )
        
        return validated
    
    @classmethod
    def validate_skills_list(cls, skills: List[str]) -> List[str]:
        """
        Validate list of skills
        
        Args:
            skills: List of skill names
            
        Returns:
            List of sanitized skills
            
        Raises:
            ValueError: If any skill is invalid
        """
        validated_skills = []
        
        for skill in skills:
            validated_skill = cls.validate_skill(skill)
            validated_skills.append(validated_skill)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_skills = []
        for skill in validated_skills:
            skill_lower = skill.lower()
            if skill_lower not in seen:
                seen.add(skill_lower)
                unique_skills.append(skill)
        
        return unique_skills


# Test
if __name__ == "__main__":
    validator = InputValidator()
    
    # Test email validation
    try:
        email = validator.validate_email("test@example.com")
        print(f"✅ Valid email: {email}")
    except ValueError as e:
        print(f"❌ Invalid email: {e}")
    
    # Test dangerous input
    try:
        dangerous = validator.sanitize_text("<script>alert('XSS')</script>")
        print(f"❌ Should have been blocked!")
    except ValueError:
        print(f"✅ Dangerous input blocked")
    
    # Test skill validation
    try:
        skill = validator.validate_skill("Python 3.9")
        print(f"✅ Valid skill: {skill}")
    except ValueError as e:
        print(f"❌ Invalid skill: {e}")