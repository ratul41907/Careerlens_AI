"""
Test File Security
"""
import pytest
import tempfile
import os
from src.security.file_security import FileSecurityChecker


class TestFileSecurityChecker:
    """Test file security checks"""
    
    def test_sanitize_filename(self):
        """Test filename sanitization"""
        # Path traversal attempt
        filename = FileSecurityChecker.sanitize_filename("../../etc/passwd")
        assert ".." not in filename
        assert "/" not in filename
        
        # Normal filename
        filename = FileSecurityChecker.sanitize_filename("my_resume.pdf")
        assert filename == "my_resume.pdf"
    
    def test_validate_file_extension(self):
        """Test file extension validation"""
        # Create temp files
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b'%PDF-1.4\n')  # PDF header
            tmp_path = tmp.name
        
        try:
            is_valid, message = FileSecurityChecker.validate_file(tmp_path)
            # May fail MIME check but extension should be valid
            assert "not allowed" not in message.lower()
        finally:
            os.unlink(tmp_path)
    
    def test_reject_disallowed_extension(self):
        """Test rejection of disallowed extensions"""
        with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as tmp:
            tmp.write(b'MZ')  # EXE header
            tmp_path = tmp.name
        
        try:
            is_valid, message = FileSecurityChecker.validate_file(tmp_path)
            assert not is_valid
            assert "not allowed" in message.lower()
        finally:
            os.unlink(tmp_path)
    
    def test_reject_empty_file(self):
        """Test rejection of empty files"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp_path = tmp.name  # Empty file
        
        try:
            is_valid, message = FileSecurityChecker.validate_file(tmp_path)
            assert not is_valid
            assert "empty" in message.lower()
        finally:
            os.unlink(tmp_path)
    
    def test_reject_oversized_file(self):
        """Test rejection of oversized files"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            # Create file larger than 1MB (txt limit)
            tmp.write(b'A' * (2 * 1024 * 1024))
            tmp_path = tmp.name
        
        try:
            is_valid, message = FileSecurityChecker.validate_file(tmp_path)
            assert not is_valid
            assert "too large" in message.lower()
        finally:
            os.unlink(tmp_path)
    
    def test_get_file_hash(self):
        """Test file hash calculation"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b'Test content')
            tmp_path = tmp.name
        
        try:
            hash1 = FileSecurityChecker.get_file_hash(tmp_path)
            hash2 = FileSecurityChecker.get_file_hash(tmp_path)
            
            # Same file should have same hash
            assert hash1 == hash2
            assert len(hash1) == 64  # SHA-256 produces 64 hex chars
        finally:
            os.unlink(tmp_path)
    
    def test_detect_dangerous_content(self):
        """Test detection of dangerous content"""
        # Create file with executable signature
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b'MZ')  # Windows executable signature
            tmp_path = tmp.name
        
        try:
            is_safe, message = FileSecurityChecker._check_dangerous_content(tmp_path)
            assert not is_safe
            assert "dangerous" in message.lower()
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])