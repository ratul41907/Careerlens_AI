"""
File Security Checker - Secure file upload handling
Validates file types, sizes, and content
"""
import os
import magic
import hashlib
from typing import Optional, List, Tuple
from pathlib import Path


class FileSecurityChecker:
    """
    Check uploaded files for security issues
    """
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {
        'pdf': 'application/pdf',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'doc': 'application/msword',
        'txt': 'text/plain',
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg'
    }
    
    # Maximum file sizes (in bytes)
    MAX_FILE_SIZES = {
        'pdf': 10 * 1024 * 1024,      # 10MB
        'docx': 5 * 1024 * 1024,       # 5MB
        'doc': 5 * 1024 * 1024,        # 5MB
        'txt': 1 * 1024 * 1024,        # 1MB
        'png': 5 * 1024 * 1024,        # 5MB
        'jpg': 5 * 1024 * 1024,        # 5MB
        'jpeg': 5 * 1024 * 1024        # 5MB
    }
    
    # Dangerous file signatures (magic bytes)
    DANGEROUS_SIGNATURES = [
        b'MZ',              # Windows executable
        b'\x7fELF',         # Linux executable
        b'<?php',           # PHP script
        b'#!/',             # Shell script
        b'<script',         # JavaScript
    ]
    
    @classmethod
    def validate_file(
        cls,
        file_path: str,
        allowed_extensions: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """
        Validate uploaded file
        
        Args:
            file_path: Path to uploaded file
            allowed_extensions: List of allowed extensions (optional)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file exists
            if not os.path.exists(file_path):
                return False, "File does not exist"
            
            # Get file extension
            file_ext = Path(file_path).suffix.lower().lstrip('.')
            
            # Check if extension is allowed
            allowed = allowed_extensions or list(cls.ALLOWED_EXTENSIONS.keys())
            if file_ext not in allowed:
                return False, f"File type '.{file_ext}' not allowed. Allowed: {', '.join(allowed)}"
            
            # Check file size
            file_size = os.path.getsize(file_path)
            max_size = cls.MAX_FILE_SIZES.get(file_ext, 5 * 1024 * 1024)
            
            if file_size > max_size:
                max_mb = max_size / (1024 * 1024)
                return False, f"File too large. Maximum size: {max_mb}MB"
            
            if file_size == 0:
                return False, "File is empty"
            
            # Check MIME type (requires python-magic)
            try:
                mime = magic.Magic(mime=True)
                detected_mime = mime.from_file(file_path)
                expected_mime = cls.ALLOWED_EXTENSIONS.get(file_ext)
                
                # Some flexibility for MIME types
                if expected_mime and not cls._mime_matches(detected_mime, expected_mime):
                    return False, f"File content doesn't match extension. Expected: {expected_mime}, Got: {detected_mime}"
            except:
                # python-magic not available, skip MIME check
                pass
            
            # Check for dangerous content
            is_safe, danger_msg = cls._check_dangerous_content(file_path)
            if not is_safe:
                return False, danger_msg
            
            return True, "File is safe"
            
        except Exception as e:
            return False, f"File validation error: {str(e)}"
    
    @classmethod
    def _mime_matches(cls, detected: str, expected: str) -> bool:
        """Check if MIME types match (with some flexibility)"""
        # Exact match
        if detected == expected:
            return True
        
        # Allow text/plain for various text files
        if expected == 'text/plain' and detected.startswith('text/'):
            return True
        
        # Allow various image formats
        if expected.startswith('image/') and detected.startswith('image/'):
            return True
        
        return False
    
    @classmethod
    def _check_dangerous_content(cls, file_path: str) -> Tuple[bool, str]:
        """
        Check file for dangerous content
        
        Args:
            file_path: Path to file
            
        Returns:
            Tuple of (is_safe, error_message)
        """
        try:
            # Read first 1KB to check signatures
            with open(file_path, 'rb') as f:
                header = f.read(1024)
            
            # Check for dangerous signatures
            for signature in cls.DANGEROUS_SIGNATURES:
                if signature in header:
                    return False, "File contains potentially dangerous content"
            
            # Check for embedded executables in documents
            if b'MZ' in header[100:]:  # Executable embedded after header
                return False, "File may contain embedded executable"
            
            return True, "Content is safe"
            
        except Exception as e:
            return False, f"Content check error: {str(e)}"
    
    @classmethod
    def get_file_hash(cls, file_path: str) -> str:
        """
        Calculate SHA-256 hash of file
        
        Args:
            file_path: Path to file
            
        Returns:
            Hex string of file hash
        """
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            # Read file in chunks
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """
        Sanitize filename to prevent path traversal
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Get basename only (remove path)
        filename = os.path.basename(filename)
        
        # Remove dangerous characters
        filename = filename.replace('..', '')
        filename = filename.replace('/', '')
        filename = filename.replace('\\', '')
        filename = filename.replace('\x00', '')
        
        # Limit length
        max_length = 255
        if len(filename) > max_length:
            name, ext = os.path.splitext(filename)
            filename = name[:max_length - len(ext)] + ext
        
        return filename


# Test
if __name__ == "__main__":
    checker = FileSecurityChecker()
    
    # Test filename sanitization
    dangerous_name = "../../etc/passwd"
    safe_name = checker.sanitize_filename(dangerous_name)
    print(f"✅ Sanitized: '{dangerous_name}' → '{safe_name}'")
    
    # Test file validation (if you have a test file)
    # is_valid, message = checker.validate_file("test.pdf")
    # print(f"Validation: {message}")