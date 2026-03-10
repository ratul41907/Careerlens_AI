"""
Eligibility Validator - Validates academic credentials using OCR
"""
import pytesseract
from PIL import Image
import PyPDF2
import re
from typing import Dict, List, Optional, Tuple
from loguru import logger
import os


class EligibilityValidator:
    """
    Validate academic eligibility using OCR on transcripts
    """
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """
        Initialize eligibility validator
        
        Args:
            tesseract_path: Path to tesseract executable (auto-detect if None)
        """
        # Set Tesseract path
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            # Try common locations
            common_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                '/usr/bin/tesseract',
                '/usr/local/bin/tesseract'
            ]
            
            for path in common_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    logger.info(f"Tesseract found at: {path}")
                    break
        
        # Degree requirements (example - can be customized)
        self.degree_requirements = {
            'bachelor': {'min_gpa': 2.5, 'years': 4},
            'master': {'min_gpa': 3.0, 'years': 2},
            'phd': {'min_gpa': 3.5, 'years': 4}
        }
        
        logger.info("EligibilityValidator initialized")
    
    def validate_eligibility(self, jd_data: Dict, academic_text: str) -> Tuple[bool, str]:
        """
        Validate eligibility based on JD requirements and academic documents
        
        Args:
            jd_data: Parsed job description data
            academic_text: Extracted text from academic documents
            
        Returns:
            Tuple of (is_eligible: bool, reason: str)
        """
        try:
            # Extract requirements from JD
            required_degree = self._extract_degree_requirement(jd_data)
            required_gpa = self._extract_gpa_requirement(jd_data)
            
            # Parse academic text
            academic_data = self._parse_transcript(academic_text)
            
            failures = []
            
            # Check degree
            if required_degree:
                if academic_data['degree']:
                    degree_hierarchy = {'bachelor': 1, 'master': 2, 'phd': 3}
                    candidate_level = degree_hierarchy.get(academic_data['degree'], 0)
                    required_level = degree_hierarchy.get(required_degree.lower(), 0)
                    
                    if candidate_level < required_level:
                        failures.append(f"Degree level insufficient (has {academic_data['degree']}, requires {required_degree})")
                else:
                    failures.append("Could not verify degree from documents")
            
            # Check GPA
            if required_gpa:
                if academic_data['gpa'] and academic_data['gpa_scale']:
                    # Normalize to 4.0 scale
                    normalized_gpa = (academic_data['gpa'] / academic_data['gpa_scale']) * 4.0
                    
                    if normalized_gpa < required_gpa:
                        failures.append(f"GPA below requirement ({normalized_gpa:.2f} < {required_gpa})")
                else:
                    failures.append("Could not verify GPA from documents")
            
            # Make decision
            if failures:
                return False, "; ".join(failures)
            else:
                details = []
                if academic_data['degree']:
                    details.append(f"Degree: {academic_data['degree']}")
                if academic_data['gpa']:
                    normalized = (academic_data['gpa'] / academic_data['gpa_scale']) * 4.0 if academic_data['gpa_scale'] else academic_data['gpa']
                    details.append(f"GPA: {normalized:.2f}/4.0")
                
                reason = "All requirements met" + (f" ({', '.join(details)})" if details else "")
                return True, reason
                
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False, f"Validation failed: {str(e)}"
    
    def _extract_degree_requirement(self, jd_data: Dict) -> Optional[str]:
        """Extract degree requirement from JD data"""
        # Check if JD has degree requirements
        jd_text = jd_data.get('text', '').lower()
        
        if 'phd' in jd_text or 'doctorate' in jd_text:
            return 'phd'
        elif 'master' in jd_text or "master's" in jd_text:
            return 'master'
        elif 'bachelor' in jd_text or "bachelor's" in jd_text:
            return 'bachelor'
        
        return None
    
    def _extract_gpa_requirement(self, jd_data: Dict) -> Optional[float]:
        """Extract GPA requirement from JD data"""
        jd_text = jd_data.get('text', '')
        
        # Pattern: "minimum GPA 3.0", "GPA of 3.5", etc.
        patterns = [
            r'minimum gpa[:\s]+(\d+\.\d+)',
            r'gpa[:\s]+(\d+\.\d+)\s+or higher',
            r'gpa of[:\s]+(\d+\.\d+)',
            r'(\d+\.\d+)\s+gpa required'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, jd_text.lower())
            if match:
                return float(match.group(1))
        
        return None
    
    def _extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            logger.debug(f"Extracted {len(text)} characters from image")
            return text
        except Exception as e:
            logger.error(f"OCR error: {str(e)}")
            return ""
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            logger.debug(f"Extracted {len(text)} characters from PDF")
            return text
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            return ""
    
    def validate_transcript(self,
                           image_path: str,
                           job_requirements: Optional[Dict] = None) -> Dict:
        """
        Validate transcript image against job requirements
        
        Args:
            image_path: Path to transcript image
            job_requirements: Dict with 'degree_level' and 'min_gpa'
            
        Returns:
            Dict with validation result, extracted data, and decision
        """
        logger.info(f"Validating transcript: {image_path}")
        
        try:
            # Extract text from image
            extracted_text = self._extract_text_from_image(image_path)
            
            if not extracted_text:
                return {
                    'success': False,
                    'error': 'Failed to extract text from image',
                    'decision': 'FAIL',
                    'reason': 'OCR extraction failed'
                }
            
            # Parse transcript data
            transcript_data = self._parse_transcript(extracted_text)
            
            # Validate against requirements
            if job_requirements:
                decision = self._make_decision(transcript_data, job_requirements)
            else:
                # Default validation
                decision = self._default_validation(transcript_data)
            
            result = {
                'success': True,
                'extracted_text': extracted_text[:500] + '...' if len(extracted_text) > 500 else extracted_text,
                'transcript_data': transcript_data,
                'decision': decision['decision'],
                'reason': decision['reason'],
                'confidence': decision['confidence'],
                'details': decision['details']
            }
            
            logger.info(f"Validation complete: {decision['decision']}")
            return result
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'decision': 'FAIL',
                'reason': 'Processing error'
            }
    
    def _parse_transcript(self, text: str) -> Dict:
        """
        Parse transcript text to extract key information
        
        Returns:
            Dict with degree, gpa, institution, graduation_year
        """
        data = {
            'degree': None,
            'gpa': None,
            'gpa_scale': None,
            'institution': None,
            'graduation_year': None,
            'major': None
        }
        
        # Extract degree
        data['degree'] = self._extract_degree(text)
        
        # Extract GPA
        gpa_info = self._extract_gpa(text)
        data['gpa'] = gpa_info['gpa']
        data['gpa_scale'] = gpa_info['scale']
        
        # Extract year
        data['graduation_year'] = self._extract_year(text)
        
        # Extract institution
        data['institution'] = self._extract_institution(text)
        
        # Extract major
        data['major'] = self._extract_major(text)
        
        return data
    
    def _extract_degree(self, text: str) -> Optional[str]:
        """Extract degree level from transcript"""
        text_lower = text.lower()
        
        # Patterns for degrees
        degree_patterns = {
            'phd': [r'ph\.?d', r'doctor of philosophy', r'doctorate'],
            'master': [r'master', r'm\.s\.', r'm\.a\.', r'msc', r'mba'],
            'bachelor': [r'bachelor', r'b\.s\.', r'b\.a\.', r'bsc', r'undergraduate']
        }
        
        for degree, patterns in degree_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return degree
        
        return None
    
    def _extract_gpa(self, text: str) -> Dict:
        """
        Extract GPA and scale
        
        Returns:
            Dict with 'gpa' and 'scale'
        """
        # Pattern: "GPA: 3.8/4.0" or "GPA 3.8 out of 4.0" or "CGPA: 8.5/10"
        patterns = [
            r'gpa[:\s]+(\d+\.\d+)\s*/\s*(\d+\.\d+)',  # GPA: 3.8/4.0
            r'gpa[:\s]+(\d+\.\d+)\s+out of\s+(\d+\.\d+)',  # GPA 3.8 out of 4.0
            r'cgpa[:\s]+(\d+\.\d+)\s*/\s*(\d+\.\d+)',  # CGPA: 8.5/10
            r'grade point average[:\s]+(\d+\.\d+)\s*/\s*(\d+\.\d+)'
        ]
        
        text_lower = text.lower()
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                gpa = float(match.group(1))
                scale = float(match.group(2))
                
                return {
                    'gpa': gpa,
                    'scale': scale
                }
        
        return {'gpa': None, 'scale': None}
    
    def _extract_year(self, text: str) -> Optional[int]:
        """Extract graduation year"""
        # Look for 4-digit years (2015-2030)
        years = re.findall(r'20[1-3][0-9]', text)
        
        if years:
            # Return most recent year
            return int(max(years))
        
        return None
    
    def _extract_institution(self, text: str) -> Optional[str]:
        """Extract institution name (simple heuristic)"""
        # Look for "University" or "Institute" or "College"
        lines = text.split('\n')
        
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if any(word in line.lower() for word in ['university', 'institute', 'college']):
                if len(line) < 100:  # Not too long
                    return line
        
        return None
    
    def _extract_major(self, text: str) -> Optional[str]:
        """Extract major/field of study"""
        text_lower = text.lower()
        
        # Common majors
        majors = [
            'computer science', 'electrical engineering', 'mechanical engineering',
            'mathematics', 'physics', 'chemistry', 'biology',
            'business administration', 'economics', 'information technology'
        ]
        
        for major in majors:
            if major in text_lower:
                return major.title()
        
        return None
    
    def _make_decision(self,
                      transcript_data: Dict,
                      job_requirements: Dict) -> Dict:
        """
        Make PASS/FAIL decision based on requirements
        
        Args:
            transcript_data: Extracted transcript data
            job_requirements: Job requirements dict
            
        Returns:
            Decision dict with reason and confidence
        """
        required_degree = job_requirements.get('degree_level', 'bachelor')
        required_gpa = job_requirements.get('min_gpa', 3.0)
        required_major = job_requirements.get('major')
        
        failures = []
        warnings = []
        confidence = 100
        
        # Check degree level
        if transcript_data['degree']:
            degree_hierarchy = {'bachelor': 1, 'master': 2, 'phd': 3}
            if degree_hierarchy.get(transcript_data['degree'], 0) < degree_hierarchy.get(required_degree, 0):
                failures.append(f"Degree level insufficient (has {transcript_data['degree']}, needs {required_degree})")
                confidence -= 40
        else:
            warnings.append("Could not extract degree level")
            confidence -= 20
        
        # Check GPA
        if transcript_data['gpa'] and transcript_data['gpa_scale']:
            # Normalize to 4.0 scale
            normalized_gpa = (transcript_data['gpa'] / transcript_data['gpa_scale']) * 4.0
            
            if normalized_gpa < required_gpa:
                failures.append(f"GPA below requirement ({normalized_gpa:.2f} < {required_gpa})")
                confidence -= 30
        else:
            warnings.append("Could not extract GPA")
            confidence -= 15
        
        # Check major (if specified)
        if required_major:
            if transcript_data['major']:
                if required_major.lower() not in transcript_data['major'].lower():
                    failures.append(f"Major mismatch (has {transcript_data['major']}, needs {required_major})")
                    confidence -= 20
            else:
                warnings.append("Could not extract major")
                confidence -= 10
        
        # Make decision
        if failures:
            decision = 'FAIL'
            reason = '; '.join(failures)
        elif warnings:
            decision = 'REVIEW'
            reason = 'Missing data: ' + '; '.join(warnings)
        else:
            decision = 'PASS'
            reason = 'All requirements met'
        
        return {
            'decision': decision,
            'reason': reason,
            'confidence': max(confidence, 0),
            'details': {
                'failures': failures,
                'warnings': warnings,
                'extracted_degree': transcript_data['degree'],
                'extracted_gpa': transcript_data['gpa'],
                'extracted_major': transcript_data['major']
            }
        }
    
    def _default_validation(self, transcript_data: Dict) -> Dict:
        """Default validation without specific requirements"""
        if not transcript_data['degree']:
            return {
                'decision': 'REVIEW',
                'reason': 'Could not extract degree information',
                'confidence': 30,
                'details': transcript_data
            }
        
        if transcript_data['gpa'] and transcript_data['gpa_scale']:
            normalized_gpa = (transcript_data['gpa'] / transcript_data['gpa_scale']) * 4.0
            
            if normalized_gpa >= 3.0:
                return {
                    'decision': 'PASS',
                    'reason': f'Good academic standing (GPA: {normalized_gpa:.2f}/4.0)',
                    'confidence': 85,
                    'details': transcript_data
                }
            else:
                return {
                    'decision': 'REVIEW',
                    'reason': f'Below average GPA ({normalized_gpa:.2f}/4.0)',
                    'confidence': 60,
                    'details': transcript_data
                }
        
        return {
            'decision': 'REVIEW',
            'reason': 'Insufficient data for validation',
            'confidence': 50,
            'details': transcript_data
        }


# Convenience function
def validate_transcript(image_path: str, job_requirements: Optional[Dict] = None) -> Dict:
    """Quick validation function"""
    validator = EligibilityValidator()
    return validator.validate_transcript(image_path, job_requirements)