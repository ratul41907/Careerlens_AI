"""
CV Parser - Extracts and cleans text from PDF and DOCX files
"""
import fitz  # PyMuPDF
from docx import Document
from pathlib import Path
import re
from typing import Dict, List, Optional
from loguru import logger


class CVParser:
    """Parse CV files (PDF/DOCX) and extract clean text"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.doc']
        
    def parse(self, file_path: str) -> Dict[str, any]:
        """
        Main entry point - parse CV and return structured data
        
        Args:
            file_path: Path to CV file
            
        Returns:
            Dict with 'text', 'sections', 'success', 'error'
        """
        file_path = Path(file_path)
        
        # Validate file exists
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return {
                'success': False,
                'error': 'File not found',
                'text': '',
                'sections': {}
            }
        
        # Check file format
        if file_path.suffix.lower() not in self.supported_formats:
            logger.error(f"Unsupported format: {file_path.suffix}")
            return {
                'success': False,
                'error': f'Unsupported format. Use: {self.supported_formats}',
                'text': '',
                'sections': {}
            }
        
        # Extract text based on format
        try:
            if file_path.suffix.lower() == '.pdf':
                raw_text = self._extract_from_pdf(str(file_path))
            elif file_path.suffix.lower() in ['.docx', '.doc']:
                raw_text = self._extract_from_docx(str(file_path))
            else:
                raise ValueError(f"Unsupported format: {file_path.suffix}")
            
            # Clean the text
            clean_text = self._clean_text(raw_text)
            
            # Segment into sections
            sections = self._segment_sections(clean_text)
            
            logger.info(f"Successfully parsed CV: {file_path.name}")
            logger.info(f"Extracted {len(clean_text)} characters")
            logger.info(f"Found {len(sections)} sections")
            
            return {
                'success': True,
                'error': None,
                'text': clean_text,
                'sections': sections,
                'file_name': file_path.name
            }
            
        except Exception as e:
            logger.error(f"Error parsing CV: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'sections': {}
            }
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        text = []
        
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():
                    text.append(page_text)
                    logger.debug(f"Extracted {len(page_text)} chars from page {page_num + 1}")
        
        return "\n\n".join(text)
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX using python-docx"""
        doc = Document(file_path)
        
        # Extract from paragraphs
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        
        # Extract from tables
        tables_text = []
        for table in doc.tables:
            for row in table.rows:
                row_text = ' | '.join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                if row_text:
                    tables_text.append(row_text)
        
        all_text = "\n".join(paragraphs)
        if tables_text:
            all_text += "\n\n" + "\n".join(tables_text)
        
        logger.debug(f"Extracted {len(all_text)} chars from DOCX")
        return all_text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
        text = re.sub(r' {2,}', ' ', text)      # Max 1 space
        text = re.sub(r'\t+', ' ', text)        # Tabs to spaces
        
        # Remove common CV artifacts
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)  # Page numbers on own line
        
        # Remove bullet point characters (we'll detect structure later)
        text = re.sub(r'^[•●○■□▪▫–-]\s*', '', text, flags=re.MULTILINE)
        
        # Remove URLs (we'll extract them separately if needed)
        # text = re.sub(r'http[s]?://\S+', '[URL]', text)
        
        # Normalize whitespace around punctuation
        text = re.sub(r'\s+([.,;:])', r'\1', text)
        text = re.sub(r'([.,;:])\s*', r'\1 ', text)
        
        # Strip leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def _segment_sections(self, text: str) -> Dict[str, str]:
        """
        Segment CV into common sections
        Returns dict with section names as keys
        """
        sections = {}
        
        # Common section headers (case-insensitive)
        section_patterns = {
            'education': r'(?i)^(education|academic background|qualifications?)$',
            'experience': r'(?i)^(experience|work history|employment|professional experience)$',
            'skills': r'(?i)^(skills|technical skills|core competencies)$',
            'projects': r'(?i)^(projects?|portfolio)$',
            'certifications': r'(?i)^(certifications?|certificates?|licenses?)$',
            'summary': r'(?i)^(summary|profile|objective|about me)$',
        }
        
        lines = text.split('\n')
        current_section = 'header'  # Everything before first section
        current_content = []
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check if this line is a section header
            matched_section = None
            for section_name, pattern in section_patterns.items():
                if re.match(pattern, line_stripped):
                    matched_section = section_name
                    break
            
            if matched_section:
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = matched_section
                current_content = []
            else:
                # Add to current section
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        logger.debug(f"Segmented into sections: {list(sections.keys())}")
        return sections


# Convenience function
def parse_cv(file_path: str) -> Dict[str, any]:
    """Quick parse function"""
    parser = CVParser()
    return parser.parse(file_path)