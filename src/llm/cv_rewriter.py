"""
CV Rewriter - Improve CV bullet points using LLM
"""
from typing import List, Dict, Optional
from loguru import logger
from src.llm.ollama_client import OllamaClient


class CVRewriter:
    """
    Rewrite CV bullet points to be more impactful
    """
    
    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        """
        Initialize CV rewriter
        
        Args:
            ollama_client: Pre-initialized Ollama client
        """
        if ollama_client is None:
            self.llm = OllamaClient()
        else:
            self.llm = ollama_client
        
        logger.info("CVRewriter initialized")
    
    def rewrite_bullet(self, bullet: str, target_skill: Optional[str] = None) -> str:
        """
        Rewrite a single CV bullet point
        
        Args:
            bullet: Original bullet point
            target_skill: Skill to emphasize (optional)
            
        Returns:
            Improved bullet point
        """
        system = """You are an expert CV writer. Rewrite CV bullet points to be:
- Action-oriented (start with strong verbs)
- Quantified (include numbers/metrics when possible)
- Achievement-focused (show impact, not just duties)
- Concise (1-2 lines max)
- ATS-friendly (avoid fancy formatting)

Respond with ONLY the improved bullet point. No explanation."""
        
        if target_skill:
            prompt = f"""Original: {bullet}

Rewrite this to emphasize {target_skill} experience. Make it stronger and more impactful."""
        else:
            prompt = f"""Original: {bullet}

Rewrite this to be more impactful and achievement-focused."""
        
        improved = self.llm.generate(
            prompt,
            system=system,
            temperature=0.7,
            max_tokens=100
        )
        
        # Clean up response
        improved = improved.strip().lstrip('•-*').strip()
        
        return improved
    
    def rewrite_bullets_batch(self, bullets: List[str]) -> List[Dict]:
        """
        Rewrite multiple bullets
        
        Args:
            bullets: List of original bullets
            
        Returns:
            List of dicts with original and improved versions
        """
        results = []
        
        for bullet in bullets:
            improved = self.rewrite_bullet(bullet)
            results.append({
                'original': bullet,
                'improved': improved
            })
        
        return results
    
    def generate_skill_bullet(self, skill: str, context: str = "") -> str:
        """
        Generate a new bullet point for a missing skill
        
        Args:
            skill: Skill to create bullet for
            context: Additional context (job title, industry, etc.)
            
        Returns:
            Generated bullet point
        """
        system = """You are an expert CV writer. Create strong, achievement-focused CV bullet points.
Use action verbs, include metrics when relevant, and focus on impact.
Respond with ONLY the bullet point. No explanation."""
        
        if context:
            prompt = f"""Create a CV bullet point demonstrating {skill} experience in the context of: {context}

Make it specific, quantified, and achievement-focused."""
        else:
            prompt = f"""Create a CV bullet point demonstrating strong {skill} skills.

Make it specific, quantified, and achievement-focused."""
        
        bullet = self.llm.generate(
            prompt,
            system=system,
            temperature=0.8,
            max_tokens=100
        )
        
        # Clean up
        bullet = bullet.strip().lstrip('•-*').strip()
        
        return bullet


# Convenience function
def rewrite_cv_bullet(bullet: str, target_skill: Optional[str] = None) -> str:
    """Quick rewrite function"""
    rewriter = CVRewriter()
    return rewriter.rewrite_bullet(bullet, target_skill)