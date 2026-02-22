"""
Ollama LLM Client - Local language model integration
"""
import requests
import json
from typing import Dict, List, Optional
from loguru import logger


class OllamaClient:
    """
    Client for interacting with local Ollama LLM
    """
    
    def __init__(self,
                 model: str = "gemma2:2b",
                 base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama client
        
        Args:
            model: Model name (tinyllama, llama3.2:3b, etc.)
            base_url: Ollama API endpoint
        """
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
        logger.info(f"OllamaClient initialized with model: {model}")
        
        # Verify Ollama is running
        if not self._check_ollama_running():
            logger.warning("Ollama server not detected. Make sure Ollama is running.")
    
    def _check_ollama_running(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def generate(self,
                prompt: str,
                system: Optional[str] = None,
                temperature: float = 0.7,
                max_tokens: int = 500,
                stream: bool = False) -> str:
        """
        Generate text using Ollama
        
        Args:
            prompt: User prompt
            system: System prompt (instructions)
            temperature: Randomness (0-1, lower = more focused)
            max_tokens: Maximum response length
            stream: Stream response (not implemented)
            
        Returns:
            Generated text
        """
        # Build full prompt with system message
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            logger.debug(f"Sending request to Ollama (model: {self.model})")
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=60  # LLMs can take time
            )
            
            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code}")
                return f"Error: {response.text}"
            
            result = response.json()
            generated_text = result.get('response', '').strip()
            
            logger.debug(f"Generated {len(generated_text)} characters")
            return generated_text
            
        except requests.exceptions.Timeout:
            logger.error("Ollama request timeout")
            return "Error: Request timeout. Ollama might be processing slowly."
        except Exception as e:
            logger.error(f"Ollama error: {str(e)}")
            return f"Error: {str(e)}"
    
    def generate_with_json(self,
                          prompt: str,
                          system: Optional[str] = None,
                          temperature: float = 0.3) -> Dict:
        """
        Generate JSON response (for structured output)
        
        Args:
            prompt: User prompt
            system: System prompt
            temperature: Lower for more consistent JSON
            
        Returns:
            Parsed JSON dict
        """
        # Add JSON instruction to prompt
        json_prompt = f"{prompt}\n\nRespond ONLY with valid JSON. No markdown, no explanation."
        
        response = self.generate(
            json_prompt,
            system=system,
            temperature=temperature,
            max_tokens=1000
        )
        
        # Try to extract JSON from response
        try:
            # Remove markdown code blocks if present
            clean_response = response.strip()
            if clean_response.startswith('```'):
                # Extract content between ```json and ```
                lines = clean_response.split('\n')
                json_lines = []
                in_json = False
                for line in lines:
                    if line.strip().startswith('```'):
                        in_json = not in_json
                        continue
                    if in_json:
                        json_lines.append(line)
                clean_response = '\n'.join(json_lines)
            
            return json.loads(clean_response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Response was: {response}")
            return {"error": "Invalid JSON response", "raw": response}


# Convenience function
def get_ollama_client(model: str = "gemma2:2bb") -> OllamaClient:
    """Get Ollama client instance"""
    return OllamaClient(model=model)