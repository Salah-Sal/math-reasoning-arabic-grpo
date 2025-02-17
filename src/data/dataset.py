import json
import glob
from pathlib import Path
from typing import Dict, List, Optional
from datasets import Dataset
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)

class ArabicMathDataset:
    """Dataset class for Arabic math problems."""
    
    def __init__(self, data_dir: str | Path, system_prompt: Optional[str] = None):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing JSON problem files
            system_prompt: Optional system prompt to prepend to questions
        """
        self.data_dir = Path(data_dir)
        self.system_prompt = system_prompt or "أنت مساعد ذكي متخصص في حل المسائل الرياضية. قم بحل المسألة التالية خطوة بخطوة."
        self._data: List[Dict] = []
        self._load_data()
        
    def _load_data(self) -> None:
        """Load and process all JSON files in the data directory."""
        try:
            json_files = glob.glob(str(self.data_dir / "*.json"))
            logger.info(f"Found {len(json_files)} JSON files in {self.data_dir}")
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # Extract question and answer
                    question = data['translation']['question']
                    answer = self._extract_answer(data['translation']['answer'])
                    
                    # Create formatted example
                    example = {
                        'prompt': [
                            {
                                'role': 'system',
                                'content': self.system_prompt
                            },
                            {
                                'role': 'user',
                                'content': question
                            }
                        ],
                        'answer': answer
                    }
                    
                    self._data.append(example)
                    
                except Exception as e:
                    logger.error(f"Error processing file {json_file}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
            
    def _extract_answer(self, answer_text: str) -> str:
        """Extract numerical answer from the answer text and convert Arabic numerals."""
        try:
            # Split on #### and take the last part
            parts = answer_text.split('####')
            if len(parts) < 2:
                raise ValueError("No answer delimiter (####) found")
                
            numerical_answer = parts[-1].strip()
            
            # Convert Arabic numerals to English
            arabic_to_english = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
            numerical_answer = numerical_answer.translate(arabic_to_english)
            
            # Extract only digits
            numerical_answer = ''.join(c for c in numerical_answer if c.isdigit())
            
            return numerical_answer
            
        except Exception as e:
            logger.error(f"Error extracting answer: {str(e)}")
            return ""
    
    def is_valid(self) -> bool:
        """Check if the dataset is valid."""
        return len(self._data) > 0 and all(
            isinstance(item.get('answer', ''), str) and
            len(item.get('prompt', [])) == 2
            for item in self._data
        )
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self._data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a specific example from the dataset."""
        return self._data[idx]
    
    def to_huggingface_dataset(self) -> Dataset:
        """Convert to HuggingFace Dataset format."""
        return Dataset.from_list(self._data) 