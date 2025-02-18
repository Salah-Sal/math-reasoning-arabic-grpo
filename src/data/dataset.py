import json
import glob
from pathlib import Path
from typing import Dict, List, Optional
from datasets import Dataset
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)

class ArabicMathDataset:
    """Dataset class for Arabic math problems."""
    
    def __init__(self, 
                 data_dir: str | Path, 
                 system_prompt: Optional[str] = None,
                 cache_dir: Optional[Path] = None):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing JSON problem files
            system_prompt: Optional system prompt to prepend to questions
            cache_dir: Optional directory for caching processed data
        """
        logger.info(f"Initializing ArabicMathDataset with data_dir={data_dir}, cache_dir={cache_dir}")
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.system_prompt = system_prompt or "أنت مساعد ذكي متخصص في حل المسائل الرياضية. قم بحل المسألة التالية خطوة بخطوة."
        self._data: List[Dict] = []
        
        # Verify directory exists
        if not self.data_dir.exists():
            logger.error(f"Data directory does not exist: {self.data_dir}")
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Create cache directory if specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cache directory ready: {self.cache_dir}")
        
        self._load_data()
        
    def _load_data(self) -> None:
        """Load and process all JSON files in the data directory."""
        try:
            json_files = glob.glob(str(self.data_dir / "*.json"))
            logger.info(f"Found {len(json_files)} JSON files in {self.data_dir}")
            
            if not json_files:
                logger.error(f"No JSON files found in {self.data_dir}")
                raise FileNotFoundError(f"No JSON files found in {self.data_dir}")
            
            # Log sample of file paths for verification
            logger.debug(f"Sample files: {json_files[:3]}")
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Verify expected structure
                    if 'translation' not in data:
                        logger.warning(f"Missing 'translation' key in {json_file}")
                        continue
                        
                    if 'question' not in data['translation'] or 'answer' not in data['translation']:
                        logger.warning(f"Missing question/answer in translation: {json_file}")
                        continue
                    
                    # Extract question and answer
                    question = data['translation']['question']
                    answer = self._extract_answer(data['translation']['answer'])
                    
                    # Log sample data for verification
                    if len(self._data) == 0:
                        logger.debug(f"First example - Question: {question[:100]}...")
                        logger.debug(f"First example - Answer: {answer}")
                    
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
                    
            logger.info(f"Successfully loaded {len(self._data)} examples")
            
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