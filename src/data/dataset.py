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
        self._hf_dataset: Optional[Dataset] = None
        
        # Verify directory exists
        if not self.data_dir.exists():
            logger.error(f"Data directory does not exist: {self.data_dir}")
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Create cache directory if specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cache directory ready: {self.cache_dir}")
        
        self._load_data()
        
    @property
    def features(self):
        """Get dataset features for HuggingFace compatibility."""
        if self._hf_dataset is None:
            self._hf_dataset = self.to_huggingface_dataset()
        return self._hf_dataset.features
    
    @property
    def column_names(self):
        """Get column names for HuggingFace compatibility."""
        if self._hf_dataset is None:
            self._hf_dataset = self.to_huggingface_dataset()
        return self._hf_dataset.column_names
    
    def __getitem__(self, idx):
        """Get an item from the dataset."""
        if self._hf_dataset is None:
            self._hf_dataset = self.to_huggingface_dataset()
        return self._hf_dataset[idx]
    
    def __len__(self):
        """Get dataset length."""
        return len(self._data)
    
    def to_huggingface_dataset(self) -> Dataset:
        """Convert to HuggingFace Dataset format."""
        logger.info("Converting to HuggingFace Dataset")
        try:
            if not self._data:
                logger.error("No data loaded to convert to HuggingFace Dataset")
                raise ValueError("Dataset is empty")
            
            # Convert our data format to HuggingFace format
            hf_data = []
            for item in self._data:
                hf_item = {
                    'prompt': item['prompt'],
                    'answer': item['answer']
                }
                hf_data.append(hf_item)
            
            dataset = Dataset.from_list(hf_data)
            logger.info(f"Successfully converted to HuggingFace Dataset with {len(dataset)} examples")
            logger.debug(f"Dataset features: {dataset.features}")
            logger.debug(f"Dataset columns: {dataset.column_names}")
            
            return dataset
        except Exception as e:
            logger.error(f"Error converting to HuggingFace Dataset: {str(e)}")
            raise
    
    def sample_batch(self, batch_size: Optional[int] = None) -> Dict:
        """Sample a batch of examples for monitoring.
        
        Args:
            batch_size: Optional batch size, defaults to per_device_train_batch_size if None
            
        Returns:
            Dictionary containing batch examples
        """
        logger.info(f"Sampling batch with size {batch_size}")
        
        if not self._data:
            logger.warning("No data available for sampling")
            return {}
        
        # Use default batch size if none provided
        effective_batch_size = batch_size or 8  # Default from notebook
        logger.info(f"Using effective batch size: {effective_batch_size}")
        
        try:
            import random
            indices = random.sample(range(len(self._data)), min(effective_batch_size, len(self._data)))
            batch = [self._data[i] for i in indices]
            
            # Log batch statistics
            logger.info(f"Sampled batch of size {len(batch)}")
            if batch:
                logger.debug("First example in batch:")
                logger.debug(f"  Prompt length: {len(str(batch[0]['prompt']))}")
                logger.debug(f"  Answer length: {len(str(batch[0]['answer']))}")
            
            return {
                'examples': batch,
                'size': len(batch),
                'original_size': effective_batch_size
            }
            
        except Exception as e:
            logger.error(f"Error sampling batch: {str(e)}")
            return {
                'examples': [],
                'size': 0,
                'error': str(e)
            }
    
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