from textSummarizer.config.configuration import ConfigurationManager
from transformers import pipeline
import os


class PredictionPipeline:
    def __init__(self):
        try:
            self.config = ConfigurationManager().get_model_evaluation_config()
        except:
            self.config = None
        self.pipeline = None
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize the pipeline with DistilBART only"""
        try:
            # Check if custom trained DistilBART model exists
            if (self.config and 
                os.path.exists(self.config.model_path) and 
                os.path.exists(self.config.tokenizer_path) and 
                os.path.isdir(self.config.model_path) and 
                os.path.isdir(self.config.tokenizer_path)):
                
                # Use custom trained DistilBART model
                model_path = self.config.model_path
                print("Using custom trained DistilBART model...")
            else:
                # Use pre-trained DistilBART model
                model_path = "sshleifer/distilbart-cnn-12-6"
                print("Using pre-trained DistilBART model...")
            
            # Initialize DistilBART pipeline
            self.pipeline = pipeline(
                "summarization", 
                model=model_path,
                device=-1,  # CPU usage for stability
                framework="pt"
            )
            print("DistilBART pipeline initialized successfully")
            
        except Exception as e:
            print(f"Error initializing DistilBART pipeline: {e}")
            self.pipeline = None

    def predict(self, text):
        """Generate summary for the given text"""
        try:
            # Validate input
            if not text or len(text.strip()) < 10:
                return "Error: Text is too short for summarization. Please provide at least 10 characters."
            
            # Check if pipeline is available
            if self.pipeline is None:
                return "Error: Summarization model is not available. Please try again later."
            
            # Truncate very long texts to avoid memory issues and timeouts
            max_input_length = 800  # Reduced for faster processing
            if len(text) > max_input_length:
                text = text[:max_input_length] + "..."
                print(f"Text truncated to {max_input_length} characters for faster processing")
            
            print("Processing text for summarization...")
            
            # Generate summary with DistilBART-optimized parameters
            gen_kwargs = {
                "max_length": 142,      # DistilBART optimal length
                "min_length": 30,       # Good minimum for summaries
                "do_sample": False,     # Deterministic for consistency  
                "early_stopping": True, # Stop when done
                "num_beams": 4,         # Good balance for DistilBART
                "no_repeat_ngram_size": 3  # Avoid repetition
            }

            # Generate summary with DistilBART
            result = self.pipeline(text, **gen_kwargs)
            summary = result[0]["summary_text"]
            
            print("Summary generated successfully")
            return summary
            
        except Exception as e:
            error_msg = f"Error during summarization: {str(e)}"
            print(error_msg)
            
            # Return a user-friendly error message
            if "timeout" in str(e).lower():
                return "The summarization is taking longer than expected. Please try with shorter text."
            elif "memory" in str(e).lower() or "cuda" in str(e).lower():
                return "Memory error occurred. Please try with shorter text."
            else:
                return f"Sorry, I couldn't summarize this text right now. Please try again with different text."