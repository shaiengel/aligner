import logging
import os
from logging.handlers import TimedRotatingFileHandler
from bidi.algorithm import get_display
import arabic_reshaper

def init_logger(file_name: str) -> logging.Logger:
        
    
   # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            #TimedRotatingFileHandler(
            #filename=f'logs/{worker_logger_name}',
            #when='midnight',        # Create a new file at midnight
            #interval=1,             # Create a new file every 1 day
            #backupCount=30,         # Keep up to 30 days of logs
            #encoding='utf-8',       # Use UTF-8 with BOM for Hebrew support
            #utc=False               # Use local time (not UTC)
            #),
            logging.StreamHandler()  # This will also print to console
        ]
    )
    logger = logging.getLogger(file_name)

    return logger



def format_rtl(text: str) -> str:
    # Reshape the text
    #return text
    reshaped_text = arabic_reshaper.reshape(text)
    # Convert to visual representation
    return get_display(reshaped_text)
