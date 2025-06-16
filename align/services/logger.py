import logging
import os
from logging.handlers import TimedRotatingFileHandler
from bidi.algorithm import get_display
import arabic_reshaper
from pathlib import Path

logger = None

def create_folder():
    """Create logs directory if it doesn't exist"""
    if not os.path.exists('logs'):
        os.makedirs('logs')

def init_logger(worker_logger_name: str) -> logging.Logger:
    global logger

    if logger is not None:
        return logger
   
    create_folder()
    
    # Create a named logger instead of configuring root logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create file handler
    file_handler = TimedRotatingFileHandler(
        filename=f'logs/{worker_logger_name}',
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8',
        utc=False
    )
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger

def get_logger():
    """
    Get the configured logger instance
    
    Returns:
        logging.Logger: Configured logger instance
    """
    global logger
    if logger is None:
        pid = os.getpid()
        worker_logger_name = f"message_service_{pid}.log"
        logger = init_logger(worker_logger_name)

    return logger        


def format_rtl(text: str) -> str:
    # Reshape the text
    #return text
    reshaped_text = arabic_reshaper.reshape(text)
    # Convert to visual representation
    return get_display(reshaped_text)
