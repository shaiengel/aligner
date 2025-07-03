from pathlib import Path
import logging
import re
from align.services.logger import get_logger

logger = get_logger()

def create_folder(folder_name):
    try:
        # Get current working directory and create new folder path
        current_path = Path.cwd()
        new_folder_path = current_path / folder_name
        
        # Create folder if it doesn't exist
        new_folder_path.mkdir(exist_ok=True)
        logger.info(f"Folder created successfully at: {new_folder_path}")
        
    except Exception as e:
        logging.info(f"Error creating folder: {str(e)}")

def format_text(text):
    # First handle combined punctuation marks without spacing between them
    combined_punct = ['?!', '!?', '...']
    temp_replacements = {
        '?!': '__QMARK_EMARK__',
        '!?': '__EMARK_QMARK__',
        '...': '__ELLIPSIS__'
    }
    
    # Temporarily replace combined punctuation
    for punct in combined_punct:
        text = text.replace(punct, temp_replacements[punct])
    
    # Add spaces after single punctuation marks
    text = re.sub(r'([.,!?:;])(?![\s\d])', r'\1 ', text)
    
    # Restore combined punctuation
    for punct, temp in temp_replacements.items():
        text = text.replace(temp, punct + ' ')
    
    # Remove any double spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()     


def format_time(seconds: float) -> str:
    total_millis = int(round(seconds * 1000))
    hours = total_millis // 3600000
    minutes = (total_millis % 3600000) // 60000
    seconds_part = (total_millis % 60000) // 1000
    millis = total_millis % 1000
    return f"{hours:02}:{minutes:02}:{seconds_part:02}.{millis:03}"