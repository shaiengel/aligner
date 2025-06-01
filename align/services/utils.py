from pathlib import Path
import logging

def create_folder(folder_name):
    try:
        # Get current working directory and create new folder path
        current_path = Path.cwd()
        new_folder_path = current_path / folder_name
        
        # Create folder if it doesn't exist
        new_folder_path.mkdir(exist_ok=True)
        logging.info(f"Folder created successfully at: {new_folder_path}")
        
    except Exception as e:
        logging.info(f"Error creating folder: {str(e)}")