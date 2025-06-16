from align.handler.aligner import aligner
from align.services.utils import create_folder
import os
from align.services.logger import get_logger

logger = get_logger()

START_SEARCHING_INDEX = 40

def align_massechet(audio_repo, audio_file_template, doc_repo, doc_file_template, output_repo, start_page):
    
    audio_files = os.listdir(audio_repo)
    doc_files = os.listdir(doc_repo)
    if len(audio_files) != len(doc_files):
        raise ValueError(f"Number of audio files {audio_repo} and document files {doc_repo} must match.")
    number_of_files = len(audio_files)
    index = start_page
    start_search_index = START_SEARCHING_INDEX
    while index < number_of_files + start_page:            
        start_search_index = min(start_search_index + 10, START_SEARCHING_INDEX)
        audio_file = f"{audio_repo}\\{audio_file_template.format(index)}"
        logger.info(f"Processing {audio_file} with start_search_index {start_search_index}")    
        if index == start_page:
            doc_file0 = f"{doc_repo}\\{doc_file_template.format(index)}"
            doc_file1 = f"{doc_repo}\\{doc_file_template.format(index + 1)}"
            start_search_index = aligner(audio_file, ["", doc_file0, doc_file1], output_repo, start_search_index)
        elif index == number_of_files + start_page:
            doc_file0 = f"{doc_repo}\\{doc_file_template.format(index - 1)}"
            doc_file1 = f"{doc_repo}\\{doc_file_template.format(index)}"
            start_search_index = aligner(audio_file, [doc_file0, doc_file1, ""], output_repo, start_search_index)
        else:            
            doc_file0 = f"{doc_repo}\\{doc_file_template.format(index - 1)}"
            doc_file1 = f"{doc_repo}\\{doc_file_template.format(index)}"
            doc_file2 = f"{doc_repo}\\{doc_file_template.format(index + 1)}"
            start_search_index = aligner(audio_file, [doc_file0, doc_file1, doc_file2], output_repo, start_search_index)

        start_search_index -= START_SEARCHING_INDEX
        index += 1   

def align_repo(respos_dict):
    for item in respos_dict:
        audio_repo = item['audio_repo']
        audio_file_template = item['audio_file_template']
        doc_repo = item['doc_repo']
        doc_file_template = item['doc_file_template']
        output_repo = item['output_repo']
        start_page = item.get('start_page', 2)
        
        create_folder(output_repo)
        massechet = audio_repo.split('\\')[-1]
        massechet_output_repo = f"{output_repo}\\{massechet}"
        create_folder(massechet_output_repo)
        logger.info(f"Start Aligning massechet = {massechet}")
        align_massechet(audio_repo, audio_file_template, doc_repo, doc_file_template, massechet_output_repo, start_page)
