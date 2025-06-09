from align.handler.aligner import aligner
import os

START_SEARCHING_INDEX = 40

def align_massechet(audio_repo, audio_file_template, doc_repo, doc_file_template, output_repo, start_page):
    
    audio_files = os.listdir(audio_repo)
    doc_files = os.listdir(doc_repo)
    if len(audio_files) != len(doc_files):
        raise ValueError(f"Number of audio files {audio_repo} and document files {doc_repo} must match.")
    number_of_files = len(audio_files)
    index = start_page
    while index <= number_of_files + start_page:
        start_search_index -= START_SEARCHING_INDEX
        if start_search_index > START_SEARCHING_INDEX:
            start_search_index = START_SEARCHING_INDEX
        else:
            start_search_index += 10
        audio_file = audio_file_template.format(index)
        if index == start_page:
            doc_file0 = doc_file_template.format(index)
            doc_file1 = doc_file_template.format(index + 1)
            start_search_index = aligner(audio_file, ["", doc_file0, doc_file1], output_repo, start_search_index)
        elif index == number_of_files + start_page:
            doc_file0 = doc_file_template.format(index - 1)
            doc_file1 = doc_file_template.format(index)
            start_search_index = aligner(audio_file, [doc_file0, doc_file1, ""], output_repo, start_search_index)
        else:            
            doc_file0 = doc_file_template.format(index)
            doc_file1 = doc_file_template.format(index + 1)
            doc_file2 = doc_file_template.format(index + 2)
            start_search_index = aligner(audio_file, [doc_file0, doc_file1, doc_file2], output_repo, start_search_index)