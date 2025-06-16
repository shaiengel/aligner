from align.services.audio_utils import preprocess_audio
from align.handler.transcribe import (audio_to_text, 
                                      audio_to_text_ivirit, 
                                      audio_to_text_aligner,
                                      write_to_srt,
                                      get_model,
                                      convert_audio
            )
from align.services.logger import init_logger, format_rtl
from align.services.docx_util import (
    remove_marks_for_aligner,
    read_docx,
    combine_end_text,
    combine_start_text,
    remove_parantheses
    
)
from align.services.utils import create_folder
from align.services.statistics import add_probabilties_to_srt, weighted_run_score
import re

from align.services.logger import get_logger

logger = get_logger()

START_SEARCHING_INDEX = 40
END_SEARCHING_INDEX = -40
PROBABILTY_THRESHOLD = 0.10

def aligner(audio_file, text: list[str], output_repo, start_search_index=START_SEARCHING_INDEX):
    
    text0 = read_docx(text[0])
    text1 = read_docx(text[1])
    text2 = read_docx(text[2]) 
    audio_data = convert_audio(audio_file)  # Ensure audio is in the correct format
    model = get_model()   
    start_text, probability_list = search_starting(model, audio_data, [text0, text1, text2], start_search_index)
    
    cut_from_start = 0
    full_text = start_text
    logger.info(f"length of text after searching for start {len(full_text.split())} words")     
    if text2 != "":
        while len(full_text.split()) >  START_SEARCHING_INDEX:        
            cut_from_start = search_end(probability_list, full_text)
            logger.info(f"search end returned {cut_from_start} words from the start")       
            words = full_text.split()
            full_text = ' '.join(words[:cut_from_start + 1]) 
            response, captured_warnings = audio_to_text_aligner(model, audio_data, full_text)
            probability_list = response.ori_dict["segments"][0]["words"]  
            check_cut_from_start = search_end(probability_list, full_text) 
            logger.info(f"search end check returned {check_cut_from_start} words from the start")                  
            words = full_text.split()
            full_text = ' '.join(words[:check_cut_from_start + 1])
            probability_list = probability_list[:check_cut_from_start + 1]
            if check_cut_from_start == cut_from_start:
                cut_from_start = check_cut_from_start
                break
            cut_from_start = check_cut_from_start
            
        
    
    logger.info(f"search end final cut_from_start {cut_from_start} words from the start")     
    response, captured_warnings = audio_to_text_aligner(model, audio_data, full_text, True)
    logger.info(captured_warnings)
    #logger.info(f"Final word alignement")
    #logger.info(format_rtl(full_text))
    output_file = write_to_srt(response, audio_file, output_repo) 
    result_probability_list = add_probabilties_to_srt(output_file, response.ori_dict["segments"][0]["words"])
    w = weighted_run_score(result_probability_list)
    print(f"weighted_run_score: w = {w}")
    cut_from_end = len(start_text.split()) - cut_from_start - 1
    return cut_from_end
    


def search_starting(model, audio_file, text: list[str], words_count = START_SEARCHING_INDEX):    
    
    clean_text0 = remove_marks_for_aligner(text[0])
    clean_text1 = remove_marks_for_aligner(text[1])
    clean_text2 = remove_marks_for_aligner(text[2])

    clean_text0 = remove_parantheses(clean_text0)
    clean_text1 = remove_parantheses(clean_text1)
    clean_text2 = remove_parantheses(clean_text2)
    if clean_text2 == "":   #last page
        combined_text = clean_text1
    else:
        combined_text = combine_end_text(clean_text1, clean_text2, START_SEARCHING_INDEX)
    
    if clean_text0 == "": #first page
        response, captured_warnings = audio_to_text_aligner(model, audio_file, combined_text)
        return combined_text, response.ori_dict["segments"][0]["words"]     
    
    while words_count >= END_SEARCHING_INDEX: 
        logger.info(f"Aligning backword with {words_count} words from last page")   
        text_search = combine_start_text(clean_text0, combined_text, words_count)     
        response, captured_warnings = audio_to_text_aligner(model, audio_file, text_search)

        found = False
        word_counter_checker = 5

        i = 0
        while i < word_counter_checker:
            probablity = response.ori_dict["segments"][0]["words"][i]['probability']
            if probablity > PROBABILTY_THRESHOLD:                    
                if i == 0:
                    found = check_following_words(response)
                    if not found:
                        i += 1
                        continue
                else:
                    word_counter_checker = i                             
                break
            i += 1  
        if found:
            logger.info(f"Final word alignement")
            logger.info(f"found aligning backword with {words_count} words from last page")               
            return text_search, response.ori_dict["segments"][0]["words"]     
        words_count -= word_counter_checker 

    return None, None

def search_end(probability_list, text: list[str]):
    i = len(probability_list) - 1    
    while i > 0:
        probablity = probability_list[i]['probability']
        if probablity > PROBABILTY_THRESHOLD:                    
            found = check_preceding_words(probability_list, i)
            if not found:
                i -= 1
            else:
                return i                    
        i -= 1
    return None

def check_following_words(response):
    word_counter_checker = 11
    i = 1
    counter = 0
    while i < word_counter_checker:
        probablity = response.ori_dict["segments"][0]["words"][i]['probability']  
        if probablity > PROBABILTY_THRESHOLD: 
            counter += 1 
        i += 1
    if counter > 7:
        return True
    else:
        return False
    
def check_preceding_words(probability_list, index):
    word_counter_checker = 10
    i = index
    end = index - word_counter_checker
    counter = 0
    while i > end:
        probablity = probability_list[i]['probability']  
        if probablity > PROBABILTY_THRESHOLD: 
            counter += 1 
        i -= 1
    if counter > 7:
        return True
    else:
        return False    
    
