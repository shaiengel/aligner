from align.services.audio_utils import preprocess_audio
from align.handler.transcribe import (audio_to_text, 
                                      audio_to_text_ivirit, 
                                      audio_to_text_aligner,
                                      write_to_srt,
                                      get_model,                                      
                                      convert_audio,
                                      find_silence
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
import json

logger = get_logger()

START_SEARCHING_INDEX = 40
END_SEARCHING_INDEX = -75
PROBABILTY_THRESHOLD = 0.10

def aligner(audio_file, text: list[str], output_repo, start_search_index=START_SEARCHING_INDEX):
    
    text0 = read_docx(text[0])
    text1 = read_docx(text[1])
    text2 = read_docx(text[2])     
    audio_data = convert_audio(audio_file)  # Ensure audio is in the correct format
    model = get_model()   
    start_text, probability_list = search_starting(model, audio_data, [text0, text1, text2], start_search_index)
    
    if start_text is None:
        logger.error(f"Failed to find a suitable starting point in the text for audio file {audio_file}.")
        return START_SEARCHING_INDEX*2
    cut_from_start = 0
    full_text = start_text
    saved_warnings = []
    logger.info(f"length of text after searching for start {len(full_text.split())} words")     
    if text2 != "":
        while len(full_text.split()) >  START_SEARCHING_INDEX:        
            cut_from_start = search_end(probability_list, full_text)
            logger.info(f"search end returned {cut_from_start} words from the start")       
            words = full_text.split()
            full_text = ' '.join(words[:cut_from_start + 1]) 
            response, _ = audio_to_text_aligner(model, audio_data, full_text)            
            probability_list = response.ori_dict["segments"][0]["words"]  
            check_cut_from_start = search_end(probability_list, full_text) 
            logger.info(f"search end check returned {check_cut_from_start} words from the start")                  
            words = full_text.split()
            full_text = ' '.join(words[:check_cut_from_start + 1])
            probability_list = probability_list[:check_cut_from_start + 1]
            if abs(check_cut_from_start - cut_from_start) <= 2:
                cut_from_start = check_cut_from_start
                break
            cut_from_start = check_cut_from_start
            
        
    
    logger.info(f"search end final cut_from_start {cut_from_start} words from the start") 
    response, _ = audio_to_text_aligner(model, audio_data, full_text, True)
    output_file = write_to_srt(response, audio_file, output_repo) 
    srt_statistics_repo = f"{output_repo}\\srt_statistics"
    create_folder(srt_statistics_repo)
    result_probability_list = add_probabilties_to_srt(output_file, response.ori_dict["segments"][0]["words"], srt_statistics_repo)
    w = weighted_run_score(result_probability_list)
    logger.info(f"weighted_run_score: w = {w}")
    if w > 0.25:
        logger.warning(f"WARNING weighted_run_score is {w}. Check the audio quality or the text alignment.")

    silence_folder = f"{output_repo}\\silences"
    create_folder(silence_folder)
    find_silence(audio_file, f"{output_file}.silences", silence_folder)       
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
        combined_text = combine_end_text(clean_text1, clean_text2, END_SEARCHING_INDEX * -1)
    
    if clean_text0 == "": #first page
        response, _ = audio_to_text_aligner(model, audio_file, combined_text)
        return combined_text, response.ori_dict["segments"][0]["words"]     
    
    saved_warnings = []
    while words_count >= END_SEARCHING_INDEX: 
        logger.info(f"Aligning backword with {words_count} words from last page")   
        text_search = combine_start_text(clean_text0, combined_text, words_count)     
        response, captured_warnings = audio_to_text_aligner(model, audio_file, text_search)
        if captured_warnings:
                save_warnings(saved_warnings, captured_warnings, words_count)

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

    logger.info(f"try to search for the best matching")
    words_count = find_best_matching(saved_warnings)
    logger.info(f"found best matching with {words_count} words")
    text_search = combine_start_text(clean_text0, combined_text, words_count)     
    response, _ = audio_to_text_aligner(model, audio_file, text_search)
    return text_search, response.ori_dict["segments"][0]["words"]     

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


def save_warnings(saved_warnings, captured_warnings, backword_words):
    segment_pattern = re.compile(r'(\d+)/(\d+) segments failed to align')

    entry = {
        "backword_words": None,
        "mismatched_segments": None,
        "total_segments": None
    }

    for line in captured_warnings:
        entry["backword_words"] = backword_words            

        segment_match = segment_pattern.search(line)
        if segment_match:
            entry["mismatched_segments"] = int(segment_match.group(1))
            entry["total_segments"] = int(segment_match.group(2))
            break

    # Append the parsed result to saved_warnings
    saved_warnings.append(entry)

def find_best_matching(saved_warnings):
    
    best_index = None
    best_mismatched = None
    best_total = None

    for i in range(len(saved_warnings)):
        entry = saved_warnings[i]
        mm = entry["mismatched_segments"]
        ts = entry["total_segments"]

        if mm is not None:
            if (
                best_mismatched is None
                or mm < best_mismatched
                or (mm == best_mismatched and ts < best_total)
            ):
                best_index = i
                best_mismatched = mm
                best_total = ts

    # Step 3: Check no-info entries in context of neighbors
    for i in range(len(saved_warnings)):
        entry = saved_warnings[i]

        if entry["mismatched_segments"] is None:
            # Look for previous segment info
            prev_i = i - 1
            while prev_i >= 0 and saved_warnings[prev_i]["mismatched_segments"] is None:
                prev_i -= 1

            # Look for next segment info
            next_i = i + 1
            while next_i < len(saved_warnings) and saved_warnings[next_i]["mismatched_segments"] is None:
                next_i += 1

            prev_ok = prev_i >= 0 and saved_warnings[prev_i]["mismatched_segments"] is not None
            next_ok = next_i < len(saved_warnings) and saved_warnings[next_i]["mismatched_segments"] is not None

            use_this = False

            if prev_ok and next_ok:
                prev_mm = saved_warnings[prev_i]["mismatched_segments"]
                next_mm = saved_warnings[next_i]["mismatched_segments"]

                if prev_mm <= best_mismatched and next_mm <= best_mismatched:
                    use_this = True

            elif prev_ok:
                prev_mm = saved_warnings[prev_i]["mismatched_segments"]
                if prev_mm <= best_mismatched:
                    use_this = True

            if use_this:
                best_index = i
                best_mismatched = prev_mm  # Use known low
                best_total = 0  # Force preference
                break  # Pick first valid no-info entry
    

    if best_index is not None:
        logger.info("\nBest match:")
        logger.info(json.dumps(saved_warnings[best_index], indent=2))
        return saved_warnings[best_index]["backword_words"]
    else:
        logger.warning("\nNo valid match found.")  
        return None  
    
    
