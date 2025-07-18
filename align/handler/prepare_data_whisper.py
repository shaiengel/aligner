from datetime import datetime
import re
import os
from align.services.create_dataset import (
    generate_slices,
    prepare_training_dataset,
    concatenate_all_datasets,
    create_dataset_card,
    split_dataset,
    save_dataset,
    upload_dataset_to_hub
    )
from stable_whisper.result import WhisperResult, Segment, WordTiming
from align.services.silence_segment_analyzer import read_silence_segments, decide_state_transition
from align.handler.transcribe import load_audio
from align.services.logger import get_logger
from align.services.statistics import slices_statistics

logger = get_logger()

PROBABILITY_THRESHOLD = 0.10
MAX_WORD_DURATION = 1.6  # seconds
AVERAGE_WORD_DURATION_THRESHOLD = 0.35
ALLOWED_ERROR_PERCENTAGE = 0.416

def srt_time_to_seconds(time_str):
    # Convert SRT format: "00:00:01,780" to seconds
    fmt = "%H:%M:%S.%f"
    time_str = time_str.replace(',', '.')

    # Parse the time string
    dt = datetime.strptime(time_str, "%H:%M:%S.%f")

    # Convert to seconds since start of the day
    seconds = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1_000_000
    return seconds
    


def prepare_data_from_srt(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    blocks = re.split(r'\n\s*\n', content.strip())
    entries = []
    segments = []

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 4:
            continue

        interval = lines[1].strip()
        match = re.match(r'(.+?) --> (.+)', interval)
        if not match:
            continue

        start_raw, end_raw = match.groups()
        start = srt_time_to_seconds(start_raw)
        end = srt_time_to_seconds(end_raw) 
        duration = end - start       

        # Handle multi-line sentence
        prob_line = lines[-2].strip()
        sentence_lines = lines[2:-2]
        sentence = ' '.join(sentence_lines).strip()
        word_count = len(sentence.split())
        probabilities = eval(prob_line)
        #probabilities = probabilities[::-1]  # Reverse to match the order of words in the sentence
        

        word_intervals_line = lines[4]
        word_time_line = word_intervals_line.strip()
        word_time = eval(word_time_line)
        

        # Convert intervals to durations
        try:
            durations = []
            for start_word, end_word in word_time:                
                durations.append(srt_time_to_seconds(end_word) - srt_time_to_seconds(start_word))
                
        except Exception:
            continue

        if durations:
            max_duration = max(durations)

        entries.append({
            'start': start,
            'end': end,
            'duration': duration,
            'max_duration': max_duration,
            'sentence': sentence,
            'word_count': word_count,
            'probabilities': probabilities,
            'word_time': word_time,
            'is_valid': True,
            'merge_allowed': False,
        })

        #segments.append(Segment(
        #    start=start,
        #    end=end,
        #    text=sentence            
        #))
        

    return entries

def remove_suspected_text_from_audio(audio_file, entries, silence_file):    
    
    silence_segements = read_silence_segments(silence_file)

    for i in range(len(entries) - 1):
        current = entries[i]
        next_entry = entries[i + 1] 

        silenced_intervals = []

        last_word_time = current['word_time'][0][1]
        segment_end_time = current['end']
        next_start = next_entry['start']
        result, new_time = decide_state_transition(last_word_time, next_start, silence_segements)
        if result == "before_inside_0" or result == "before_before_1" or result == "inside_before_1" or result == "merge":
                entries[i + 1]["merge_allowed"] = True
        #elif result == "far":  
        #    silenced_intervals.append((last_word_time, new_time))                   

        if segment_end_time > last_word_time:            
            if result == "before_inside_0" or result == "before_before_1" or result == "inside_before_1":
                silenced_intervals.append((last_word_time, new_time))                            
            else:
                silenced_intervals.append((last_word_time, segment_end_time)) 
            
             
def reconstruct_segments_from_entries(entries):
    
    segments = []
    for entry in entries:
        sentence = entry['sentence']
        probabilities = entry['probabilities']
        word_times = entry['word_time']

        probabilities = probabilities[::-1]  # Reverse to match the order of words in the sentence
        word_times = word_times[::-1]

        # Split sentence into words (assumes each word matches index with prob & time)
        words = sentence.split()
        word_timings = []

        for i, word in enumerate(words):
            start_time, end_time = word_times[i]
            start = srt_time_to_seconds(start_time)
            end = srt_time_to_seconds(end_time) 
            probability = probabilities[i]
            word = f"{word} "
            word_timing = WordTiming(word=word, start=start, end=end, probability=probability)
            word_timings.append(word_timing)

        segments.append(Segment(words=word_timings))
    
    return segments

def clean_segments(segments, entries):
    counter = 0
    segments = segments[3:]
    new_entries = entries[3:]
    new_segments = []
    for i, segment in enumerate(segments):
        if (
            not is_avg_word_duration_ok(new_entries[i]) or
            not is_first_word_probability_ok(new_entries[i]) or
            not is_distribution_dense_enough(new_entries[i]) or
            not is_word_max_duration_ok(new_entries[i])
        ):
            print(f"Bad segment at index {i}: {segment.text}")
            counter += 1
            for word in segment.words:
                word.probability = PROBABILITY_THRESHOLD-0.01  # Set low probability for words in bad segments
        else:
            new_segments.append(segment)        
    
    print(f"Total bad segments: {counter}")   
       
    return new_segments




def is_avg_word_duration_ok(entry, threshold=AVERAGE_WORD_DURATION_THRESHOLD):
    if entry['word_count'] == 0:
        return False
    avg_duration = entry['duration'] / entry['word_count']
    return avg_duration >= threshold

def is_first_word_probability_ok(entry, threshold=PROBABILITY_THRESHOLD):
    probs = entry['probabilities']
    return probs and probs[-1] >= threshold


def is_distribution_dense_enough(entry, threshold=PROBABILITY_THRESHOLD, max_fraction=ALLOWED_ERROR_PERCENTAGE):
    probs = entry['probabilities']
    if not probs:
        return False
    low_count = sum(1 for p in probs if p < threshold)
    low_probaility_density = low_count / len(probs)
    if low_probaility_density >= max_fraction:
        return False
    return True

def is_word_max_duration_ok(entry):
    return entry['max_duration'] < MAX_WORD_DURATION

def prepare_data_massechet(audio_repo, audio_file_template, doc_repo, doc_file_template, start_page, start_search_page=2):
    
    audio_files = os.listdir(audio_repo)
    
    
    number_of_files = len(audio_files)
    index = start_search_page  
    all_datasets = []  
    while index < number_of_files + start_page:            
        audio_file = f"{audio_repo}\\{audio_file_template.format(index)}"  
        srt_file = f"{doc_repo}\\{doc_file_template.format(index)}" 
        entries = prepare_data_from_srt(srt_file)
        segments = reconstruct_segments_from_entries(entries)
        segments = clean_segments(segments, entries) 
        metadata ={
            "source_id": f"{doc_file_template.format(index)}"        
        }     
        file_dataset = prepare_training_dataset(slice_length=30, segments=segments, audio_file=audio_file, per_segment_quality_threshold=PROBABILITY_THRESHOLD, metadata=metadata)
        all_datasets.append(file_dataset)
        
        index += 1
    return all_datasets          

def prepare_data_repo(respos_dict):
    all_datasets = [] 
    for item in respos_dict:
        audio_repo = item['audio_repo']
        audio_file_template = item['audio_file_template']
        doc_repo = item['doc_repo']
        doc_file_template = item['doc_file_template']
        start_page = item.get('start_page', 2)        
        
        massechet = audio_repo.split('\\')[-1]           
        logger.info(f"Start prepare_data massechet = {massechet}")
        massechet_dataset = prepare_data_massechet(audio_repo, audio_file_template, doc_repo, doc_file_template, start_page)
        
        all_datasets.extend(massechet_dataset)
        logger.info(f"Finished prepare_data massechet = {massechet}")

    output_dataset = concatenate_all_datasets(all_datasets)    

    output_dataset.info.dataset_name = "gmara_citing"
    output_dataset.info.version = "1.0.0"

    dataset_card = create_dataset_card()
    dataset_split = split_dataset(output_dataset, 0.05)
    upload_dataset_to_hub(dataset_split, dataset_card, "shaiengel/daf-yomi-talmud-whisper-training")        
    

if __name__ == '__main__': 
    repos = [
        {'audio_repo': 'repo_audio\\brachot', 'audio_file_template': 'Bsafa_Brura-01_BR-{}.mp3', 'doc_repo': 'output_repo\\brachot\\srt_statistics', 'doc_file_template': 'Bsafa_Brura-01_BR-{}.srt', 'start_page': 2}
    ] 
    prepare_data_repo(repos)     
    
    
    # file = "output_repo\\brachot4\\srt_statistics\\Bsafa_Brura-01_BR-38.srt"
    # audio_file = "repo_audio\\brachot\Bsafa_Brura-01_BR-38.mp3"
    # silence_file = "output_repo\\brachot4\\silences\\Bsafa_Brura-01_BR-38.srt.silences"
    # entries = prepare_data_from_srt(file)
    # segments = reconstruct_segments_from_entries(entries)
    # segments = clean_segments(segments, entries)
    #audio_loader = load_audio(audio_file, slice_length=30)
    #slices = generate_slices(segments, audio_loader.get_duration(), slice_length=30, per_segment_quality_threshold=PROBABILITY_THRESHOLD)
    #slices_statistics(slices, audio_loader.get_duration())
    # metadata ={
    #     "source_id": "Bsafa_Brura-01_BR-38"        
    # }
    # all_datasets = [] 
    # file_dataset = prepare_training_dataset(slice_length=30, segments=segments, audio_file=audio_file, per_segment_quality_threshold=PROBABILITY_THRESHOLD, metadata=metadata)
    # all_datasets.append(file_dataset)    
    # output_dataset = concatenate_all_datasets(all_datasets)
    # print(output_dataset)

    # output_dataset.info.dataset_name = "gmara_citing"
    # output_dataset.info.version = "1.0.0"

    # dataset_card = create_dataset_card()
    # dataset_split = split_dataset(output_dataset, 0.05)
    # save_dataset(dataset_split, dataset_card, "Bsafa_Brura-01_BR-38")
    
    
       
    

    
