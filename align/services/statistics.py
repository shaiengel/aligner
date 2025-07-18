
import re
from align.services.logger import init_logger, format_rtl
from align.services.logger import get_logger
from pathlib import Path
from align.services.utils import format_time

logger = get_logger()

def add_probabilties_to_srt(srt_file, words_with_probs, srt_statistics_repo):
    low_confidence_counter = 0
    # Read SRT file
    with open(srt_file, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    # Split into subtitle blocks
    entries = re.split(r'\n\n+', content.strip())
    word_index = 0
    output_lines = []
    entry_number = 0 

    result_probability_list = []

    for entry in entries:
        entry_number += 1
        lines = entry.strip().split('\n')
        if len(lines) < 3:
            continue

        index = lines[0]
        time = lines[1]
        text_lines = lines[2:]

        output_lines.append(index)
        output_lines.append(time)

        for line in text_lines:
            # Count words in this subtitle line
            word_count = len(line.strip().split())
            # Get matching probabilities
            matched_words = words_with_probs[word_index:word_index + word_count]
            probs = [round(w['probability'], 3) for w in matched_words]
            word_index += word_count

            # Write original line and probability line
            output_lines.append(line)
            probs.reverse()
            output_lines.append(str(probs))
            result_probability_list.append(probs)

            #time_ranges = [(round(float(w['start']), 3), round(float(w['end']), 3)) for w in matched_words]
            time_ranges = [
                (format_time(float(w['start'])), format_time(float(w['end'])))
                for w in matched_words
            ]
            time_ranges.reverse()  # Reverse to match reversed probs
            output_lines.append(str(time_ranges))

            low_conf_count = sum(1 for p in probs if p < 0.1)
            if low_conf_count > 2:
                logger.info(f"Low confidence words in entry_number: {entry_number} - {low_conf_count} - {text_lines}")
                low_confidence_counter += 1
            if len(probs) == 1 and low_conf_count == 1:
                logger.info(f"Low confidence 1 words in entry_number: {entry_number} - {low_conf_count} - {text_lines}")
                low_confidence_counter += 1
            if len(probs) == 2 and low_conf_count == 2:
                logger.info(f"Low confidence 2 words in entry_number: {entry_number} - {low_conf_count} - {text_lines}")
                low_confidence_counter += 1    

        output_lines.append('')  # Blank line

    logger.info(f"Low confidence words count: {low_confidence_counter}/{len(entries)}")
    # Write output file
    filename = Path(srt_file).stem   
    output_file = Path(srt_statistics_repo) / f'{filename}.srt'  
    with open(output_file, 'w', encoding='utf-8-sig') as f:
        f.write('\n'.join(output_lines))  

    return result_probability_list     

def weighted_run_score(y):
    runs = []
    current_run = 0
    
    for val_arr in y:
        counter_stat = 0
        val = 0
        for i in val_arr:
            if i < 0.1:
                counter_stat += 1
        if counter_stat > 2:
            val = 1

        if val == 1:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
                current_run = 0
    
    if current_run > 0:
        runs.append(current_run)
    
    return sum(r**2 for r in runs) / len(y) if runs else 0         

def slices_statistics(slices, audio_duration):
    counter = 0
    sum_duration = 0
    for segment in slices:
        if len(segment["segments"]) != 0:
            counter += 1
            end_time = segment["segments"][-1].get("end")
            # Fallback to second-last if missing or None
            if end_time is None:
                if len(segment) > 1:
                    duration = segment["segments"][-2]["end"] - segment["segments"][0]["start"]
                else:
                    duration = segment["segments"][0]["end"] - segment["segments"][0]["start"]
            else:
                if len(segment) > 1:
                    duration = segment["segments"][-1]["end"] - segment["segments"][0]["start"]
                else:
                    duration = segment["segments"][0]["end"] - segment["segments"][0]["start"]
                
            sum_duration += duration
    print(f"Total slices after cleaning: {counter}/{len(slices)}, total duration: {sum_duration}/{audio_duration} seconds")
    return sum_duration