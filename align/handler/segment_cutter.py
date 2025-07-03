import re
import os
import json
import ast
from datetime import datetime
from datetime import timedelta
from align.services.silence_segment_analyzer import read_silence_segments, decide_state_transition
#from align.handler.transcribe import find_silence

PROBABILTY_THRESHOLD = 0.10
MAX_WORD_DURATION = 1.6  # seconds
AVERAGE_WORD_DURATION_THRESHOLD = 0.35
ALLOWED_ERROR_PERCENTAGE = 0.416

def srt_time_to_ffmpeg_format(time_str):
    # Convert SRT format: "00:00:01,780" to FFmpeg-friendly "00:00:01.780"
    return time_str.replace(',', '.')

def get_duration(start_str, end_str):
    fmt = "%H:%M:%S.%f"
    start_dt = datetime.strptime(start_str, fmt)
    end_dt = datetime.strptime(end_str, fmt)
    return (end_dt - start_dt).total_seconds() 

def parse_srt_with_details(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    blocks = re.split(r'\n\s*\n', content.strip())
    result = []

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 4:
            continue

        interval = lines[1].strip()
        match = re.match(r'(.+?) --> (.+)', interval)
        if not match:
            continue

        start_raw, end_raw = match.groups()
        start = srt_time_to_ffmpeg_format(start_raw)
        end = srt_time_to_ffmpeg_format(end_raw)
        duration = get_duration(start, end)

        # Handle multi-line sentence
        prob_line = lines[-2].strip()
        word_time_line = lines[-1].strip()
        sentence_lines = lines[2:-2]
        sentence = ' '.join(sentence_lines).strip()
        word_count = len(sentence.split())

        probabilities = json.loads(prob_line)
        word_time = ast.literal_eval(word_time_line)

        word_intervals_line = lines[4]
        try:
            word_intervals = eval(word_intervals_line.strip())
        except Exception:
            continue

        # Convert intervals to durations
        try:
            durations = [
                (
                    datetime.strptime(end, "%H:%M:%S.%f") -
                    datetime.strptime(start, "%H:%M:%S.%f")
                ).total_seconds()
                for start, end in word_intervals
            ]
        except Exception:
            continue

        if durations:
            max_duration = max(durations)

        result.append({
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

    return result

def is_avg_word_duration_ok(entry, threshold=AVERAGE_WORD_DURATION_THRESHOLD):
    if entry['word_count'] == 0:
        return False
    avg_duration = entry['duration'] / entry['word_count']
    return avg_duration >= threshold

def is_first_word_probability_ok(entry, threshold=PROBABILTY_THRESHOLD):
    probs = entry['probabilities']
    return probs and probs[-1] >= threshold


def is_distribution_dense_enough(entry, threshold=PROBABILTY_THRESHOLD, max_fraction=ALLOWED_ERROR_PERCENTAGE):
    probs = entry['probabilities']
    if not probs:
        return False
    low_count = sum(1 for p in probs if p < threshold)
    low_probaility_density = low_count / len(probs)
    if low_probaility_density >= max_fraction:
        return False
    return True


def adjust_boundaries_old(entries):
    fmt = "%H:%M:%S.%f"

    def to_dt(time_str):
        return datetime.strptime(time_str, fmt)

    def to_str(dt_obj):
        return dt_obj.strftime(fmt)

    for i in range(len(entries) - 1):
        current = entries[i]
        next_entry = entries[i + 1]

        # Skip if next entry has invalid or weak last probability
        if not next_entry['probabilities'] or next_entry['probabilities'][-1] < PROBABILTY_THRESHOLD:
            continue

        current_end = to_dt(current['end'])
        next_start = to_dt(next_entry['start'])
        midpoint = current_end + (next_start - current_end) / 2

        # Update both entries
        current['end'] = to_str(midpoint)
        next_entry['start'] = to_str(midpoint)

        # Recompute durations
        current['duration'] = (to_dt(current['end']) - to_dt(current['start'])).total_seconds()
        next_entry['duration'] = (to_dt(next_entry['end']) - to_dt(next_entry['start'])).total_seconds()

def adjust_boundaries(entries, silence_file):
    fmt = "%H:%M:%S.%f"

    def to_seconds(time_str):
        # Replace comma with dot
        time_str = time_str.replace(',', '.')

        # Parse the time string
        dt = datetime.strptime(time_str, "%H:%M:%S.%f")

        # Convert to seconds since start of the day
        seconds = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1_000_000
        return seconds

    def to_str(seconds):
        td = timedelta(seconds=seconds)

        # Extract hours, minutes, seconds, microseconds
        total_seconds = int(td.total_seconds())
        microseconds = td.microseconds
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60

        # Format the string
        time_str = f"{hours:02}:{minutes:02}:{secs:02}.{microseconds // 1000:03}"
        return time_str
    
    silence_segements = read_silence_segments(silence_file)

    for i in range(len(entries) - 1):
        current = entries[i]
        next_entry = entries[i + 1]        

        #current_end = to_seconds(current['end'])
        current_end = to_seconds(current['word_time'][0][1])
        next_start = to_seconds(next_entry['start'])
        result, new_time = decide_state_transition(current_end, next_start, silence_segements)
        if result == "before_inside_0" or result == "before_before_1":
            #entries[i]['end'] = to_str(new_time)
            #entries[i]['duration'] = new_time - to_seconds(entries[i]['start'])
            entries[i + 1] ['start'] = to_str(new_time)
            entries[i+ 1]['duration'] = to_seconds(entries[i + 1]['end']) - new_time
        elif result == "inside_before_1":
            entries[i]['end'] = to_str(new_time)
            entries[i]['duration'] = new_time - to_seconds(entries[i]['start'])
            entries[i + 1] ['start'] = to_str(new_time)
            entries[i+ 1]['duration'] = to_seconds(entries[i + 1]['end']) - new_time
        elif result == "bad":
            entries[i + 1]["is_valid"] = False
            print(f"bad segment {i+2}")
        
        if result == "before_inside_0" or result == "before_before_1" or result == "inside_before_1" or result == "merge":
            entries[i + 1]["merge_allowed"] = True             
    

def clean_entries(entries):    
    cleaned = []
    for entry in entries:
        if (
            is_avg_word_duration_ok(entry) and
            is_first_word_probability_ok(entry) and
            is_distribution_dense_enough(entry) and
            is_word_max_duration_ok(entry)
        ):
            cleaned.append(entry)
    return cleaned

def time_str_to_dt(tstr):
    return datetime.strptime(tstr, "%H:%M:%S.%f")

def dt_to_time_str(dt):
    return dt.strftime("%H:%M:%S.%f")


def is_first_prob_low(entry):
    return entry['probabilities'] and entry['probabilities'][-1] < PROBABILTY_THRESHOLD

def is_word_max_duration_ok(entry):
    return entry['max_duration'] < MAX_WORD_DURATION

def is_total_duration_ok(entry1, entry2):
    return entry1["duration"] + entry2["duration"] < 20.0

def is_allowed_to_merge(entry):
    return entry.get('merge_allowed', False)


def merge_entries(entries):
    merged = []
    i = 0

    while i < len(entries):
        current = entries[i]
        combined = current.copy()

        while (
            i + 1 < len(entries)
            and is_allowed_to_merge(entries[i + 1])
            and not is_first_word_probability_ok(entries[i + 1])
            and is_word_max_duration_ok(combined)
            and is_word_max_duration_ok(entries[i + 1])
            and is_total_duration_ok(combined, entries[i + 1])
            and is_distribution_dense_enough(entries[i + 1])
        ):
            next_entry = entries[i + 1]

            combined['end'] = next_entry['end']
            combined['duration'] = current['duration'] + next_entry['duration']
            combined['max_duration'] = max(current['max_duration'], next_entry['max_duration'])
            combined['sentence'] += ' ' + next_entry['sentence']
            combined['probabilities'] = next_entry['probabilities'] + combined['probabilities']
            combined['word_count'] += next_entry['word_count']            

            i += 1  # Advance for chaining merge
        merged.append(combined)
        i += 1

    return merged

def rebuild_srt(entries, output_file):
    """
    Rebuilds an SRT file content from a list of merged entries.
    Each entry must contain 'start', 'end', and 'sentence'.
    """
    def format_timestamp(tstr):
        # Convert from "HH:MM:SS.ssssss" to "HH:MM:SS,mmm"
        dt = datetime.strptime(tstr, "%H:%M:%S.%f")
        return dt.strftime("%H:%M:%S,%f")[:-3]  # Trim to milliseconds

    srt_lines = []

    for idx, entry in enumerate(entries, start=1):
        start_ts = entry['start']
        end_ts = entry['end']

        srt_lines.append(str(idx))
        srt_lines.append(f"{start_ts} --> {end_ts}")
        srt_lines.append(entry['sentence'])
        srt_lines.append('')  # Blank line between entries

    with open(output_file, 'w', encoding='utf-8-sig') as f:
        f.write('\n'.join(srt_lines)) 


if __name__ == '__main__':
    file = "output_repo\\brachot4\\srt_statistics\\Bsafa_Brura-01_BR-38.srt"
    silence_file = "output_repo\\brachot4\\silences\\Bsafa_Brura-01_BR-38.srt.silences"
    #silence = find_silence("repo_audio\\brachot\Bsafa_Brura-01_BR-38.mp3")
    #print(f"{silence}")
    parsed_data = parse_srt_with_details(file)
    #adjust_boundaries1(parsed_data)
    adjust_boundaries(parsed_data, silence_file)
    parsed_data = parsed_data[3:]
    merged = merge_entries(parsed_data)
    cleaned_data = clean_entries(merged)
    rebuild_srt(cleaned_data, "output_repo\\brachot4\\fixed_srt\\Bsafa_Brura-01_BR-38.srt")

    #create_silences_file("repo_audio\\brachot", "output_repo\\brachot4\\silences")