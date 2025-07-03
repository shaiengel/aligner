from datetime import datetime
import os
import matplotlib.pyplot as plt
from collections import Counter

PROBABILTY_THRESHOLD = 0.10

def time_to_seconds(time_str):
    """Convert a timestamp (HH:MM:SS.sss) to total seconds."""
    return sum(x * float(t) for x, t in zip([3600, 60, 1], time_str.split(":")))


def parse_file(filepath, durations_list):
    total_duration = 0
    valid_word_count = 0

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = [line.strip() for line in f if line.strip()]

    for i in range(0, len(lines), 5):
        try:
            prob_line = lines[i + 3]
            time_ranges_line = lines[i + 4]

            probabilities = eval(prob_line)
            time_ranges = eval(time_ranges_line)

            if not isinstance(probabilities, list) or not isinstance(time_ranges, list):
                continue

            for prob, (start, end) in zip(probabilities, time_ranges):
                if prob >= PROBABILTY_THRESHOLD:
                    duration = time_to_seconds(end) - time_to_seconds(start)
                    durations_list.append(duration)
                    total_duration += duration
                    valid_word_count += 1

        except (IndexError, SyntaxError, TypeError, ValueError) as e:
            print(f"Skipping block in {filepath} at line {i}: {e}")
            continue

    print(f"Average word time of {filepath}: {total_duration / valid_word_count:.3f} seconds")        
    return total_duration, valid_word_count


def average_across_folder(folder_path):
    grand_total_duration = 0
    grand_valid_word_count = 0
    all_durations = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.srt'):
            filepath = os.path.join(folder_path, filename)
            file_duration, file_word_count = parse_file(filepath, all_durations)
            grand_total_duration += file_duration
            grand_valid_word_count += file_word_count

    average = grand_total_duration / grand_valid_word_count if grand_valid_word_count else 0.0
    return average, all_durations

def plot_histogram(durations):
    plt.figure(figsize=(10, 5))
    plt.hist(durations, bins=30, color='skyblue', edgecolor='black')
    plt.title("Histogram of Word Durations (prob ≥ 0.1)")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_duration_counts(durations, decimal_places=2):
    # Round durations to reduce floating-point noise
    rounded = [round(d, decimal_places) for d in durations]
    
    # Count how many times each rounded duration appears
    duration_counts = Counter(rounded)

    # Sort by duration
    sorted_durations = sorted(duration_counts.items())

    # Split into two lists: x (duration) and y (count)
    x_vals, y_vals = zip(*sorted_durations)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='teal')
    plt.title(f"Word Duration Frequency (rounded to {decimal_places} decimal places)")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Number of Words")
    plt.grid(True)
    plt.tight_layout()
    plt.show()    

if __name__ == '__main__':
    file = "output_repo\\brachot4\\srt_statistics\\Bsafa_Brura-01_BR-38.srt"
    audio_file = "repo_audio\\brachot\\Bsafa_Brura-01_BR-38.mp3" 
    folder_path = "output_repo\\brachot4\\srt_statistics"
    average, all_durations = average_across_folder(folder_path)     
    print(f"Average word time: {average:.3f} seconds")
    #plot_histogram(all_durations)
    plot_duration_counts(all_durations)