import os
from pydub import AudioSegment
import re
from align.services.utils import create_folder
from pathlib import Path
from datetime import datetime

def srt_time_to_milliseconds(time_str):
    # Replace comma with dot for microseconds
    time_fixed = time_str.replace(',', '.')
    dt = datetime.strptime(time_fixed, "%H:%M:%S.%f")
    ms = (dt.hour * 3600 + dt.minute * 60 + dt.second) * 1000 + dt.microsecond // 1000
    return ms

def cut_audio_segments(audio_file, srt_file, output_folder):
    
    # Load the MP3 audio
    audio = AudioSegment.from_mp3(audio_file)

    # Read and parse the SRT file
    with open(srt_file, "r", encoding="utf-8-sig") as f:
        content = f.read()

    # Split by double newlines to get each subtitle block
    entries = content.strip().split("\n\n")

    # Regex pattern for time extraction
    time_pattern = re.compile(r'(.+?) --> (.+)')
    

    folder = f"{output_folder}\\{Path(audio_file).stem}"
    create_folder(folder)

    for entry in entries:
        lines = entry.strip().split("\n")
        if len(lines) >= 3:
            index = lines[0].strip()
            times = lines[1].strip()
            text_lines = lines[2:]
            text = " ".join(text_lines).strip()
            match = re.findall(time_pattern, times)

            if len(match) == 1:
                start = srt_time_to_milliseconds(match[0][0])
                end = srt_time_to_milliseconds(match[0][1])                

                # Slice the audio
                segment = audio[start:end]

                # Export the segment
                output_audio_path  = os.path.join(folder, f"{index}.mp3")
                segment.export(output_audio_path , format="mp3")
                
                # Export the text file
                output_text_path = os.path.join(folder, f"{index}.txt")
                with open(output_text_path, "w", encoding="utf-8-sig") as text_file:
                    text_file.write(text)

                print(f"Exported {output_audio_path} and {output_text_path}")

if __name__ == '__main__':
    file = "output_repo\\brachot4\\fixed_srt\\Bsafa_Brura-01_BR-38.srt"
    audio_file = "repo_audio\\brachot\Bsafa_Brura-01_BR-38.mp3"
    output_folder = "output_repo\\brachot4\\segments"  
    cut_audio_segments(audio_file, file, output_folder)         