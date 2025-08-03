import os
from pydub import AudioSegment
from pydub.generators import WhiteNoise
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

def convert_mp3_to_wav(mp3_file: str, wav_file: str, output_repo, target_sample_rate: int = 16000):
    audio = AudioSegment.from_mp3(mp3_file).set_channels(1).set_frame_rate(target_sample_rate)
    file_name = Path(wav_file).name
    output_file = Path(output_repo) / file_name
    audio.export(output_file, format="wav")
    print(f"Converted {mp3_file} to {wav_file} at {target_sample_rate} Hz")
    return output_file


def generate_white_noise(duration_ms: int, sample_rate: int = 16000, volume_db: float = -50.0) -> AudioSegment:
    noise = WhiteNoise().to_audio_segment(duration=duration_ms, volume=volume_db).set_frame_rate(sample_rate).set_channels(1)
    print(f"Generated white noise: {duration_ms} ms at {volume_db} dB and {sample_rate} Hz")
    return noise


def apply_noise_to_intervals(wav_file: str, noise: AudioSegment, intervals: list[tuple[int, int]]):
    audio = AudioSegment.from_wav(wav_file).set_channels(1).set_frame_rate(16000)
    sample_rate = audio.frame_rate
    duration_ms = len(audio)

    # Prepare silent noise track
    masked_noise = AudioSegment.silent(duration=duration_ms, frame_rate=sample_rate).set_channels(1)

    for start_ms, end_ms in intervals:
        start_ms = max(0, start_ms)
        end_ms = min(duration_ms, end_ms)
        noise_slice = noise[start_ms:end_ms]
        masked_noise = masked_noise.overlay(noise_slice, position=start_ms)

    # Mute the original audio in those intervals
    for start_ms, end_ms in intervals:
        audio = audio[:start_ms] + AudioSegment.silent(duration=(end_ms - start_ms), frame_rate=sample_rate) + audio[end_ms:]

    # Overlay noise on muted audio
    output = audio.overlay(masked_noise)

    # Export at 16kHz WAV
    output.set_frame_rate(16000).export(wav_file, format="wav")
    print(f"apply_noise_to_intervals to {wav_file}")

def remove_audio_from_interval(wav_file: str, start_ms: int, end_ms: int) -> AudioSegment:
    white_noise = generate_white_noise(end_ms - start_ms)
    apply_noise_to_intervals(wav_file, white_noise, [(start_ms, end_ms)])    


if __name__ == '__main__':
    file = "output_repo\\brachot4\\fixed_srt\\Bsafa_Brura-01_BR-38.srt"
    audio_file = "repo_audio\\brachot\Bsafa_Brura-01_BR-38.mp3"
    output_folder = "output_repo\\brachot4\\segments"  
    cut_audio_segments(audio_file, file, output_folder)         