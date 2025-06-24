
import stable_whisper
import datetime
from pydub import AudioSegment, silence
from pathlib import Path
from transformers import pipeline
from align.services.audio_utils import preprocess_audio
from transformers import pipeline
from faster_whisper import WhisperModel

import librosa
import numpy as np

import warnings

from align.services.logger import get_logger

logger = get_logger()

def get_model():
    model = stable_whisper.load_model('base', device='cuda')    
    return model

def convert_audio(audio_file):
    audio_data, sr = librosa.load(audio_file, sr=16000, mono=True)
    audio_data = audio_data.astype(np.float32)
    return audio_data


def audio_to_text_aligner(model, audio_data, text, vad=False):
    captured_warnings = []
    def warning_collector(message, category, filename, lineno, file=None, line=None):
        captured_warnings.append(str(message))
    
    original_showwarning = warnings.showwarning
    warnings.showwarning = warning_collector
    
    result = model.align(audio_data, text, language='he', vad=vad)  # Use VAD if needed 

    warnings.showwarning = original_showwarning 

    for warning in captured_warnings:
        logger.warning(f"{warning}")

    return result, captured_warnings

def clean_alignment_response(response):
    print("Cleaning alignment response...")
    last_probability = 1.0
    for i, segment in enumerate(response):
       delta = segment.end - segment.start
       if len(segment.words) > 0:
        start_probability = segment.words[0].probability 
        if start_probability < 0.1 and last_probability < 0.1:
           print(f"Removing segment {i} with low start probability: {start_probability} and last probability: {last_probability}")
           response.remove_segment(i)
           last_probability = 1
           continue
        else:
           last_probability = segment.words[-1].probability

        number_of_words = len(segment.words)
        counter_of_low_prob = 0
        for word in segment.words:
            if word.probability < 0.1:
                counter_of_low_prob += 1
        if number_of_words == 1 and counter_of_low_prob == 1:
            print(f"Removing segment {i} with only one low probability word: {counter_of_low_prob}")
            response.remove_segment(i)
            last_probability = 1
            continue
        elif number_of_words == 2 and counter_of_low_prob == 2:
            print(f"Removing segment {i} with two low probability words: {counter_of_low_prob}")
            response.remove_segment(i)
            last_probability = 1
            continue                 
        elif counter_of_low_prob >= 3:
            print(f"Removing segment {i} with too many low probability words: {counter_of_low_prob}")
            response.remove_segment(i)
            last_probability = 1
            continue 
           
        number_of_words = len(segment.words)
        delta = segment.end - segment.start
        avg = delta / number_of_words if number_of_words > 0 else 0
        if avg <= 0.2:
            print(f"Removing segment {i} with average duration {avg:.2f} seconds")
            last_probability = 1
            response.remove_segment(i)

    return response

        
        
    return response

def write_to_srt(result, audio_file, output_folder):
    filename = Path(audio_file).stem   
    output_file = Path(output_folder) / f'{filename}.srt'  
    result = result.merge_by_gap(min_gap = 0.2) 
    result = result.split_by_duration(max_dur = 20) 
    
    #result.adjust_gaps()  
    result.to_srt_vtt(f'{output_file}', word_level=False)
    return output_file
    

def audio_to_text(audio_file, text, output_folder = "output"):

    model = stable_whisper.load_model('base', device='cuda')    
    result = model.align(audio_file, text, language='he')     
    filename = Path(audio_file).stem   
    output_file = Path(output_folder) / f'{filename}.srt'  
    result.to_srt_vtt(f'{output_file}', word_level=False)
    return output_file, result
    

def audio_to_text_ivirit(audio_file, text):
    model = stable_whisper.load_hf_whisper('ivrit-ai/whisper-large-v3', device='cuda')
    result = model.align(audio_file, text, language='he')    
    filename = Path(audio_file).stem 
    result.to_srt_vtt(f'{filename}.srt', word_level=False)

def audio_to_text_all(audio_file, text):
    model = stable_whisper.load_model('base', device='cuda')
    result = model.align(audio_file, text, language='he') 
    filename = Path(audio_file).stem   
    result.save_as_json(f'{filename}.json')
    result.to_srt_vtt(f'{filename}.srt', word_level=False)
    result.to_srt_vtt(f'{filename}.vtt', word_level=False)

def audio_to_transcribe_ivrit(audio_file):
    model = stable_whisper.load_hf_whisper('ivrit-ai/whisper-large-v3', device='cuda')
    result = model.transcribe(audio_file, language='he')    
    filename = Path(audio_file).stem 
    result.to_srt_vtt(f'{filename}.srt', word_level=False)

def audio_to_transcribe_ivrit_hf(audio_file):
    print(datetime.datetime.now())
    preprocess_audio(audio_file, "temp.wav")
    pipe = pipeline("automatic-speech-recognition", model="ivrit-ai/whisper-large-v3", device=1)
    result = pipe("temp.wav", generate_kwargs={"language": "he"})
    print(datetime.datetime.now())
    print(result["text"])  



def audio_to_transcribe_fast(audio_file):
    
    print(datetime.datetime.now())
    #preprocess_audio(audio_file, "temp.wav")
    model = WhisperModel("ivrit-ai/whisper-large-v3-ct2", device="cuda", compute_type="int8_float16")
    segments, info = model.transcribe(audio_file, language="he", beam_size=5 )   
    print(datetime.datetime.now())
    
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")


    

def find_silence(audio_file):
    # Load the MP3 (sample rate doesn't matter here)
    audio = AudioSegment.from_mp3(audio_file)

    # Optional: Convert to mono and 16kHz (to match Whisper expectations)
    #audio = audio.set_channels(1).set_frame_rate(16000)

    # Detect silence
    silences = silence.detect_silence(audio, min_silence_len=400, silence_thresh=-40)

    # Convert to seconds
    silence_segments = [(start / 1000, end / 1000) for start, end in silences]  
    print("Silence segments (in seconds):", silence_segments)  