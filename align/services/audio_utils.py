import librosa
import soundfile as sf
import os

SAMPLING_RATE = 16000
def preprocess_audio(input_path: str, output_path: str, max_duration_sec: int = 30):
    """
    Resample audio to 16 kHz mono and truncate to the first 30 seconds.

    Args:
        input_path (str): Path to the input audio file.
        output_path (str): Path to save the processed audio file.        
        max_duration_sec (int): Maximum duration in seconds to keep (default is 30).
    """
    # Load and resample audio
    y, _ = librosa.load(input_path, sr=SAMPLING_RATE, mono=True)

    # Truncate to max duration (e.g. 30 sec)
    max_samples = SAMPLING_RATE * max_duration_sec
    y = y[:max_samples]

    # Save as WAV
    sf.write(output_path, y, SAMPLING_RATE)
    print(f"Saved: {output_path} ({len(y)/SAMPLING_RATE:.2f} seconds)")