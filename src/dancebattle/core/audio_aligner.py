import librosa
import numpy as np
from scipy.signal import correlate

def _find_offset_onset(y1, y2, sr):
    """
    [Original Method] Finds time offset using Onset Strength Envelope cross-correlation.
    Good for rhythmic alignment but less robust for complex music.
    """
    hop_length = 512
    onset_env1 = librosa.onset.onset_strength(y=y1, sr=sr, hop_length=hop_length)
    onset_env2 = librosa.onset.onset_strength(y=y2, sr=sr, hop_length=hop_length)
    
    correlation = correlate(onset_env1, onset_env2, mode='full')
    lag_index = np.argmax(correlation)
    frame_offset = lag_index - (len(onset_env2) - 1)
    
    return librosa.frames_to_time(frame_offset, sr=sr, hop_length=hop_length)

def _find_offset_chroma(y1, y2, sr):
    """
    [Advanced Method] Finds time offset using Chromagram cross-correlation.
    This is highly robust and inspired by advanced audio synchronization toolkits.
    It aligns based on harmonic content (melody and chords).
    """
    hop_length = 512
    
    # 1. Compute Chromagrams using a Constant-Q Transform (best for music)
    # This creates a (12, T) array, where 12 is the pitch classes and T is time frames.
    print("Calculating chromagram for Audio A...")
    chroma_a = librosa.feature.chroma_cqt(y=y1, sr=sr, hop_length=hop_length)
    print("Calculating chromagram for Audio B...")
    chroma_b = librosa.feature.chroma_cqt(y=y2, sr=sr, hop_length=hop_length)

    # 2. Correlate each of the 12 chroma bands and sum the results
    # This finds the best alignment across all musical notes.
    print("Calculating cross-correlation on chromagrams...")
    total_correlation = np.zeros(chroma_a.shape[1] + chroma_b.shape[1] - 1)
    
    for i in range(chroma_a.shape[0]):
        # Correlate the i-th chroma vector (e.g., all 'C' notes over time)
        # using 'fft' method for massive speed improvement.
        band_corr = correlate(chroma_a[i], chroma_b[i], mode='full', method='fft')
        total_correlation += band_corr

    # 3. Find the peak of the summed correlation
    lag_index = np.argmax(total_correlation)
    
    # 4. Convert the frame index to a time offset
    frame_offset = lag_index - (chroma_b.shape[1] - 1)
    
    return librosa.frames_to_time(frame_offset, sr=sr, hop_length=hop_length)


def find_time_offset(audio_path1: str, audio_path2: str, sample_rate: int = 22050, method: str = 'chroma') -> float:
    """
    Calculates the time offset between two audio files using a specified method.

    Args:
        audio_path1: Path to the first audio file (reference).
        audio_path2: Path to the second audio file (to be aligned).
        sample_rate: The sample rate to use for analysis.
        method: The alignment method to use.
                'chroma': (Default, Recommended) Robust alignment based on musical harmony.
                'onset': Fast alignment based on rhythm. Less accurate.

    Returns:
        The time offset in seconds. A positive value means audio2 starts after audio1.
    """
    print(f"Loading audio for alignment using '{method}' method...")
    y1, sr = librosa.load(audio_path1, sr=sample_rate)
    y2, _ = librosa.load(audio_path2, sr=sample_rate)

    if method == 'chroma':
        offset = _find_offset_chroma(y1, y2, sr)
    elif method == 'onset':
        offset = _find_offset_onset(y1, y2, sr)
    else:
        raise ValueError(f"Unknown alignment method: {method}. Available methods are 'chroma' and 'onset'.")

    print(f"Found robust time offset: {offset:.4f} seconds.")
    print(f"(A positive value means video B starts {abs(offset):.2f}s after video A)")
    
    return offset