import numpy as np
from tqdm import tqdm
from pykalman import KalmanFilter
from sklearn.decomposition import PCA
import os
from pathlib import Path
import random
from concurrent.futures import ProcessPoolExecutor

def range_fft(data: np.ndarray, N: int) -> np.ndarray:
    # Validate inputs
    if not isinstance(data, np.ndarray) or not np.iscomplexobj(data):
        raise ValueError("Input data must be a complex-valued NumPy array.")
    if not isinstance(N, int) or N <= 0:
        raise ValueError("Interpolation factor N must be a positive integer.")

    # Extract dimensions of the input data
    n_sample, n_channel = data.shape

    # Zero-padding the data
    interpolated_data = np.zeros((n_sample * N, n_channel), dtype='complex')
    interpolated_data[0:n_sample, :] = data

    # Apply a Hann window
    window = np.hanning(n_sample * N)

    # Perform FFT for each channel
    range_profile = np.zeros((n_sample * N, n_channel), dtype='complex')
    for m in range(n_channel):
        range_profile[:, m] = np.fft.fft(interpolated_data[:, m] * window, n_sample * N)

    return range_profile


def azimuth_fft(range_profile: np.ndarray) -> np.ndarray:
    # Validate input
    if not isinstance(range_profile, np.ndarray) or not np.iscomplexobj(range_profile):
        raise ValueError("Input range_profile must be a complex-valued NumPy array.")

    n_sample, n_channel = range_profile.shape

    # Define channels to use for azimuth FFT
    selected_channels = [0, 1, 2, 3, 8, 9, 10, 11]
    if max(selected_channels) >= n_channel:
        raise ValueError("Selected channels exceed the available range in range_profile.")

    # Initialize azimuth profile
    azimuth_profile = np.zeros((n_sample, 180), dtype='complex')

    # Compute azimuth FFT for each sample
    for m in range(n_sample):
        # Extract selected channels for current sample
        temp = range_profile[m, selected_channels]
        # Perform FFT and apply FFT shift
        azimuth_profile[m, :] = np.fft.fftshift(np.fft.fft(temp, 180))

    return azimuth_profile


def elevation_fft(range_profile: np.ndarray) -> np.ndarray:
    """
    Perform elevation FFT using selected channels and a 60-point FFT.

    Parameters:
        range_profile (np.ndarray): Input complex data of shape (n_sample, n_channel).

    Returns:
        np.ndarray: Elevation profile after FFT, of shape (n_sample, 60).
    """
    # Validate input
    if not isinstance(range_profile, np.ndarray) or not np.iscomplexobj(range_profile):
        raise ValueError("Input range_profile must be a complex-valued NumPy array.")

    n_sample, n_channel = range_profile.shape

    # Define channels to use for elevation FFT
    selected_channels = [9, 7]
    if max(selected_channels) >= n_channel:
        raise ValueError("Selected channels exceed the available range in range_profile.")

    # Initialize elevation profile
    elevation_profile = np.zeros((n_sample, 180), dtype='complex')

    # Compute elevation FFT for each sample
    for m in range(n_sample):
        # Extract selected channels for current sample
        temp = range_profile[m, selected_channels]
        # Perform FFT and apply FFT shift
        elevation_profile[m, :] = np.fft.fftshift(np.fft.fft(temp, 180))

    return elevation_profile


def process_frame(transposed_data: np.ndarray,
                  sample_rate: float,
                  c: float,
                  slope: float,
                  n_sample: int,
                  N: int,
                  noise: np.ndarray,
                  target_frames: int) -> np.ndarray:
    """
    Process radar frames to compute the filtered trajectory of a target in range, azimuth, and elevation.

    Parameters:
        transposed_data (np.ndarray): Transposed input radar data of shape (target_frames, n_sample, n_channel).
        sample_rate (float): Sampling rate of the radar.
        c (float): Speed of light in m/s.
        slope (float): Slope of the radar chirp in Hz/s.
        n_sample (int): Number of samples per frame.
        N (int): Interpolation factor.
        noise (np.ndarray): Noise matrix of the same shape as a single frame in transposed_data.
        target_frames (int): Total number of frames to process.

    Returns:
        np.ndarray: Filtered target trajectory data of shape (filtered_frames, 3) with columns [range, azimuth, elevation].
    """
    # Initialize storage for processed data
    tr_map = np.zeros((target_frames, 180), dtype=np.float32)
    ta_map = np.zeros((target_frames, 180), dtype=np.float32)
    te_map = np.zeros((target_frames, 180), dtype=np.float32)

    # Process each frame
    for i in range(target_frames):
        current_data = transposed_data[i, :, :] - noise

        tr_map[i, :] = np.abs(range_fft(current_data, N)[: 180]).mean(axis=1)
        ta_map[i, :] = np.abs(azimuth_fft(current_data)).mean(axis=0)
        te_map[i, :] = np.abs(elevation_fft(current_data)).mean(axis=0)

    return tr_map, ta_map, te_map

def process_single_file(file_path: str, n_sample: int, n_channel: int, slope: float, sample_rate: float, N: int,
                        c: float) -> None:
    """
    Process a single .npy file to extract and project the trajectory.
    Parameters:
        file_path (str): Path to the .npy file.
        n_sample (int): Number of samples per frame.
        n_channel (int): Number of channels per frame.
        slope (float): Radar chirp slope (Hz/s).
        sample_rate (float): Sampling rate (Hz).
        N (int): FFT interpolation factor.
        c (float): Speed of light (m/s).
    """
    dataset_root = "mmPencil_dataset/mmWave"
    try:
        data_cube = np.load(file_path)
    except Exception as e:
        print(f"[Error] Failed to load: {file_path}\n  {e}")
        return None

    if data_cube.ndim != 3 or data_cube.shape[1:] != (n_sample, n_channel):
        print(f"[Warning] Skipping due to unexpected shape: {file_path} - Shape: {data_cube.shape}")
        return None

    noise = data_cube.mean(axis=0)
    target_frames = data_cube.shape[0]

    try:
        tr_map, ta_map, te_map = process_frame(data_cube, sample_rate, c, slope, n_sample, N, noise, target_frames)

        # Extract filename components based on relative path to dataset root
        rel_path = os.path.relpath(file_path, dataset_root)
        parts = Path(file_path).parts
        if len(parts) < 4:
            print(f"[Error] Unexpected path format: {rel_path}")
            return None

        if parts[-5] == "22-Writing-Scenario":
            subject, experiment_1, experiment_2, experiment_3, word, trial_filename = parts[-6:]
            experiment = f"{experiment_1}_{experiment_2}_{experiment_3}"
        else:
            subject, experiment, word, trial_filename = parts[-4:]

        trial = trial_filename.replace('.npy', '')
        output_name = f"{experiment}_{subject}_{word}_{trial}.npy"

        output_dir = "mmPencil_dataset/mmWave_HotMap"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_name)

        combined_tensor = np.stack([tr_map, ta_map, te_map], axis=-1)

        np.save(output_path, combined_tensor)

    except Exception as e:
        print(f"[Error] Processing failed: {file_path}\n  {e}")

if __name__ == '__main__':
    # Define key parameters
    n_sample = 108  # Number of samples per frame
    n_channel = 12  # Number of channels per frame
    slope = 66.0105e12  # Radar chirp slope (Hz/s)
    sample_rate = 2e6  # Sampling rate (Hz)
    N = 4  # FFT interpolation factor
    c = 3e8  # Speed of light (m/s)

    # Define specific directories to process
    target_directories = [
        "mmPencil_dataset/mmWave"
    ]

    # Collect all .npy file paths from target directories
    npy_file_list = []
    for target_dir in target_directories:
        if not os.path.exists(target_dir):
            print(f"[Warning] Directory does not exist: {target_dir}")
            continue

        for dirpath, _, filenames in os.walk(target_dir):
            for filename in filenames:
                if filename.endswith(".npy"):
                    npy_file_list.append(os.path.join(dirpath, filename))

    print(f"Found {len(npy_file_list)} .npy files in target directories.")

    # Print directory statistics
    for target_dir in target_directories:
        dir_files = [f for f in npy_file_list if f.startswith(target_dir)]
        print(f"  {os.path.basename(target_dir)}: {len(dir_files)} files")

    print("\nStart...\n")

    # Set random seed for reproducible sampling
    random.seed(42)

    # Using ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(
            process_single_file,
            npy_file_list,
            [n_sample] * len(npy_file_list),
            [n_channel] * len(npy_file_list),
            [slope] * len(npy_file_list),
            [sample_rate] * len(npy_file_list),
            [N] * len(npy_file_list),
            [c] * len(npy_file_list)),
            total=len(npy_file_list), desc="Processing Dataset", unit="file"))

    print(
        f"\nProcessing completed! All data has been saved as (n * 100, 180, 3) tensors.")