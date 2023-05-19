import os
import librosa
import numpy as np
from tqdm import tqdm
from pathlib import Path
from abc import ABC, abstractmethod

# TODO: Cargar los hiperparametros del prepro desde el yaml

# Mel-spectrogram parameters
N_FFT = 2048
HOP_LENGTH = 512

# Audio parameters
AUDIO_SR = 22050
AUDIO_LEN = 30
N_CHUNKS = 5


class Extractor(ABC):
    @abstractmethod
    def extract(self, signal: np.ndarray) -> np.ndarray:
        pass


class MelSpectrogramExtractor(Extractor):
    def __init__(
        self, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH, sr: int = AUDIO_SR
    ) -> np.ndarray:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr

    def extract(self, signal: np.ndarray) -> np.ndarray:
        """Method for extract the MelSpectrogram given a signal in np.ndarray
        type

        Parameters
        ----------
        signal : np.ndarray
            Signal to be processed

        Returns
        -------
        np.ndarray
            An 2D np.ndarray with the Mel-Spectrogram of the signal
        """

        mel_spectrogram = librosa.feature.melspectrogram(
            y=signal, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )

        return librosa.power_to_db(mel_spectrogram)


class DataManager:
    def load_audio(self, audio_path: str, sr: int = AUDIO_SR) -> np.ndarray:
        """Method to load an audio signal given the path (audio_path) and the
        sample rate (sr)

        Parameters
        ----------
        audio_path : str
            Path of the audio signal
        sr : int, optional
            Sample rate of the audio, by default AUDIO_SR

        Returns
        -------
        np.ndarray
            Array containing the audio signal information
        """

        audio_array, _ = librosa.load(audio_path, sr=sr)

        return audio_array

    def dataset_tree(self, dataset_path: str):
        """Method to get the structure information (folders, subfolders, files)
        for a given path (dataset_path).

        Parameters
        ----------
        dataset_path : str
            Path of the dataset

        Returns
        -------
        generator
            Generator object with the tree information of dataset_path
        """

        return os.walk(dataset_path)

    def save_array_in_npz(self, path: str, array: np.ndarray) -> None:
        """Method to save an array in numpy's compressed format

        Parameters
        ----------
        path : str
            Path to be saved the array in .npz format
        array : np.ndarray
            Array to be saved
        """

        np.savez_compressed(file=path, spectogram=array)


class AudioPreProcessor:
    def __init__(
        self,
        dataset_path: str,
        new_dataset_path: str,
        extractor: Extractor,
        data_manager: DataManager = DataManager(),
    ):
        self.dataset_path = Path(dataset_path)
        self.new_dataset_path = Path(new_dataset_path)
        self.data_manager = data_manager
        self.extractor = extractor

    def _set_audio_properties(
        self,
        audio_len: int = AUDIO_LEN,
        audio_sr: int = AUDIO_SR,
        n_chunks: int = N_CHUNKS,
    ):
        """Method to set audio properties of the audio files

        Parameters
        ----------
        audio_len : int, optional
            Length of the audios in samples, by default AUDIO_LEN
        audio_sr : int, optional
            Sample rate of the audios, by default AUDIO_SR
        n_chunks : int, optional
            Number of chunks the audios will be divided, by default N_CHUNKS
        """

        self.audio_samples = audio_len * audio_sr
        self.chunk_samples = int(self.audio_samples / n_chunks)
        self.n_chunks = n_chunks
        self.expected_time_bins = int(
            np.ceil(self.chunk_samples / self.extractor.hop_length)
        )

    def _preprocess_single_audio(self, audio_path: str):
        """Method to process only one audio file given the audio path.

        Parameters
        ----------
        audio_path : str
            Path of the audio to be preprocessed

        Returns
        -------
        list
            List of np.ndarrays containing de mel spectrograms of each chunk
        """

        # Load audio
        audio_array = self.data_manager.load_audio(audio_path)
        mel_specs = []

        # Iterate through each chunk
        for i in range(self.n_chunks):
            chunk_i = audio_array[self.chunk_samples * i : self.chunk_samples * (i + 1)]

            # Extract mel spectrogram for each chunk
            mel_spec_i = self.extractor.extract(chunk_i)

            # Check if the time bins of mel_spec_i are the expected
            if mel_spec_i.shape[1] == int(self.expected_time_bins):
                mel_specs.append(mel_spec_i)

        return mel_specs

    def _save_mel_specs_chunks(
        self, path: str, file: str, mel_specs: np.ndarray
    ) -> None:
        for i, chunk_i in enumerate(mel_specs):
            mel_npz_name = f"{file[:-4]}-{i}.npz"
            mel_npz_path = path / mel_npz_name
            self.data_manager.save_array_in_npz(path=mel_npz_path, array=chunk_i)

    def preprocess(
        self,
        audio_len: int = AUDIO_LEN,
        audio_sr: int = AUDIO_SR,
        n_chunks: int = N_CHUNKS,
    ):
        """Method to preprocess the whole dataset with the especifications given

        Parameters
        ----------
        audio_len : int, optional
            Length of the audios in samples, by default AUDIO_LEN
        audio_sr : int, optional
            Sample rate of the audios, by default AUDIO_SR
        n_chunks : int, optional
            Number of chunks the audios will be divided, by default N_CHUNKS
        """

        # Set audio properties
        self._set_audio_properties(audio_len, audio_sr, n_chunks)

        # Generating new path to save the .npz
        os.makedirs(self.new_dataset_path, exist_ok=True)

        # Iterate through dataset_path structure
        for dirpath, _, files in self.data_manager.dataset_tree(self.dataset_path):
            # To avoid being in dataset_path
            if dirpath is not str(self.dataset_path):
                # Obtaining genre from folders name
                genre = dirpath.split("/")[-1]
                new_genre_path = self.new_dataset_path / genre
                os.makedirs(new_genre_path, exist_ok=True)
                print(f"\nProcessing {genre} genre")
                files_ignored = []

                for file in tqdm(files):
                    try:
                        audio_path = os.path.join(dirpath, file)
                        # Preprocess file
                        audio_mel_specs = self._preprocess_single_audio(audio_path)
                        # Saving chunks in .npz
                        self._save_mel_specs_chunks(
                            path=new_genre_path, file=file, mel_specs=audio_mel_specs
                        )

                    # If some audio files were corrupted, it will be ignored
                    except:
                        files_ignored.append(file)

                if files_ignored != []:
                    print(f"[WARNING] Skipped {files_ignored}: Could not be processed")


if __name__ == "__main__":
    dataset_path = "dataset/genres_original"
    new_dataset_path = "dataset/genres_mel_npz"

    extractor = MelSpectrogramExtractor()
    preprocessor = AudioPreProcessor(
        dataset_path=dataset_path,
        new_dataset_path=new_dataset_path,
        extractor=extractor,
    )

    preprocessor.preprocess()
