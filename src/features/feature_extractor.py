import os
import scipy.io.wavfile as wavfile
import numpy as np
from python_speech_features import mfcc
import pandas as pd
import gc as gc


def get_features(sub_sample, rate):
    features = mfcc(sub_sample, rate, winlen=0.1, nfft=2048)
    return features


def process_file(filename, classification_class, parent_path="../../data/raw/"):

    print("Processing file " + filename)
    # read in file

    rate, audio_data = wavfile.read(parent_path + classification_class + "/" + filename)
    # audio_data, rate = librosa.load(parent_path + classification_class + "/" + filename, sr=16000)
    # audio_data = librosa.resample(data, rate, 16000)

    # Discarding first and last 10%
    length = np.shape(audio_data)[0]
    mark = int(0.1 * length)
    audio_data = audio_data[mark:-mark]

    # split huge array data into smaller chunks
    max_size = 2000
    result = np.array_split(audio_data, len(audio_data)// max_size)
    for array_chunk in result:
        # Getting features
        features = get_features(array_chunk, rate)

        # Write out to csv file
        pd.DataFrame(features).to_csv("../../data/processed/" + classification_class + ".csv",
                                      mode="a",
                                      header=False,
                                      index=False)






    # Code to batch data and calculate features per batch
    # Not useful as MFCC already is calculated on splits

    # length = np.shape(audio_data)[0]
    # samples = int(length / rate)
    # start = 0
    # end = start + rate * 2
    # for sample_idx in range(1, samples+1):
    #     sub_sample = audio_data[start:end]
    #     features = get_features(sub_sample, rate)
    #     start = end
    #     end += rate * 2
    #     print("Test")


def main():

    raw_parent = "../../data/raw/"
    source_dirs = os.listdir(raw_parent)

    # Processing each directory
    for directory in source_dirs:
        dir_list = ['playlists', 'talk', 'ads', 'music']
        if directory not in dir_list:
            file_list = os.listdir("../../data/raw/" + directory)
            print("Processing samples of class " + directory)
            for filename in file_list:
                # Here each directory is a class, passing it along with the file
                process_file(filename, directory)
        gc.collect()



if __name__ == "__main__":
    main()
