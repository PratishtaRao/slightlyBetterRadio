import sys
import subprocess
from os import listdir
from os.path import isfile, join


def download_audio(links, out_label):
    for link in links:
        # Setting up output location
        output_location = "-o \"../../data/raw/" + out_label + "/%(title)s.%(ext)s\" "

        # youtube-dl configuration file location
        config_args = "--config-location ../../external/youtube-dl.conf"

        # subprocess call to run youtube-dl
        subprocess.run("../../external/youtube-dl " + config_args + " " + output_location + " \"" + link + "\"")


def convert_mp3_to_wav(out_label):
    out = out_label + "_wav"
    input_location = "../../data/raw/" + out_label
    output_location = "../../data/raw/" + out
    onlyfiles = [f for f in listdir(input_location) if isfile(join(input_location, f))]

    for file in onlyfiles:
        # convert the downloaded mp3 audio to a 16 bit, 16khz mono wav file
        # TODO: Create the output file first, you created it manually to store the new .wav files
        cmd = "ffmpeg -i  {input_loc} -acodec pcm_s16le -ac 1 -ar 16000  {output} ".format(input_loc=input_location + "/" + file,
                                                                                           output=output_location + "/" + file[:-4] + ".wav")
        # subprocess call to run ffmpeg
        subprocess.run(cmd)


def usage():
    print("Usage: python3 download_dataset.py input_file1 input_file2 ... input_fileN\n\n"
          "input_file is a list of youtube playlist/video links. \n Can use # for comments")


def read_links(filename):
    links = list()
    with open(filename) as f:
        for line in f:
            # Ignoring comments
            if line[0] != '#':
                links.append(line.replace("\n", ""))
    return links


def main():
    if len(sys.argv) < 2:
        usage()
    else:
        # Reading file contents and sending for download
        for filename in sys.argv[1:]:
            links = read_links(filename)

            # Extracting file name for output directory location
            out = filename.split("\\")[-1]
            download_audio(links, out)
            convert_mp3_to_wav(out)

def test_mp3_convert():
    convert_mp3_to_wav("ads")


if __name__ == '__main__':
    #main()
    test_mp3_convert()
