import zipfile
import os
import sys


def extract_zip(new_dir):
    with zipfile.ZipFile('youtube.zip', 'r') as tube_zip:
        US_files = [file_name for file_name in tube_zip.namelist() if 'US' in file_name]
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        for f in US_files:
            tube_zip.extract(f, new_dir)


def clean_zip(zip_dir):
    for f in os.listdir(zip_dir):
        os.remove(zip_dir+f)
    os.rmdir(zip_dir)


if __name__ == '__main__':
    zip_dir = 'youtube/'
    if sys.argv[1] == 'extract':
        if os.path.exists(zip_dir):
            clean_zip(zip_dir)
        extract_zip(zip_dir)
    elif sys.argv[1] == 'clean':
        clean_zip(zip_dir)
    else:
        print("{} not recognized. Possible arguments are 'extract' and 'clean'."
              .format(sys.argv[1] if len(sys.argv) >= 2 else '[NO COMMAND]'))
