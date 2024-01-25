import os

def get_hashed_files(hash, dir):
    files = os.listdir(dir)
    return [file for file in files if hash in file]
