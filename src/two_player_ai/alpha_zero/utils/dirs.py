import os


def create_dirs(dirs):
    try:
        for directory in dirs:
            if not os.path.exists(directory):
                os.makedirs(directory)
        return True
    except Exception as e:
        exit(-1)
