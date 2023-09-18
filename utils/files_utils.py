#!/usr/bin/env python3

import os

class FilesUtils():
    def __init__(self):
        pass

    def fileExists(self, file_path):
        if os.path.isfile(file_path):
            return True
        return False