from config import ROOT
from pathlib import Path
import shutil
import os

class FileOrganiser:

    def __init__(self):
        pass

    def check_image_count(self, file_name='raw'):

        base = Path(ROOT) / 'image_data' / 'Outfield_Players'
        image_count = 0

        for nation in base.iterdir():
            if not nation.is_dir():
                continue
            for player in nation.iterdir():
                file = player / f'{file_name}'
                if not file.is_dir():
                    continue
                image_count += sum(1 for f in file.iterdir() if f.is_file() and not f.name.startswith('.'))

        return image_count


    def reset_preprocessing_files(self):
        base = Path(ROOT) / 'image_data' / 'Outfield_Players'

        for preprocessing_path in base.glob('*/*/preprocessing'):
            if preprocessing_path.is_dir():
                shutil.rmtree(preprocessing_path)

        for preprocessed_path in base.glob('*/*/preprocessed'):
            if preprocessed_path.is_dir():
                shutil.rmtree(preprocessed_path)





