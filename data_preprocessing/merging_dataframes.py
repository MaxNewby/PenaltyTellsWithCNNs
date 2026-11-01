import pandas as pd
from pathlib import Path
from config import ROOT

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class MergingDataframes:

    def __init__(self):
        pass

    def merge_dataframes(self):

        labels = pd.read_csv(Path(ROOT) / 'penalty_data.csv',
                             skipinitialspace=True,           # file has spaces after the commas
                             dtype={'image_name': str})       # keep "18_12_22" as text, not a number
        labels['foot_right'] = (labels['foot_taken'] == 'right').astype(int)

        d = Path(ROOT) / 'image_data'
        df = (pd.read_csv(d / 'geometry_features.csv')
              .merge(pd.read_csv(d / 'pose_features.csv'), on=['nation', 'player', 'file']))
        df['image_name'] = df['file'].map(lambda f: Path(f).stem)    # "18_12_22.jpg" -> "18_12_22"

        df = df.merge(labels[['player', 'image_name', 'players_direction', 'foot_right']],
                      on=['player', 'image_name'], how='left')
        print(df['players_direction'].isna().sum(), 'images with no label')

        missing = df[df['players_direction'].isna()][['player', 'image_name']].copy()
        missing['player_in_csv'] = missing['player'].isin(set(labels['player']))
        if len(missing) > 0:
            print(f'Mismatched data between images and labels:')
            print(missing.to_string(index=False))


        return df