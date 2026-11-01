import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class PilotModels:
    GEO = ['angle_deg', 'dist_norm']
    POSE = ['tilt_diff', 'torso_lean', 'stance_width',
            'elbow_shoulder_ratio', 'shoulder_hip_width_ratio']

    def __init__(self, n_splits=5, n_repeats=20):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.feature_sets = {
            'foot only':   ['foot_right'],
            'geometry':    self.GEO,
            'pose':        self.POSE,
            'pose + foot': self.POSE + ['foot_right'],
        }
        self.models = {
            'baseline': make_pipeline(DummyClassifier(strategy='most_frequent')),
            'logreg': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                    LogisticRegression(C=0.1, max_iter=1000)),
            'forest': make_pipeline(SimpleImputer(strategy='median'),
                                    RandomForestClassifier(n_estimators=300, max_depth=3,
                                                           min_samples_leaf=3, random_state=0)),
        }

    def prepare(self, df):
        df = df[df['players_direction'].isin(['left', 'right'])].copy()   # drop the middle shots
        df['y'] = (df['players_direction'] == 'right').astype(int)
        return df

    def run(self, df):
        df = self.prepare(df)
        y, groups = df['y'], df['player']

        print(f'{len(df)} rows, {groups.nunique()} takers')
        print(pd.crosstab(df['foot_right'], df['players_direction']))
        print('Missing share in pose columns:')
        print(df[self.POSE].isna().mean().round(2))

        rows = []
        for fs_name, cols in self.feature_sets.items():
            for m_name, model in self.models.items():
                bal, acc = [], []
                for seed in range(self.n_repeats):
                    cv = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=seed)
                    res = cross_validate(model, df[cols], y, groups=groups, cv=cv,
                                         scoring=['balanced_accuracy', 'accuracy'])
                    bal.extend(res['test_balanced_accuracy'])
                    acc.extend(res['test_accuracy'])
                rows.append({'features': fs_name, 'model': m_name,
                             'balanced_acc': np.mean(bal), 'bal_std': np.std(bal),
                             'accuracy': np.mean(acc)})
        out = pd.DataFrame(rows).round(2)
        print(out.to_string(index=False))
        return out