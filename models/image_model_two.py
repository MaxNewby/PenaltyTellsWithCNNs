from config import ROOT
from pathlib import Path
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import cross_val_score, train_test_split
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import shap


class ImageModelTwo:

    def __init__(self, size=96):
        self.size = size
        self.base = Path(ROOT) / 'image_data' / 'Outfield_Players'

    def load_image_data(self, df, target='y'):
        X_list = []
        for r in df.itertuples():
            path = self.base / r.nation / r.player / 'preprocessed' / f'{Path(r.file).stem}.png'
            img = tf.keras.utils.load_img(path, target_size=(self.size, self.size),
                                          interpolation='bilinear')
            X_list.append(tf.keras.utils.img_to_array(img))
        X_data = np.array(X_list)                       # (n, size, size, 3), values 0-255
        y_data = df[target].to_numpy().astype('float32')
        groups = df['player'].to_numpy()
        return X_data, y_data, groups

    def build_model(self):
        model = models.Sequential([
            layers.Input((self.size, self.size, 3)),
            layers.Rescaling(1. / 255),
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid'),    # one output: probability of class 1
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def cross_validate(self, df, target='y', epochs=30, n_repeats=3, group_col='player'):
        df = df.reset_index(drop=True)
        X, y, _ = self.load_image_data(df, target)
        groups = df[group_col].to_numpy()
        X_flip = X[:, :, ::-1, :]  # mirror left-right
        oof = np.zeros((n_repeats, len(df)))

        for seed in range(n_repeats):
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            for tr, te in cv.split(X, y.astype(int), groups):
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(seed)
                model = self.build_model()
                model.fit(np.concatenate([X[tr], X_flip[tr]]),
                          np.concatenate([y[tr], 1 - y[tr]]),  # mirror swaps the label
                          epochs=epochs, batch_size=8, verbose=0)
                oof[seed, te] = model(X[te], training=False).numpy().ravel()

        p = oof.mean(axis=0)
        print(f'[{target}, grouped by {group_col}] overall AUC: {roc_auc_score(y, p):.2f}')
        if target == 'y':
            for foot, name in ((1, 'right-footed'), (0, 'left-footed')):
                m = (df['foot_right'] == foot).to_numpy()
                if len(set(y[m])) == 2:
                    print(f'  AUC within {name} takers ({m.sum()} rows): {roc_auc_score(y[m], p[m]):.2f}')
        return p