from config import ROOT
from pathlib import Path
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split, GroupShuffleSplit
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import shap
import matplotlib.pyplot as plt



class ImageModel():

    def __init__(self, stage='silhouette', size=224):
        self.stage, self.size = stage, size
        self.base = Path(ROOT) / 'image_data' / 'Outfield_Players'
        self.backbone = tf.keras.applications.ResNet50(
            include_top=False, weights='imagenet', pooling='avg',
            input_shape=(size, size, 3))
        self.backbone.trainable = False

    def prepare_df(self, df):
        df = df[df['players_direction'].isin(['left', 'right'])].copy()
        df['shot_right'] = (df['players_direction'] == 'right').astype(int)       # absolute: 1 = went right
        df['shot_across'] = (df['shot_right'] != df['foot_right']).astype(int)         # relative: 1 = across the body, direction they shot != the foot used to shoot
        return df

    def load(self, df):
        imgs = []
        for r in df.itertuples():
            p = (self.base / r.nation / r.player / 'preprocessing'
                 / self.stage / f'{Path(r.file).stem}.png')
            im = Image.open(p).convert('RGB').resize((self.size, self.size))
            imgs.append(np.asarray(im, dtype='float32'))
        x = tf.keras.applications.resnet50.preprocess_input(np.stack(imgs))
        return x, x[:, :, ::-1, :]  # original, mirrored left-right

    def embed(self, x):
        return self.backbone.predict(x, batch_size=16, verbose=0)

    @staticmethod
    def build_head(dim=2048):
        head = tf.keras.Sequential([
            tf.keras.Input((dim,)),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01))])
        head.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
        return head


    def cross_validate(self, df, target='shot_right', epochs=100, n_repeats=5,
                       group_col='player', canonical=False):
        df = df.reset_index(drop=True)
        x, x_flip = self.load(df)
        if canonical:                                   # make every taker look right-footed
            left_footed = (df['foot_right'] == 0).to_numpy()
            x = x.copy()
            x[left_footed] = x_flip[left_footed]
        e = self.embed(x)
        e_flip = None if canonical else self.embed(x_flip)

        y = df[target].to_numpy().astype('float32')
        groups = df[group_col].to_numpy()
        oof = np.zeros((n_repeats, len(df)))

        for seed in range(n_repeats):
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            for tr, te in cv.split(e, y.astype(int), groups):
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(seed)
                head = self.build_head(e.shape[1])
                if canonical:
                    X_tr, y_tr = e[tr], y[tr]
                    w = {0: 0.5 / max((y_tr == 0).mean(), 1e-6),
                         1: 0.5 / max((y_tr == 1).mean(), 1e-6)}    # balance the classes
                else:
                    X_tr = np.vstack([e[tr], e_flip[tr]])
                    y_tr = np.concatenate([y[tr], 1 - y[tr]])       # mirror swaps the label
                    w = None
                head.fit(X_tr, y_tr, epochs=epochs, batch_size=16, verbose=0, class_weight=w)
                oof[seed, te] = head(e[te], training=False).numpy().ravel()

        p = oof.mean(axis=0)
        print(f'[{target}, canonical={canonical}, grouped by {group_col}] '
              f'overall AUC: {roc_auc_score(y, p):.2f}')
        if target in ('shot_right', 'shot_across'):
            for foot, name in ((1, 'right-footed'), (0, 'left-footed')):
                m = (df['foot_right'] == foot).to_numpy()
                if len(set(y[m])) == 2:
                    print(f'  AUC within {name} takers ({m.sum()} rows): {roc_auc_score(y[m], p[m]):.2f}')
        return p


    def fit_with_validation(self, df, target='shot_right',
                            epochs=100, validation_size=0.2):

        df = df.reset_index(drop=True)
        x, x_flip = self.load(df)

        e, e_flip = self.embed(x), self.embed(x_flip)
        y = df[target].to_numpy().astype('float32')
        groups = df['player'].to_numpy()

        # Split by player, not individual images
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=validation_size,
            random_state=0
        )

        train_idx, val_idx = next(
            splitter.split(e, y, groups)
        )

        # Augment training data with mirrored images
        X_train = np.vstack([e[train_idx], e_flip[train_idx]])
        y_train = np.concatenate([y[train_idx], 1 - y[train_idx]])

        # Validation uses original, unmirrored images
        X_val = e[val_idx]
        y_val = y[val_idx]

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(0)

        head = self.build_head(e.shape[1])

        history = head.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=16,
            verbose=0
        )

        return head, history


    def fit_final(self, df, target='shot_right', epochs=100):
        """Fit on all rows; return a backbone+head model and inputs, for SHAP."""
        df = df.reset_index(drop=True)
        x, x_flip = self.load(df)
        e, e_flip = self.embed(x), self.embed(x_flip)
        y = df[target].to_numpy().astype('float32')
        tf.keras.utils.set_random_seed(0)
        head = self.build_head(e.shape[1])
        head.fit(np.vstack([e, e_flip]), np.concatenate([y, 1 - y]),
                 epochs=epochs, batch_size=16, verbose=0)
        inputs = tf.keras.Input((self.size, self.size, 3))
        model = tf.keras.Model(inputs, head(self.backbone(inputs, training=False)))
        return model, x, y

    def evaluate_holdout(self, train_df, test_df, target='shot_right', epochs=100):
        """Fit on the development set, score once on images the model has never seen."""
        model, _, _ = self.fit_final(train_df, target, epochs)
        test_df = test_df.reset_index(drop=True)
        x_test, _ = self.load(test_df)
        p = model.predict(x_test, batch_size=16, verbose=0).ravel()
        y_test = test_df[target].to_numpy()
        print(f'[{target}] held-out AUC on {len(test_df)} new rows: {roc_auc_score(y_test, p):.2f}')
        if target == 'shot_right':
            for foot, name in ((1, 'right-footed'), (0, 'left-footed')):
                m = (test_df['foot_right'] == foot).to_numpy()
                if len(set(y_test[m])) == 2:
                    print(f'  AUC within {name} takers ({m.sum()} rows): {roc_auc_score(y_test[m], p[m]):.2f}')
        return p

    def train_vs_test(self, df, target='foot_right', epochs=100, group_col='player', seed=0):
        df = df.reset_index(drop=True)
        x, x_flip = self.load(df)
        e, e_flip = self.embed(x), self.embed(x_flip)
        y = df[target].to_numpy().astype('float32')
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        train_aucs, p_test = [], np.zeros(len(df))
        for tr, te in cv.split(e, y.astype(int), df[group_col]):
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            head = self.build_head(e.shape[1])
            head.fit(np.vstack([e[tr], e_flip[tr]]), np.concatenate([y[tr], 1 - y[tr]]),
                     epochs=epochs, batch_size=16, verbose=0)
            train_aucs.append(roc_auc_score(y[tr], head(e[tr], training=False).numpy().ravel()))
            p_test[te] = head(e[te], training=False).numpy().ravel()
        print(f'[{target}] train AUC {np.mean(train_aucs):.2f}  |  held-out AUC {roc_auc_score(y, p_test):.2f}')

    def acc_vs_loss_graph(self, history):

        history_dict = history.history

        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        accuracy = history_dict['accuracy']
        val_accuracy = history_dict['val_accuracy']

        epochs = range(1, len(loss) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss
        axes[0].plot(epochs, loss, 'bo-', label='Training Loss')
        axes[0].plot(epochs, val_loss, 'ro-', label='Validation Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy
        axes[1].plot(epochs, accuracy, 'bo-', label='Training Accuracy')
        axes[1].plot(epochs, val_accuracy, 'ro-', label='Validation Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()