from config import ROOT
from pathlib import Path
from file_organiser import FileOrganiser
from data_preprocessing.angle_recording import GeometryExtractor
from data_preprocessing.pose_features import PoseFeatures
from data_preprocessing.background_remover import BackgroundRemoval
from data_preprocessing.image_resizing_and_centring import ResizeCentre
from data_preprocessing.image_silhouetting import ImageSilhouette
from data_preprocessing.merging_dataframes import MergingDataframes
from models.image_model import ImageModel
from shap_analysis.shap_explanation import ShapAnalysis
import pandas as pd
import numpy as np
import time




class MainProgram:
    def __init__(self):
        self.file_organiser = FileOrganiser()
        self.geometry_extractor = GeometryExtractor(grey=False)
        self.pose_features = PoseFeatures()
        self.background_remover = BackgroundRemoval()
        self.resize_centre = ResizeCentre()
        self.image_silhouette = ImageSilhouette(save_to_preprocessed=True)
        self.merging_dataframes = MergingDataframes()
        self.shap_analysis = ShapAnalysis()

    def main(self):

        print(self.file_organiser.check_image_count('raw'))
        print(self.file_organiser.check_image_count('preprocessed'))

        if self.file_organiser.check_image_count('raw') != self.file_organiser.check_image_count('preprocessed'):

            #reset processing_files and preprocessed files
            self.file_organiser.reset_preprocessing_files()

            #Check number of image files
            self.file_organiser.check_image_count()

            #Record runup angle and distance of each image and store it in a CSV
            self.geometry_extractor.process_all()

            #Record data on player pose of each image and store it in a CSV
            self.pose_features.process_all()

            #Remove the background from the image using Apple's vision packages
            self.background_remover.remove_backgrounds()

            #Resize player images and centre them
            self.resize_centre.process_all(size=224)

            #Silhoutte each player by turning the player white and the background black
            self.image_silhouette.process_all()

        #All metadata about each penalty, including the result, pose_features and run-up geometry are merged into one dataframe.
        df = self.merging_dataframes.merge_dataframes()

        #
        km = ImageModel()

        # merged table -> left/right shots only, with 'shot_right' and 'shot_across' added
        d = km.prepare_df(df)

        # --- Positive control: can the model see stance? (footedness is a known label) ---
        #km.cross_validate(d, target='foot_right')  # 1. grouped by taker
        #km.cross_validate(d, target='foot_right', group_col='image_name')  # 2. whole matches held out

        # --- Shuffled-label control: should sit around 0.5 ---
        for s in range(3):  # 3.
            d['foot_shuffled'] = np.random.default_rng(s).permutation(d['foot_right'].to_numpy())
        #    km.cross_validate(d, target='foot_shuffled')

        #km.cross_validate(d, target='shot_across', canonical=True)  # 4. primary: across the body vs same side
        #km.cross_validate(d, target='shot_right')  # 5. secondary: absolute left/right

        # Train model with all images on if the player shooting direction
        model, x, y = km.fit_final(d, target='foot_right')

        i = int(np.where(y == 1)[0][0])  # first right-footed taker; use y == 0 for a left-footed one
        p = float(model(x[i:i + 1], training=False).numpy().ravel()[0])

        start = time.perf_counter()
        heat = self.shap_analysis.explain_one(model, x, i)
        print(f'took {time.perf_counter() - start:.0f}s')

        self.shap_analysis.plot_one(heat, x[i], title=f'true: {"right" if y[i] else "left"}-footed, model says {p:.2f}')

        km.train_vs_test(d, target='foot_right')
        km.train_vs_test(d, target='shot_right')
        km.train_vs_test(d, target='foot_shuffled')
        km.train_vs_test(d, target='shot_across')

        #Checking for overfitting with learning curves
        head, history = km.fit_with_validation(d,target='shot_right', epochs=100)
        km.acc_vs_loss_graph(history)

        head, history = km.fit_with_validation(d, target='shot_across', epochs=100)
        km.acc_vs_loss_graph(history)

mp = MainProgram()
mp.main()


