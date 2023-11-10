import numpy as np
import pandas as pd
import scipy as sp
import sklearn as sk
import os


class Kinematics_Dataframe:
    def __init__(self):
        pass

    def initialize_dataframe(self):
        self.df = pd.DataFrame()

    def add_column(self, column_name, column_data):
        dict = {"new_col": column_name}
        self.df = self.df.assign(new_col=column_data)
        self.df = self.df.rename(columns=dict)

    def add_kinematics_from_study(self, data_home_dir, studies):
        if type(studies) == list:
            # run some list things here
            pass
        else:
            study_dir = data_home_dir + "/" + studies
            for pat in [
                x for x in os.listdir(study_dir) if os.path.isdir(study_dir + "/" + x)
            ]:
                pat_dir = study_dir + "/" + pat
                for sess in [
                    x for x in os.listdir(pat_dir) if os.path.isdir(pat_dir + "/" + x)
                ]:
                    sess_dir = pat_dir + "/" + sess
                    for mvt in [
                        x
                        for x in os.listdir(sess_dir)
                        if os.path.isdir(sess_dir + "/" + x)
                    ]:
                        pass

    def initialize_study_directories_into_dataframe(self, HOME_DIR, studies):
        init_cols = [
            "Study",
            "Patient",
            "Session",
            "Movement",
            "Side",
            "Image Index",
            "fem",
            "tib",
            "kin",
        ]
        self.df = pd.DataFrame(columns=init_cols)
        for study in studies:
            study_dir = HOME_DIR + "/" + study + "/"

            # Lima folders have a bit of a different structure
            study_org_dir = study_dir  
            # Looping through patients within each study using list comprehension
            # We want to make sure that we are only grabbing directories
            for pat_id in [
                x
                for x in os.listdir(study_org_dir)
                if os.path.isdir(study_org_dir + "/" + x)
            ]:
                pat_dir = study_org_dir + "/" + pat_id

                # looping through each session using list comprehension
                for sess_id in [
                    x for x in os.listdir(pat_dir) if os.path.isdir(pat_dir + "/" + x)
                ]:
                    sess_dir = pat_dir + "/" + sess_id

                    for mvt_id in [
                        x
                        for x in os.listdir(sess_dir)
                        if (os.path.isdir(sess_dir + "/" + x))
                    ]:
                        mvt_dir = sess_dir + "/" + mvt_id
                        idx = 1

                        # Now to get the kinematics for each of the kinematics files in this

                        fem = mvt_dir + "/fem.jts"
                        tib = mvt_dir + "/tib.jts"
                        kin = mvt_dir + "/kin_gt.csv"
                        side = (
                            "right"
                            if os.path.exists(mvt_dir + "/right.txt")
                            else "left"
                        )

                        if not os.path.exists(fem):
                            fem = mvt_dir + "/fem.jtak"
                        if not os.path.exists(tib):
                            tib = mvt_dir + "/tib.jtak"
                        if not os.path.exists(kin):
                            continue

                        try:
                            fem_vals = np.loadtxt(fem, skiprows=2, delimiter=",")
                        except:
                            fem_vals = np.loadtxt(
                                fem,
                                skiprows=2,
                                delimiter=",",
                                usecols=(0, 1, 2, 3, 4, 5),
                            )

                        try:
                            tib_vals = np.loadtxt(tib, skiprows=2, delimiter=",")
                        except:
                            tib_vals = np.loadtxt(
                                tib,
                                skiprows=2,
                                delimiter=",",
                                usecols=(0, 1, 2, 3, 4, 5),
                            )

                        try:
                            kin_vals = np.loadtxt(kin, delimiter=",")
                        except:
                            kin_vals = np.loadtxt(
                                kin, delimiter=",", usecols=(0, 1, 2, 3, 4, 5)
                            )

                        number_list = np.array(
                            [
                                np.round(fem_vals, 3),
                                np.round(tib_vals, 3),
                                np.round(kin_vals, 3),
                            ]
                        )

                        N = fem_vals[:, 0].size

                        for i in range(0, N):
                            list = [
                                study,
                                pat_id,
                                sess_id,
                                mvt_id,
                                side,
                                i,
                                number_list[0, i, :],
                                number_list[1, i, :],
                                number_list[2, i, :],
                            ]
                            temp_frame = pd.DataFrame([list], columns=init_cols)
                            self.df = pd.concat(
                                [self.df, temp_frame], axis=0, ignore_index=True
                            )

    def grab_index_from_column(
        self,
        column_name,
        index,
        condition=None,
        condition_column=None,
        condition_threshold=None,
    ):
        result = []
        for i in range(0, self.df.shape[0]):
            if condition is not None:
                if condition == "greater":
                    if self.df[condition_column][i] > condition_threshold:
                        result.append(self.df[column_name][i][index])
                elif condition == "less":
                    if self.df[condition_column][i] < condition_threshold:
                        result.append(self.df[column_name][i][index])
                elif condition == "equal":
                    if self.df[condition_column][i] == condition_threshold:
                        result.append(self.df[column_name][i][index])
            else:
                result.append(self.df[column_name][i][index])

        return result

    def grab_data_from_column(
        self,
        column_name,
        condition=None,
        condition_column=None,
        condition_threshold=None,
    ):
        result = []
        for i in range(0, self.df.shape[0]):
            if condition is not None:
                if condition == "greater":
                    if self.df[condition_column][i] > condition_threshold:
                        result.append(self.df[column_name][i])
                elif condition == "less":
                    if self.df[condition_column][i] < condition_threshold:
                        result.append(self.df[column_name][i])
                elif condition == "equal":
                    if self.df[condition_column][i] == condition_threshold:
                        result.append(self.df[column_name][i])

            else:
                result.append(self.df[column_name][i])

        return result
