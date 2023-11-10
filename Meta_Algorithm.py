from utils import *
import numpy as np
import scipy as sp
import pandas as pd
import kinproc as kp
import matplotlib.pyplot as plt
import random

# derived from the paper Spliner_Flipper Class in ista analysis directory.
class Symmetry_Trap_Solver:
    def __init__(self, input_dataframe: pd.DataFrame, study: str) -> None:
        self.input_dataframe = input_dataframe
        self.study = study

        self.partition_dataframe()

        self.total_frames = len(self.dataframe)
        self.unique_pat_sess_mvt = []
        self.create_dict_of_unique_image_sequences()

        self.sub_dataframes = dict()

        self.populate_sub_dataframes()

    def partition_dataframe(self):
        # remove everything but the parts of the dataframe from the specific study listed
        self.dataframe = self.input_dataframe[
            self.input_dataframe["Study"] == self.study
        ]

        self.dataframe = self.dataframe.reset_index(drop=True)

    def create_dict_of_unique_image_sequences(self):
        # list the unique subjects in the dataframe
        # want to be able to easily parse through the df so that we can find specific seuences of images
        self.patients = self.dataframe["Patient"].unique()

        for patient in self.patients:
            sessions = self.dataframe[self.dataframe["Patient"] == patient][
                "Session"
            ].unique()

            for session in sessions:
                mvts = self.dataframe[
                    (self.dataframe["Patient"] == patient)
                    & (self.dataframe["Session"] == session)
                ]["Movement"].unique()

                for mvt in mvts:
                    self.unique_pat_sess_mvt.append([patient, session, mvt])

    def pull_specific_image_sequence(self, patient: str, session: str, mvt: str):
        # pull the specific image sequence from the dataframe
        # return the dataframe
        return self.dataframe[
            (self.dataframe["Patient"] == patient)
            & (self.dataframe["Session"] == session)
            & (self.dataframe["Movement"] == mvt)
        ].reset_index(drop=True)

    def populate_sub_dataframes(self):
        # populate the sub_dataframes with the specific image sequences
        for pat_sess_mvt in self.unique_pat_sess_mvt:
            dict_key = pat_sess_mvt[0] + "_" + pat_sess_mvt[1] + "_" + pat_sess_mvt[2]

            self.sub_dataframes[dict_key] = Image_Sequence_Dataframe(
                self.pull_specific_image_sequence(
                    pat_sess_mvt[0], pat_sess_mvt[1], pat_sess_mvt[2]
                )
            )


class Image_Sequence_Dataframe:
    def __init__(self, image_sequence: pd.DataFrame) -> None:
        self.df = image_sequence
        self.name_of_sequence = (
            self.df["Patient"][0]
            + "_"
            + self.df["Session"][0]
            + "_"
            + self.df["Movement"][0]
        )

        self.input_kinematics = np.zeros((self.df.shape[0], 6))
        self.symmetric_kinematics = np.zeros((self.df.shape[0], 6))
        self.ground_truth_kinematics = np.zeros((self.df.shape[0], 6))
        self.symmetric_distance = np.zeros((self.df.shape[0], 1))
        self.symmetry_trap_bool = np.zeros((self.df.shape[0], 1))
        self.ml_corrected_kinematics = np.zeros((self.df.shape[0], 6))
        self.ambiguous_pose_bool = np.zeros((self.df.shape[0], 1))
        self.gt_ambiguous_pose_bool = np.zeros((self.df.shape[0], 1))
        self.ground_truth_symmetric_kinematics = np.zeros((self.df.shape[0], 6))
        self.symmetry_trap_bool_validation = np.zeros((self.df.shape[0], 1))

        self.solid_angle_input_to_gt = np.zeros((self.df.shape[0], 1))
        self.solid_angle_sym_to_gt = np.zeros((self.df.shape[0], 1))

    def create_spline(self):
        if self.USE_SPLINE:
            self.flx_spline_coords = self.ml_processed_kinematics[
                np.where(self.not_ambig_bool == 1)[0], 3
            ].reshape(-1)
            self.vv_spline_coords = self.ml_processed_kinematics[
                np.where(self.not_ambig_bool == 1)[0], 4
            ].reshape(-1)
            self.ie_spline_coords = self.ml_processed_kinematics[
                np.where(self.not_ambig_bool == 1)[0], 5
            ].reshape(-1)
            self.spline_t = np.where(self.not_ambig_bool == 1)[0]

            self.flx_spline = sp.interpolate.CubicSpline(
                self.spline_t, self.flx_spline_coords
            )
            self.vv_spline = sp.interpolate.CubicSpline(
                self.spline_t, self.vv_spline_coords
            )
            self.ie_spline = sp.interpolate.CubicSpline(
                self.spline_t, self.ie_spline_coords
            )

            self.ambig_flx_spline = self.flx_spline(
                np.where(self.ambiguous_pose_bool == 1)[0]
            )
            self.ambig_vv_spline = self.vv_spline(
                np.where(self.ambiguous_pose_bool == 1)[0]
            )
            self.ambig_ie_spline = self.ie_spline(
                np.where(self.ambiguous_pose_bool == 1)[0]
            )
        else:
            pass

    def plot_spline(self):
        if self.USE_SPLINE:
            plt.figure(figsize=(11, 7))

            xs = np.linspace(0, len(self.df), 1000)
            spline_vals = sp.interpolate.splev(xs, self.spline)
            plt.plot(xs, spline_vals, label="Spline")
            plt.legend(loc="best", ncol=2)
            plt.title(self.name_of_sequence + " Spline")
            plt.show()

    def apply_spline(self):
        if self.USE_SPLINE:
            for ambig_pose_loc in np.where(self.ambiguous_pose_bool == 1)[0]:
                input_pose = self.input_kinematics[ambig_pose_loc, :]
                sym_pose = self.symmetric_kinematics[ambig_pose_loc, :]
                spline_pose = np.array(
                    [
                        0,
                        0,
                        0,
                        self.flx_spline(ambig_pose_loc),
                        self.vv_spline(ambig_pose_loc),
                        self.ie_spline(ambig_pose_loc),
                    ]
                )
                
                input_to_spline_dist = kp.jtsFunctions.solid_angle_difference_between_poses(
                    input_pose, spline_pose
                )
                sym_to_spline_dist = kp.jtsFunctions.solid_angle_difference_between_poses(
                    sym_pose, spline_pose
                )
                if input_to_spline_dist < sym_to_spline_dist:
                    self.ml_processed_kinematics[ambig_pose_loc, :] = input_pose
                    self.outputs[ambig_pose_loc] = 0
                else:
                    self.ml_processed_kinematics[ambig_pose_loc, :] = sym_pose
                    self.outputs[ambig_pose_loc] = 1
        else:
            pass
                
    def blind_flip(self):
        # want to use this method when it is not possible to create a spline due to lack of data
        # we can try this to only include frames that are inside the ambiguous zone
        # have an if/then to control whether it is outside of the AZ, not only looping through that. We need to include all the data for accurate measures of sensitivity and specificity

        self.switch = []

        for i in range(0, self.df.shape[0]):
            if self.ambig_pose_bool[i]:
                # this should be images inside the ambiguos zone

                tib_orig_pose = self.df["tib_jta"][i]
                tib_dual_pose = np.array(sym_trap_dual(tib_orig_pose))

                fem_pose = self.df["fem_jta"][i]

                orig_pose_matrix = np.linalg.inv(
                    kp.process.pose_to_matrix(fem_pose)
                ) @ kp.process.pose_to_matrix(tib_orig_pose)
                dual_pose_matrix = np.linalg.inv(
                    kp.process.pose_to_matrix(fem_pose)
                ) @ kp.process.pose_to_matrix(tib_dual_pose)

                orig_vv = kp.jtsFunctions.getRotations("312", orig_pose_matrix)[1]
                dual_vv = kp.jtsFunctions.getRotations("312", dual_pose_matrix)[1]

                if abs(orig_vv - dual_vv) < 0.2:
                    self.switch.append("No")
                elif orig_vv > dual_vv:
                    self.switch.append("Yes")
                elif dual_vv > orig_vv:
                    self.switch.append("No")

            else:
                self.switch.append("No")


    def confusion_matrix(self):
        targets = self.targets
        outputs = self.outputs
        tp = np.sum((targets == 1) & (outputs == 1))
        tn = np.sum((targets == 0) & (outputs == 0))
        fp = np.sum((targets == 0) & (outputs == 1))
        fn = np.sum((targets == 1) & (outputs == 0))
        self.cm = np.array([[tp, fp], [fn, tn]])
        print(self.cm)
        return tp, tn, fp, fn
    def calculate_symmetric_pose(self, side_dict, plot=False):
        side = side_dict[self.name_of_sequence]

        # With testing out of the way, we are going to populate a list of kinematics that can be parsed
        for i in range(0, len(self.df)):
            self.input_kinematics[i, :] = kp.process.relative_kinematics(
                fem_pose=self.df["fem_jta"][i],
                tib_pose=self.df["tib_jta"][i],
                side=side,
            )
            self.symmetric_kinematics[i, :] = kp.process.relative_kinematics(
                fem_pose=self.df["fem_jta"][i],
                tib_pose=np.array(sym_trap_dual(self.df["tib_jta"][i])),
                side=side,
            )
            self.ground_truth_kinematics[i, :] = kp.process.relative_kinematics(
                fem_pose=self.df["fem_gt"][i], tib_pose=self.df["tib_gt"][i], side=side
            )
            self.ground_truth_symmetric_kinematics[
                i, :
            ] = kp.process.relative_kinematics(
                fem_pose=self.df["fem_gt"][i],
                tib_pose=np.array(sym_trap_dual(self.df["tib_gt"][i])),
                side=side,
            )

            self.symmetric_distance[i] = solid_angle_distance_to_dual(
                self.df["tib_jta"][i]
            )
            # self.symmetry_trap_bool[i] = self.df["tib_sym_trap"][i]

            self.solid_angle_input_to_gt[
                i
            ] = kp.jtsFunctions.solid_angle_difference_between_poses(
                self.df["tib_jta"][i], self.df["tib_gt"][i]
            )
            self.solid_angle_sym_to_gt[
                i
            ] = kp.jtsFunctions.solid_angle_difference_between_poses(
                np.array(sym_trap_dual(self.df["tib_jta"][i])), self.df["tib_gt"][i]
            )
        self.symmetry_trap_bool_validation[
            np.where(self.solid_angle_input_to_gt > self.solid_angle_sym_to_gt)
        ] = True
        self.targets = self.symmetry_trap_bool_validation

        self.symmetry_trap_bool = self.symmetry_trap_bool_validation

        if plot:
            fig, ax = plt.subplots(2, 2, figsize=(15, 10))
            ax[0, 0].plot(self.input_kinematics[:, 3], label="Input")
            ax[0, 0].plot(self.symmetric_kinematics[:, 3], label="Symmetric")
            ax[0, 0].plot(self.ground_truth_kinematics[:, 3], label="Ground Truth")
            ax[0, 0].plot(self.ground_truth_symmetric_kinematics[:, 3], label="GT Sym")
            # plot locations with symmetry trap
            ax[0, 0].scatter(
                np.where(self.symmetry_trap_bool == True)[0],
                self.input_kinematics[:, 3][
                    np.where(self.symmetry_trap_bool == True)[0]
                ],
                color="red",
                label="Symmetry Trap",
            )
            # second sym trap measurements
            ax[0, 0].scatter(
                np.where(self.symmetry_trap_bool_validation == True)[0],
                self.symmetric_kinematics[:, 3][
                    np.where(self.symmetry_trap_bool_validation == True)[0]
                ],
                color="green",
                label="Symmetry Trap Validation",
            )
            ax[0, 0].set_title("Flexion/Extension")
            ax[0, 0].legend()

            ax[0, 1].plot(self.input_kinematics[:, 4], label="Input")
            ax[0, 1].plot(self.symmetric_kinematics[:, 4], label="Symmetric")
            ax[0, 1].plot(self.ground_truth_kinematics[:, 4], label="Ground Truth")
            ax[0, 1].plot(self.ground_truth_symmetric_kinematics[:, 4], label="GT Sym")
            # second sym trap measurements
            ax[0, 1].scatter(
                np.where(self.symmetry_trap_bool_validation == True)[0],
                self.symmetric_kinematics[:, 4][
                    np.where(self.symmetry_trap_bool_validation == True)[0]
                ],
                color="green",
                label="Symmetry Trap Validation",
            )
            # plot locations with symmetry trap
            ax[0, 1].scatter(
                np.where(self.symmetry_trap_bool == True)[0],
                self.input_kinematics[:, 4][
                    np.where(self.symmetry_trap_bool == True)[0]
                ],
                color="red",
                label="Symmetry Trap",
            )
            ax[0, 1].set_title("Abduction/Adduction")
            ax[0, 1].legend()

            ax[1, 0].plot(self.input_kinematics[:, 5], label="Input")
            ax[1, 0].plot(self.symmetric_kinematics[:, 5], label="Symmetric")
            ax[1, 0].plot(self.ground_truth_kinematics[:, 5], label="Ground Truth")
            ax[1, 0].plot(self.ground_truth_symmetric_kinematics[:, 5], label="GT Sym")
            # plot locations with symmetry trap
            ax[1, 0].scatter(
                np.where(self.symmetry_trap_bool == True)[0],
                self.input_kinematics[:, 5][
                    np.where(self.symmetry_trap_bool == True)[0]
                ],
                color="red",
                label="Symmetry Trap",
            )
            # second sym trap measurements
            ax[1, 0].scatter(
                np.where(self.symmetry_trap_bool_validation == True)[0],
                self.symmetric_kinematics[:, 5][
                    np.where(self.symmetry_trap_bool_validation == True)[0]
                ],
                color="green",
                label="Symmetry Trap Validation",
            )
            ax[1, 0].set_title("Internal/External Rotation")
            ax[1, 0].legend()

            ax[1, 1].plot(self.symmetric_distance)
            # plot regions with symmetry trap
            ax[1, 1].scatter(
                np.where(self.symmetry_trap_bool == True)[0],
                self.symmetric_distance[np.where(self.symmetry_trap_bool == True)[0]],
                color="red",
                label="Symmetry Trap",
            )
            ax[1, 1].set_title("Solid Angle Distance to Symmetric Pose")

            plt.show()

    def is_pose_in_symmetry_trap(self, input_pose, symmetric_pose, gt_pose):
        # calculate the distance between the input pose and the gt pose
        dist_to_gt = np.linalg.norm(input_pose - gt_pose)
        # calculate the distance between the symmetric pose and the gt pose
        dist_to_sym = np.linalg.norm(symmetric_pose - gt_pose)

        # if the distance to the symmetric pose is less than the distance to the gt pose, then the input pose is in the symmetry trap
        if dist_to_sym < dist_to_gt:
            return True
        else:
            return False

    def correct_kinematics_with_machine_learning(
        self, sklearn_model, test_data=None, test_labels=None
    ):
        X_in = np.array(
            [
                self.input_kinematics[:, 4], # X rotation
                self.input_kinematics[:, 5], # y rotation 
                self.input_kinematics[:, 3], # z rotation
                self.symmetric_distance[:, 0],
            ]
        ).T
        X_sym = np.array(
            [
                self.symmetric_kinematics[:, 4],
                self.symmetric_kinematics[:, 5],
                self.symmetric_kinematics[:, 3],
                self.symmetric_distance[:, 0],
            ]
        ).T
        X_gt = np.array(
            [
                self.ground_truth_kinematics[:, 4],
                self.ground_truth_kinematics[:, 5],
                self.ground_truth_kinematics[:, 3],
                self.symmetric_distance[:, 0],
            ]
        ).T

        X_gt_sym = np.array(
            [
                self.ground_truth_symmetric_kinematics[:, 4],
                self.ground_truth_symmetric_kinematics[:, 5],
                self.ground_truth_symmetric_kinematics[:, 3],
                self.symmetric_distance[:, 0],
            ]
        ).T

        try:
            input_preds = sklearn_model.predict(X_in)
            symmetric_preds = sklearn_model.predict(X_sym)
            gt_preds = sklearn_model.predict(X_gt)
            gt_sym_preds = sklearn_model.predict(X_gt_sym)
        except:
            print("Forward Pass didnt work")
            return

        self.keep_input_kinematics = np.zeros([len(self.input_kinematics), 1])
        self.flip_input_kinematics = np.zeros([len(self.input_kinematics), 1])
        self.keep_gt_kinematics = np.zeros([len(self.input_kinematics), 1])
        self.flip_gt_kinematics = np.zeros([len(self.input_kinematics), 1])

        self.ml_processed_kinematics = np.zeros([len(self.input_kinematics), 6])
        self.ml_gt_processed_kinematics = np.zeros([len(self.input_kinematics), 6])
        
        
        # running on input kinematics
        self.outputs = np.zeros([len(self.input_kinematics), 1])
        # for outputs, symmetry trap is "pathology" and gets flagged as 1
        for i in range(len(self.input_kinematics)):
            if (input_preds[i] == 1) & (symmetric_preds[i] == 0):
                self.keep_input_kinematics[i] = 1
                self.outputs[i] = 0
                self.ml_processed_kinematics[i] = self.input_kinematics[i]
            elif (input_preds[i] == 0) & (symmetric_preds[i] == 1):
                self.flip_input_kinematics[i] = 1
                self.outputs[i] = 1
                self.ml_processed_kinematics[i] = self.symmetric_kinematics[i]
            elif input_preds[i] == symmetric_preds[i]:
                self.ambiguous_pose_bool[i] = 1

        self.not_ambig_bool = self.keep_input_kinematics + self.flip_input_kinematics

        if len(self.input_kinematics) - np.sum(self.ambiguous_pose_bool) < 5:
            self.USE_SPLINE = False
        else:
            self.USE_SPLINE = True
            
        # running on ground truth kinematics
        for i in range(len(self.input_kinematics)):
            if (gt_preds[i] == 1) & (gt_sym_preds[i] == 0):
                self.keep_gt_kinematics[i] = 1
                self.ml_gt_processed_kinematics[i] = self.ground_truth_kinematics[i]
            elif (gt_preds[i] == 0) & (gt_sym_preds[i] == 1):
                self.flip_gt_kinematics[i] = 1
                self.ml_gt_processed_kinematics[i] = self.ground_truth_symmetric_kinematics[i]
            elif gt_preds[i] == gt_sym_preds[i]:
                self.ambiguous_pose_bool[i] = 1
    
    def create_kinematics_with_symmetry_traps(self, plot = False):
        UF_BLUE = "#0021A5"
        UF_ORANGE = "#FA4616"
        # Here, we want to take some ground truth kinematics and spruce in some symmetry traps
        # We want to do this for plotting purposes
        self.gt_with_symmetry_traps = np.zeros((self.df.shape[0], 6))
        # pick some random frames to plug in symmetry traps
        self.frames_to_sym = random.sample(range(0,self.gt_with_symmetry_traps.shape[0]), 3)
        for i in range(0,len(self.gt_with_symmetry_traps)):
            if i in self.frames_to_sym:
                self.gt_with_symmetry_traps[i,:] = self.symmetric_kinematics[i,:]
            else:
                self.gt_with_symmetry_traps[i,:] = self.ground_truth_kinematics[i,:]
        if plot:
            ratio = 0.5
            font_size = 22*ratio
            fig, ax = plt.subplots(1, 3, figsize=(30*ratio, 10*ratio))
            ax[0].plot(self.ground_truth_kinematics[:,3], "-o", label = "Corrected Kinematics", color = UF_ORANGE)
            ax[0].plot(self.gt_with_symmetry_traps[:,3], "-o", label = "Input Kinematics", color = UF_BLUE)
            ax[0].scatter(self.frames_to_sym, self.gt_with_symmetry_traps[:,3][self.frames_to_sym], color="red", label="Symmetry Trap",zorder=10, marker="x")
            ax[0].set_title("Flexion/Extension", fontsize=font_size)
            ax[1].plot(self.ground_truth_kinematics[:,4], "-o", label = "Corrected Kinematics", color = UF_ORANGE)
            ax[1].plot(self.gt_with_symmetry_traps[:,4], "-o", label = "Input Kinematics", color = UF_BLUE)
            ax[1].scatter(self.frames_to_sym, self.gt_with_symmetry_traps[:,4][self.frames_to_sym], color="red", label="Symmetry Trap", zorder=10,marker="x")
            ax[1].set_title("Abduction/Adduction",fontsize=font_size)
            ax[2].plot(self.ground_truth_kinematics[:,5], "-o", label = "Corrected Kinematics", color = UF_ORANGE)
            ax[2].plot(self.gt_with_symmetry_traps[:,5], "-o", label = "Input Kinematics", color = UF_BLUE)
            ax[2].scatter(self.frames_to_sym, self.gt_with_symmetry_traps[:,5][self.frames_to_sym], color="red", label = "Symmetry Trap",zorder=10, marker="x")
            ax[2].set_title("Internal/External Rotation",fontsize=font_size)
            ax[0].legend(fontsize=font_size-2,loc = "best")
            ax[1].legend(fontsize=font_size-2, loc = "best")
            ax[2].legend(fontsize=font_size-2, loc = "best")
            ax[0].set_xlabel("Frame Number",fontsize = font_size)
            ax[1].set_xlabel("Frame Number",fontsize = font_size)
            ax[2].set_xlabel("Frame Number", fontsize=font_size)
            ax[0].set_ylabel("Angle (Degrees)",fontsize = font_size)
            ax[1].set_ylabel("Angle (Degrees)",fontsize = font_size)
            ax[2].set_ylabel("Angle (Degrees)",fontsize = font_size)
            for axs in ax:
                axs.tick_params(axis="both", labelsize=font_size)
                # set x axis ticks to increase by 2 
                axs.set_xticks(np.arange(0, len(self.gt_with_symmetry_traps[:,5]), 2))
                # set y label ticks to have degree symbol
                axs.set_yticklabels([str(int(x)) + u"\u00b0" for x in axs.get_yticks()])
            
    def plotting_all_kinematics(self):
        spline_x_coords = np.arange(0, len(self.df), 0.5)
        fig, ax = plt.subplots(2, 2, figsize=(25, 15))
        ax[0, 0].plot(self.input_kinematics[:, 3], label="Input")
        ax[0, 0].plot(self.symmetric_kinematics[:, 3], label="Symmetric")
        ax[0, 0].plot(self.ground_truth_kinematics[:, 3], label="Ground Truth")
        ax[0, 0].plot(self.ground_truth_symmetric_kinematics[:, 3], label="GT Sym")
        # plot locations with symmetry trap
        ax[0, 0].scatter(
            np.where(self.symmetry_trap_bool == True)[0],
            self.input_kinematics[:, 3][
                np.where(self.symmetry_trap_bool == True)[0]
            ],
            color="red",
            label="Symmetry Trap",
        )
        if self.USE_SPLINE:
            ax[0,0].plot(self.spline_t, self.flx_spline_coords,color="purple", label="Spline")
        # second sym trap measurements
        ax[0, 0].scatter(
            np.where(self.symmetry_trap_bool_validation == True)[0],
            self.symmetric_kinematics[:, 3][
                np.where(self.symmetry_trap_bool_validation == True)[0]
            ],
            color="green",
            label="Symmetry Trap Validation",
        )
        ax[0, 0].set_title("Flexion/Extension")
        ax[0, 0].legend(loc="best")

        ax[0, 1].plot(self.input_kinematics[:, 4], label="Input")
        ax[0, 1].plot(self.symmetric_kinematics[:, 4], label="Symmetric")
        ax[0, 1].plot(self.ground_truth_kinematics[:, 4], label="Ground Truth")
        ax[0, 1].plot(self.ground_truth_symmetric_kinematics[:, 4], label="GT Sym")
        if self.USE_SPLINE:
            ax[0,1].plot(self.spline_t, self.vv_spline_coords,color="purple", label="Spline")
        # second sym trap measurements
        ax[0, 1].scatter(
            np.where(self.symmetry_trap_bool_validation == True)[0],
            self.symmetric_kinematics[:, 4][
                np.where(self.symmetry_trap_bool_validation == True)[0]
            ],
            color="green",
            label="Symmetry Trap Validation",
        )
        # plot locations with symmetry trap
        ax[0, 1].scatter(
            np.where(self.symmetry_trap_bool == True)[0],
            self.input_kinematics[:, 4][
                np.where(self.symmetry_trap_bool == True)[0]
            ],
            color="red",
            label="Symmetry Trap",
        )
        ax[0, 1].set_title("Abduction/Adduction")
        ax[0, 1].legend(loc="best")

        ax[1, 0].plot(self.input_kinematics[:, 5], label="Input")
        ax[1, 0].plot(self.symmetric_kinematics[:, 5], label="Symmetric")
        ax[1, 0].plot(self.ground_truth_kinematics[:, 5], label="Ground Truth")
        ax[1, 0].plot(self.ground_truth_symmetric_kinematics[:, 5], label="GT Sym")
        if self.USE_SPLINE:
            ax[1,0].plot(self.spline_t, self.ie_spline_coords,color="purple", label="Spline")
        # plot locations with symmetry trap
        ax[1, 0].scatter(
            np.where(self.symmetry_trap_bool == True)[0],
            self.input_kinematics[:, 5][
                np.where(self.symmetry_trap_bool == True)[0]
            ],
            color="red",
            label="Symmetry Trap",
        )
        # second sym trap measurements
        ax[1, 0].scatter(
            np.where(self.symmetry_trap_bool_validation == True)[0],
            self.symmetric_kinematics[:, 5][
                np.where(self.symmetry_trap_bool_validation == True)[0]
            ],
            color="green",
            label="Symmetry Trap Validation",
        )
        ax[1, 0].set_title("Internal/External Rotation")
        ax[1, 0].legend(loc="best")

        ax[1, 1].plot(self.symmetric_distance)
        # plot regions with symmetry trap
        ax[1, 1].scatter(
            np.where(self.symmetry_trap_bool == True)[0],
            self.symmetric_distance[np.where(self.symmetry_trap_bool == True)[0]],
            color="red",
            label="Symmetry Trap",
        )
        ax[1, 1].set_title("Solid Angle Distance to Symmetric Pose")

        plt.show()
    def determine_symmetry_distance_for_each_frame(self):
        correct_locs = np.where(self.targets == self.outputs)[0]
        incorrect_locs = np.where(self.targets != self.outputs)[0]
        sym_dist_correct = self.symmetric_distance[correct_locs]
        sym_dist_incorrect = self.symmetric_distance[incorrect_locs]
        
        if not len(sym_dist_correct) == 0:
            print("Mean Symmetric Distance for Correctly Classified Frames: ", np.mean(sym_dist_correct))
        if not len(sym_dist_incorrect) == 0:
            print("Mean Symmetric Distance for Incorrectly Classified Frames: ", np.mean(sym_dist_incorrect))
        
        return sym_dist_correct, sym_dist_incorrect
        
