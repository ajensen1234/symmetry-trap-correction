from ista_utils import *
import numpy as np
import scipy as sp
import pandas as pd
import kinproc as kp
import matplotlib.pyplot as plt

class Spline_Kinematics_Flipper():
    def __init__(
        self,
        input_dataframe: pd.DataFrame,
        study: str) -> None:
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
        self.dataframe = self.input_dataframe[self.input_dataframe['Study'] == self.study]
        
        self.dataframe = self.dataframe.reset_index(drop=True)
        
    def create_dict_of_unique_image_sequences(self):
        # list the unique subjects in the dataframe
        # want to be able to easily parse through the df so that we can find specific seuences of images 
        self.patients = self.dataframe['Patient'].unique()
        
        for patient in self.patients:
            sessions = self.dataframe[self.dataframe['Patient'] == patient]['Session'].unique()

            
            for session in sessions:
                mvts = self.dataframe[(self.dataframe['Patient'] == patient) & (self.dataframe['Session'] == session)]['Movement'].unique()
            
                
                for mvt in mvts:
                    self.unique_pat_sess_mvt.append([patient, session, mvt])
    
    def pull_specific_image_sequence(self, patient: str, session: str, mvt: str):
        # pull the specific image sequence from the dataframe
        # return the dataframe
        return self.dataframe[(self.dataframe['Patient'] == patient) & (self.dataframe['Session'] == session) & (self.dataframe['Movement'] == mvt)].reset_index(drop=True)
    
    def populate_sub_dataframes(self):
        # populate the sub_dataframes with the specific image sequences
        for pat_sess_mvt in self.unique_pat_sess_mvt:
            dict_key = pat_sess_mvt[0] + "_" + pat_sess_mvt[1] + "_" + pat_sess_mvt[2]
            
            self.sub_dataframes[dict_key] = Image_Sequence_Dataframe(self.pull_specific_image_sequence(pat_sess_mvt[0], pat_sess_mvt[1], pat_sess_mvt[2]))
    

class Image_Sequence_Dataframe():
    def __init__ (self, image_sequence: pd.DataFrame) -> None:
        self.df = image_sequence
        self.name_of_sequence = self.df["Patient"][0] + "_" + self.df["Session"][0] + "_" + self.df["Movement"][0]
        
        self.list_images_outside_ambiguous_zone()
        self.list_sym_trap_poses_in_amibiguous_zone()
        self.create_spline()
        self.apply_spline()
        
        self.calculate_all_metrics()
        
        
        
        
        
    def list_images_outside_ambiguous_zone(self):
        az_loc = 3.5
        
        apparent_vv = []
        for i in range(0,self.df.shape[0]):
            tib_pose = self.df["tib_jta"][i]
            apparent_vv.append(ambiguous_pose(tib_pose))
        
        apparent_vv = np.array(apparent_vv)
        
        self.ambig_pose_bool = [x for x in (apparent_vv > az_loc)]
        
        self.outside_az_df = self.df[self.ambig_pose_bool].reset_index(drop=True)
        self.inside_az_df = self.df[~np.array(self.ambig_pose_bool)].reset_index(drop=True)
        
        if self.outside_az_df.shape[0] < 5:
            self.USE_SPLINE = False
        else:
            self.USE_SPLINE = True
    
    def list_sym_trap_poses_in_amibiguous_zone(self):
        if self.USE_SPLINE:
            self.in_az_sym_trap = self.inside_az_df[self.inside_az_df["tib_sym_trap"] == True].reset_index(drop=True)
            self.in_az_no_sym_trap = self.inside_az_df[self.inside_az_df["tib_sym_trap"] == False].reset_index(drop=True)
            self.out_az_no_sym_trap = self.outside_az_df[self.outside_az_df["tib_sym_trap"] == False].reset_index(drop=True)
            self.out_az_sym_trap = self.outside_az_df[self.outside_az_df["tib_sym_trap"] == True].reset_index(drop=True)
            
    def create_spline(self):
        if self.USE_SPLINE:
            self.spline_coordinates = np.zeros((self.outside_az_df.shape[0],2))
            self.in_az_coordinates = np.zeros((self.inside_az_df.shape[0],2))
            self.in_az_sym_trap_coordinates = np.zeros((self.in_az_sym_trap.shape[0],2))
            self.in_az_no_sym_trap_coordinates = np.zeros((self.in_az_no_sym_trap.shape[0],2))
            self.out_az_no_sym_trap_coordinates = np.zeros((self.out_az_no_sym_trap.shape[0],2))
            self.out_az_sym_trap_coordinates = np.zeros((self.out_az_sym_trap.shape[0],2))
            
            for i in range(self.outside_az_df.shape[0]):
                self.spline_coordinates[i,0] = self.outside_az_df["Image Index"][i]
                self.spline_coordinates[i,1] = self.outside_az_df["tib_jta"][i][4]
            
            self.spline = sp.interpolate.splrep(self.spline_coordinates[:,0], self.spline_coordinates[:,1], s=2)
            
            for i in range(0,self.inside_az_df.shape[0]):
                self.in_az_coordinates[i,0] = self.inside_az_df["Image Index"][i]
                self.in_az_coordinates[i,1] = self.inside_az_df["tib_jta"][i][4]
            for i in range(0, self.in_az_sym_trap.shape[0]):
                self.in_az_sym_trap_coordinates[i,0] = self.in_az_sym_trap["Image Index"][i]
                self.in_az_sym_trap_coordinates[i,1] = self.in_az_sym_trap["tib_jta"][i][4]
            for i in range(0, self.in_az_no_sym_trap.shape[0]):
                self.in_az_no_sym_trap_coordinates[i,0] = self.in_az_no_sym_trap["Image Index"][i]
                self.in_az_no_sym_trap_coordinates[i,1] = self.in_az_no_sym_trap["tib_jta"][i][4]
            for i in range(0, self.out_az_no_sym_trap.shape[0]):
                self.out_az_no_sym_trap_coordinates[i,0] = self.out_az_no_sym_trap["Image Index"][i]
                self.out_az_no_sym_trap_coordinates[i,1] = self.out_az_no_sym_trap["tib_jta"][i][4]
            for i in range(0, self.out_az_sym_trap.shape[0]):
                self.out_az_sym_trap_coordinates[i,0] = self.out_az_sym_trap["Image Index"][i]
                self.out_az_sym_trap_coordinates[i,1] = self.out_az_sym_trap["tib_jta"][i][4]
            
        else:
            pass
    
    def plot_spline(self):
        if self.USE_SPLINE:
            plt.figure(figsize=(11,7))
            #plt.plot(self.spline_coordinates[:,0], self.spline_coordinates[:,1], 'o', label='outside az data')
            #plt.plot(self.in_az_coordinates[:,0], self.in_az_coordinates[:,1], 'o', label='inside az data')
            plt.plot(self.in_az_sym_trap_coordinates[:,0], self.in_az_sym_trap_coordinates[:,1], 'bx', label='inside az sym trap data')
            plt.plot(self.in_az_no_sym_trap_coordinates[:,0], self.in_az_no_sym_trap_coordinates[:,1], 'bo', label='inside az no sym trap data')
            plt.plot(self.out_az_no_sym_trap_coordinates[:,0], self.out_az_no_sym_trap_coordinates[:,1], 'ro', label='outside az no sym trap data')
            plt.plot(self.out_az_sym_trap_coordinates[:,0], self.out_az_sym_trap_coordinates[:,1], 'rx', label='outside az sym trap data')
            if self.flipped_image_frames_coordinates.shape[0] > 0:
                plt.plot(self.flipped_image_frames_coordinates[:,0], self.flipped_image_frames_coordinates[:,1], 'gx', label='flipped image frames')
            
            xs = np.linspace(0, len(self.df), 1000)
            spline_vals = sp.interpolate.splev(xs, self.spline)
            plt.plot(xs, spline_vals, label="Spline")
            plt.legend(loc='best', ncol=2)
            plt.title(self.name_of_sequence + " Spline")
            plt.show()
            
        
    def apply_spline(self):
        self.switch = []
        self.flipped_image_frames = []
        if self.USE_SPLINE:
            for i in range(0,self.df.shape[0]):
                if not self.ambig_pose_bool[i]:
                    # this should only be poses outside the ambiguous zone
                    spline_at_val = sp.interpolate.splev(self.df["Image Index"][i], self.spline)
                    tib_orig_pose = self.df["tib_jta"][i]
                    tib_dual_pose = np.array(sym_trap_dual(tib_orig_pose))
                    
                    fem_pose = self.df["fem_jta"][i]
                    
                    orig_pose_matrix = np.linalg.inv(kp.jtsFunctions.pose_to_matrix(fem_pose)) @ kp.jtsFunctions.pose_to_matrix(tib_orig_pose)
                    
                    dual_pose_matrix = np.linalg.inv(kp.jtsFunctions.pose_to_matrix(fem_pose)) @ kp.jtsFunctions.pose_to_matrix(tib_dual_pose)
                    
                    orig_vv = kp.jtsFunctions.getRotations("312",orig_pose_matrix)[1]
                    dual_vv = kp.jtsFunctions.getRotations("312",dual_pose_matrix)[1]
                    
                    if np.abs(spline_at_val - orig_vv) < np.abs(spline_at_val - dual_vv):
                        self.switch.append("No")
                    else:
                        self.switch.append("Yes")
                        self.flipped_image_frames.append((self.df["Image Index"][i], dual_vv))
                else:
                    self.switch.append("No")
            
                
            self.flipped_image_frames_coordinates = np.array(self.flipped_image_frames)
                    
        else:
            self.blind_flip()
    
    def blind_flip(self):
        # want to use this method when it is not possible to create a spline due to lack of data
        # we can try this to only include frames that are inside the ambiguous zone
        # have an if/then to control whether it is outside of the AZ, not only looping through that. We need to include all the data for accurate measures of sensitivity and specificity
        
        self.switch = []
        
        for i in range(0,self.df.shape[0]):
            if self.ambig_pose_bool[i]:
                # this should be images inside the ambiguos zone
                
                tib_orig_pose = self.df["tib_jta"][i]
                tib_dual_pose = np.array(sym_trap_dual(tib_orig_pose))
                
                fem_pose = self.df["fem_jta"][i]
                
                orig_pose_matrix = np.linalg.inv(kp.jtsFunctions.pose_to_matrix(fem_pose)) @ kp.jtsFunctions.pose_to_matrix(tib_orig_pose)
                dual_pose_matrix = np.linalg.inv(kp.jtsFunctions.pose_to_matrix(fem_pose)) @ kp.jtsFunctions.pose_to_matrix(tib_dual_pose)
                
                orig_vv = kp.jtsFunctions.getRotations("312",orig_pose_matrix)[1]
                dual_vv = kp.jtsFunctions.getRotations("312",dual_pose_matrix)[1]
                
                if abs(orig_vv - dual_vv) < 0.2:
                    self.switch.append("No")
                elif orig_vv > dual_vv:
                    self.switch.append("Yes")
                elif dual_vv > orig_vv:
                    self.switch.append("No")
                    
                    
            else:
                self.switch.append("No")
        
    def calculate_true_positive(self):
        self.true_positive = []
        for i in range(0, len(self.switch)):
            if (self.switch[i] ==  "Yes") & (self.df["tib_sym_trap"][i] == True):
                self.true_positive.append("Yes")
            else:
                self.true_positive.append("No")
    
    def calculate_false_positive(self):
        self.false_positive = []
        for i in range(0, len(self.switch)):
            if (self.switch[i] ==  "Yes") & (self.df["tib_sym_trap"][i] == False):
                self.false_positive.append("Yes")
            else:
                self.false_positive.append("No")
    
    def calculate_true_negative(self):
        self.true_negative = []
        for i in range(0, len(self.switch)):
            if (self.switch[i] ==  "No") & (self.df["tib_sym_trap"][i] == False):
                self.true_negative.append("Yes")
            else:
                self.true_negative.append("No")
    
    def calculate_false_negative(self):
        self.false_negative = []
        for i in range(0, len(self.switch)):
            if (self.switch[i] ==  "No") & (self.df["tib_sym_trap"][i] == True):
                self.false_negative.append("Yes")
            else:
                self.false_negative.append("No")
                
    def calculate_accuracy(self):
        self.accuracy = (self.true_positive.count("Yes") + self.true_negative.count("Yes")) / len(self.switch)
    
    def calculate_precision(self):
        self.precision = self.true_positive.count("Yes") / (self.true_positive.count("Yes") + self.false_positive.count("Yes"))
        
    def calculate_recall(self):
        self.recall = self.true_positive.count("Yes") / (self.true_positive.count("Yes") + self.false_negative.count("Yes"))
        
    def calculate_f1(self):
        self.f1 = 2 * ((self.precision * self.recall) / (self.precision + self.recall))
    
    def calculate_true_negative_rate(self):
        self.true_negative_rate = self.true_negative.count("Yes") / (self.true_negative.count("Yes") + self.false_positive.count("Yes"))
    
    def calculate_all_metrics(self):
        self.calculate_true_positive()
        self.calculate_false_positive()
        self.calculate_true_negative()
        self.calculate_false_negative()
        try:
            self.calculate_accuracy()
        except:
            pass
        try:
            self.calculate_precision()
        except:
            pass
        try:
            self.calculate_recall()
        except:
            pass
        try:
            self.calculate_f1()
        except:
            pass
        
        try:
            self.calculate_true_negative_rate()
        except:
            pass
    
    def confusion_matrix(self):
        self.confusion_matrix = pd.DataFrame({"True Positive": self.true_positive.count("Yes"), "False Positive": self.false_positive.count("Yes"), "True Negative": self.true_negative.count("Yes"), "False Negative": self.false_negative.count("Yes")}, index = ["Predicted Positive", "Predicted Negative"])
        print(self.confusion_matrix)