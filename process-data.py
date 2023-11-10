import kinproc.jtsFunctions as jts
import kinproc.rot
import kinproc.process
import kinproc.utility as util
from datetime import date
from pathlib import Path
import os
import glob
import shutil
import numpy as np


def main():
    # First thing we want to do is create a logger
    # using the same logger that we got from the JTML_seg system (works well)
    today_date = date.today()
    today_as_str = today_date.strftime("%y.%m.%d")
    logger_name = "./" + today_as_str + "_logger.log"
    log = util.set_logger(logger_name)

    # Now we want to start looping through the data that we are working with. Need to find the data first

    DATA_HOME_DIR = (
        "/media/ajensen123@ad.ufl.edu/Andrew's External SSD/Data/Datasets_FemCleaned"
    )
    study_list = ["Actiyas", "GMK", "Lima", "Toshi", "Duluth", "Ghent", "Florent", "Arizona"]
    types = ["gt"]

    # Use the created function in utility to list each of the movements in each of the directories
    mvt_dirs = util.get_mvt_dirs(DATA_HOME_DIR, study_list)

    for dir in mvt_dirs:
        # only want to grab the studies where we ran test data through joint track auto
        if True:
            dir_no_C_drive = dir[
                len(DATA_HOME_DIR) :
            ]  # remove "C:/Datasets_FemCleaned/" from each of the directories for easier naming
            # list the four sets of kinematics
            gt_fem = dir + "/fem.jts"
            gt_tib = dir + "/tib.jts"

            if not os.path.exists(gt_fem):
                gt_fem = dir + "/fem.jtak"
            if not os.path.exists(gt_tib):
                gt_tib = dir + "/tib.jtak"

            # grabbing the name of each of the stls for the studies

            fem_stl_name = glob.glob1(dir, "*fem*.stl")[0]
            tib_stl_name = glob.glob1(dir, "*tib*.stl")[0]

            fem_stl_path = dir + "/" + fem_stl_name
            tib_stl_path = dir + "/" + tib_stl_name

            # determining which side the study is on based on the name of the stl model provided (might need to do a better job of this, but it seems to work for now)

            if "left" in fem_stl_name or "L" in fem_stl_name or "Left" in dir:
                side = "left"
            elif "right" in fem_stl_name or "R" in fem_stl_name or "Right" in dir:
                side = "right"

            cwd = os.getcwd()
            data_dir = "/supervised_kinematics_new/" + dir_no_C_drive + "/"

            real_data_dir = cwd + "/" + data_dir

            if not os.path.exists(real_data_dir):
                os.makedirs(real_data_dir)

            type = "gt"
            # run through ground truth data
            if type in types:
                try:
                    kinproc.process.kneeData(
                        stlFem=fem_stl_path,
                        stlTib=tib_stl_path,
                        jtsFem=gt_fem,
                        jtsTib=gt_tib,
                        side=side,
                        videoFileName="video_gt",
                        dataFileName="data_gt.mat",
                        homeDir=real_data_dir,
                        videoBool=False,
                        type=type,
                    )
                except Exception as e:
                    print("Could not create kinematics for: ", dir)
                    print("Error: ", e)

            # Want to copy each of the main kinematics files into the sub-folder that contains the information for each of them - keep everything in one place
            try:
                shutil.copy(src=gt_fem, dst=real_data_dir)
            except:
                pass
            try:
                shutil.copy(src=gt_tib, dst=real_data_dir)

            except:
                pass

            # this lets us know the operative leg
            Path(real_data_dir + "/" + side + ".txt").touch()


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
