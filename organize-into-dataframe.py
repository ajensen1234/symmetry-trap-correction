import numpy as np
import pandas as pd
import kinproc.utility
import os
import glob
import sys


def main():
    # Define the column names for the dataframe
    # We want NFD kinematics, GT kinematics, JTA kinematics, the image name, number, location, relative kinematics, and some extra check boxes at the end
    # we also want to include the differences between each of the measured values
    col = [
        "Study",
        "Patient",
        "Session",
        "Movement",
        "Image Index",
        "fem",
        "tib",
        "kin",
    ]
    df = pd.DataFrame(columns=col)
    pd.set_option(
        "display.max_colwidth", None
    )  # prevents pandas from truncating values

    # Now we want to loop through each of the studies
    study_list = ["Actiyas", "GMK", "Lima", "Toshi"]
    HOME_DIR = "./processed_kinematics"
    for study in study_list:
        study_dir = HOME_DIR + "/" + study

        # Lima folders have a bit of a different structure
        if study != "Lima":
            study_org_dir = study_dir + "/" + study + "_Organized"
        else:
            study_org_dir = study_dir + "/" + study + "_Organized_Updated"

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
                    print(study, pat_id, sess_id, mvt_id)

                    # Now to get the kinematics for each of the kinematics files in this

                    fem = mvt_dir + "/fem.jts"
                    tib = mvt_dir + "/tib.jts"
                    kin_jta = mvt_dir + "/kin_jta.csv"

                    if not os.path.exists(fem_gt):
                        fem_gt = mvt_dir + "/fem.jtak"

                    try:
                        fem_gt_vals = np.loadtxt(fem_gt, skiprows=2, delimiter=",")
                    except:
                        fem_gt_vals = np.loadtxt(
                            fem_gt,
                            skiprows=2,
                            delimiter=",",
                            usecols=(0, 1, 2, 3, 4, 5),
                        )

                    try:
                        tib_gt_vals = np.loadtxt(tib_gt, skiprows=2, delimiter=",")
                    except:
                        tib_gt_vals = np.loadtxt(
                            tib_gt,
                            skiprows=2,
                            delimiter=",",
                            usecols=(0, 1, 2, 3, 4, 5),
                        )

                    try:
                        fem_jta_vals = np.loadtxt(fem_jta, skiprows=2, delimiter=",")
                    except:
                        fem_jta_vals = np.loadtxt(
                            fem_jta,
                            skiprows=2,
                            delimiter=",",
                            usecols=(0, 1, 2, 3, 4, 5),
                        )

                    try:
                        tib_jta_vals = np.loadtxt(tib_jta, skiprows=2, delimiter=",")
                    except:
                        tib_jta_vals = np.loadtxt(
                            tib_jta,
                            skiprows=2,
                            delimiter=",",
                            usecols=(0, 1, 2, 3, 4, 5),
                        )

                    try:
                        kin_gt_vals = np.loadtxt(kin_gt, delimiter=",")
                    except:
                        kin_gt_vals = np.loadtxt(
                            kin_gt, delimiter=",", usecols=(0, 1, 2, 3, 4, 5)
                        )

                    try:
                        kin_jta_vals = np.loadtxt(kin_jta, delimiter=",")
                    except:
                        kin_jta_vals = np.loadtxt(
                            kin_jta, delimiter=",", usecols=(0, 1, 2, 3, 4, 5)
                        )

                    try:
                        kin_fft_vals = np.loadtxt(kin_fft, delimiter=",")
                    except:
                        kin_fft_vals = np.loadtxt(
                            kin_fft, delimiter=",", usecols=(0, 1, 2, 3, 4, 5)
                        )

                    try:
                        kin_fft_perfect_vals = np.loadtxt(
                            kin_fft_perfect, delimiter=","
                        )
                    except:
                        kin_fft_perfect_vals = np.loadtxt(
                            kin_fft_perfect, delimiter=",", usecols=(0, 1, 2, 3, 4, 5)
                        )

                    try:
                        fem_fft_vals = np.loadtxt(fem_fft, skiprows=2, delimiter=",")
                    except:
                        fem_fft_vals = np.loadtxt(
                            fem_fft,
                            skiprows=2,
                            delimiter=",",
                            usecols=(0, 1, 2, 3, 4, 5),
                        )

                    try:
                        tib_fft_vals = np.loadtxt(tib_fft, skiprows=2, delimiter=",")
                    except:
                        tib_fft_vals = np.loadtxt(
                            tib_fft,
                            skiprows=2,
                            delimiter=",",
                            usecols=(0, 1, 2, 3, 4, 5),
                        )

                    try:
                        fem_fft_perfect_vals = np.loadtxt(
                            fem_fft_perfect, skiprows=2, delimiter=","
                        )
                    except:
                        fem_fft_perfect_vals = np.loadtxt(
                            fem_fft_perfect,
                            skiprows=2,
                            delimiter=",",
                            usecols=(0, 1, 2, 3, 4, 5),
                        )

                    try:
                        tib_fft_perfect_vals = np.loadtxt(
                            tib_fft_perfect, skiprows=2, delimiter=","
                        )
                    except:
                        tib_fft_perfect_vals = np.loadtxt(
                            tib_fft_perfect,
                            skiprows=2,
                            delimiter=",",
                            usecols=(0, 1, 2, 3, 4, 5),
                        )

                    # print(kin_jta_vals.shape)

                    number_list = np.array(
                        [
                            np.round(fem_gt_vals, 2),
                            np.round(fem_jta_vals, 2),
                            np.round(fem_fft_vals, 2),
                            np.round(fem_fft_perfect_vals, 2),
                            np.round(tib_gt_vals, 2),
                            np.round(tib_jta_vals, 2),
                            np.round(tib_fft_vals, 2),
                            np.round(tib_fft_perfect_vals, 2),
                            np.round(kin_gt_vals, 2),
                            np.round(kin_jta_vals, 2),
                            np.round(kin_fft_vals, 2),
                            np.round(fem_gt_vals - fem_jta_vals, 2),
                            np.round(fem_gt_vals - fem_fft_vals, 2),
                            np.round(fem_gt_vals - fem_fft_perfect_vals, 2),
                            np.round(tib_gt_vals - tib_jta_vals, 2),
                            np.round(tib_gt_vals - tib_fft_vals, 2),
                            np.round(tib_gt_vals - tib_fft_perfect_vals, 2),
                            np.round(kin_gt_vals - kin_jta_vals, 2),
                            np.round(kin_gt_vals - kin_fft_vals, 2),
                            np.round(kin_gt_vals - kin_fft_perfect_vals, 2),
                        ]
                    )

                    N = fem_gt_vals[:, 0].size

                    for i in range(0, N):
                        list = [
                            study,
                            pat_id,
                            sess_id,
                            mvt_id,
                            i,
                            number_list[0, i, :],
                            number_list[1, i, :],
                            number_list[2, i, :],
                            number_list[3, i, :],
                            number_list[4, i, :],
                            number_list[5, i, :],
                            number_list[6, i, :],
                            number_list[7, i, :],
                            number_list[8, i, :],
                            number_list[9, i, :],
                            number_list[10, i, :],
                            number_list[11, i, :],
                            number_list[12, i, :],
                            number_list[13, i, :],
                            number_list[14, i, :],
                            number_list[15, i, :],
                            number_list[16, i, :],
                            number_list[17, i, :],
                            number_list[18, i, :],
                            number_list[19, i, :],
                        ]
                        # temp_array = np.array([list])
                        # print(temp_array)
                        temp_frame = pd.DataFrame([list], columns=col)
                        # print(temp_frame)
                        df = pd.concat([df, temp_frame], axis=0, ignore_index=True)
    # Having issue with truncating values: https://stackoverflow.com/questions/53316471/pandas-dataframes-to-csv-truncates-long-values
    np.set_printoptions(
        threshold=sys.maxsize
    )  # prevents the csv from truncating longer values
    # df.to_csv('test.csv', index = False)
    df.to_pickle("test.kin")
    df.to_html("df.html")

    # At this point, we have created a pickle that contains all the data that we need. In order to un-pickle the data, use the following synatx

    """
    with open('file_name.pkl', 'rb') as file:
        my_data = pickle.load(file)
    """


if __name__ == "__main__":
    main()
