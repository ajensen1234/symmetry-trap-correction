import numpy as np
import kinproc.jtsFunctions
import kinproc.process
import kinproc.rot
import kinproc.utility
import kinproc
import pickle
import pandas as pd
import numpy as np
from Kinematics_Dataframe import Kinematics_Dataframe

pd.options.display.max_rows = None
pd.options.display.max_columns = None
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import warnings
import kinproc


def bland_altmann_calib(input_vv: float) -> float:
    return ((1 + -1.4673 * 0.5) * input_vv + (-0.35112565)) / (1 - (-1.4673 * 0.5))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def getRotations(sequence, matrix, rangerot2=0):
    # Calculate the corresponding Euler angles from a 3x3 rotation matrix
    # using the specified sequence.

    # Note that this program follows the convention in the field of robotics:
    # We use column vectors (not row vectors) to denote vectors/points in 3D
    # Euclidean space:
    #     V_global = R * V_local
    # where R is a 3x3 rotation matrix, and V_* are 3x1 column vectors.
    # In this convention, for example, an R_z (rotation about z axis) is
    # [c -s 0; s c 0; 0 0 1] and NOT [c s 0; -s c 0; 0 0 1], (c = cosine and s = sine)

    # We also acknowledge that Euler angle rotations are always about the axes of
    # the local coordinate system, never the global coordinate system. (If you would
    # rather recognize the existence of Euler angles with global coordinate system based
    # rotations and further confuse people in the world, simply reverse the order: an x(3 degrees)-y(4deg)-z(5deg)
    # global-base-rotation is simply a z(5)-y(4)-x(3) local based rotation.) Following the equation
    # and convention above, an x-y-z (1-2-3) Euler angle sequence would mean:
    #     V_global = R * V_local
    #              = R_x * R_y * R_x * V_local

    # Sequence:       The rotation sequence used for output (e.g., 312, 213)

    # Matrix:         The 3x3 rotation matrix (the so-called Direction Cosine
    #                 Matrix, or DCM). 4x4 homogeneous transformation matrix is acceptable,
    #                 so long as the first 3 rows and columns are a rotation matrix.

    # rangerot2:      Optional. The range of the second rotation in output. It should be a
    #                 value of 0 (default) or 1. For symmetric Euler sequences, if rangerot2 == 0,
    #                 the 2nd otation is in the range [0, pi]. If rangerot2 == 1, the range is [-pi,0].
    #                 If the second rotation is 0 or pi, singularity occurs.

    #                 For asymmetric Euler sequences:
    #                 if rangerot== 0, the 2nd rotation is in the range [-pi/2, pi/2]; if rangerot==1,
    #                 the range is [pi/2, pi*3/2]. If the second rotation is +/- pi/2, singularity occurs.

    # Author:         Shang Mu, 2005-2010
    # Revision:       v8, 2010-05-23. 2010-07-08
    # Revision:       Amiya Gupta, 2021-08-09
    # Python:         2021-08-09

    def __c312(M):
        s2 = M[2, 1]  # x rot
        if rangerot2 == 0:
            rot2 = np.arcsin(s2)
        else:
            rot2 = np.pi - np.arcsin(s2)
        if s2 == any((0, 1)):  # singularity
            rot1 = 0
            rot3 = np.arctan2(M[0, 2], M[0, 0])
        else:
            if rangerot2 == 0:
                rot1 = np.arctan2(-M[0, 1], M[1, 1])  # z rot
                rot3 = np.arctan2(-M[2, 0], M[2, 2])  # y rot
            else:
                rot1 = np.arctan2(M[0, 1], -M[1, 1])  # z rot
                rot3 = np.arctan2(M[2, 0], -M[2, 2])  # y rot
        return rot1, rot2, rot3

    def __c313(M):
        c2 = M[2, 2]  # x rot
        if rangerot2 == 0:
            rot2 = np.arccos(c2)
        else:
            rot2 = -np.arccos(c2)
        if c2 == any((-1, 1)):  # singularity
            rot1 = 0
            rot3 = np.arctan2(M[2, 0], M[2, 1])
        else:
            if rangerot2 == 0:
                rot1 = np.arctan2(M[0, 2], -M[1, 2])  # z rot
                rot3 = np.arctan2(M[2, 0], M[2, 1])  # y rot
            else:
                rot1 = np.arctan2(-M[0, 2], M[1, 2])  # z rot
                rot3 = np.arctan2(-M[2, 0], -M[2, 1])  # y rot
        return rot1, rot2, rot3

    __rot120 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    if rangerot2 != 1 and rangerot2 != 0:
        raise Exception("Invalid value for parameter rangerot2")

    M = matrix[0:3, 0:3]
    sequence = "c" + str(sequence)

    # Asymmetric Sequences
    if sequence == "c312":
        [rot1, rot2, rot3] = __c312(M)
    elif sequence == "c123":
        [rot1, rot2, rot3] = __c312(__rot120.T @ M @ __rot120)
    elif sequence == "c213":
        [rot1, rot2, rot3] = np.negative(np.flip(__c312(M.T)))
    elif sequence == "c231":
        M = __rot120.T @ M @ __rot120
        [rot1, rot2, rot3] = __c312(__rot120.T @ M @ __rot120)
    elif sequence == "c321":
        [rot1, rot2, rot3] = np.negative(np.flip(__c312(__rot120.T @ M @ __rot120)))
    elif sequence == "c132":
        M = __rot120.T @ M @ __rot120  # 231
        [rot1, rot2, rot3] = np.negative(np.flip(__c312(__rot120.T @ M @ __rot120)))

    # Symmetric Sequences
    elif sequence == "c313":
        [rot1, rot2, rot3] = __c313(M)
    elif sequence == "c212":
        [rot1, rot2, rot3] = __c313(rot.x(-90).T @ M @ rot.x(-90))
    elif sequence == "c232":
        [rot1, rot2, rot3] = __c313(__rot120 @ M @ __rot120.T)
    elif sequence == "c131":
        M = __rot120 @ M @ __rot120.T
        [rot1, rot2, rot3] = __c313(rot.x(-90).T @ M @ rot.x(-90))
    elif sequence == "c121":
        M = __rot120 @ M @ __rot120.T
        M = rot.x(-90).T @ M @ rot.x(-90)
        [rot1, rot2, rot3] = __c313(rot.x(-90).T @ M @ rot.x(-90))

    else:
        raise Exception("getRotations(): Sequence not yet supported")
        rotations = []

    return np.rad2deg(rot1), np.rad2deg(rot2), np.rad2deg(rot3)
    # Sidenote:

    # These methods are used to compute the sequences from 312 and 313 to simplify the program and increase the efficiency.

    # There are many ways to simplify this program:

    # Preliminary (for methods 1 and 2):
    # The 12 different Euler angle sequences can be divided into 4 groups:
    # (123, 231, 312), (321, 213, 132) , (121, 232, 313), (323, 212, 131).
    # (The first two groups can be further combined to a single one using
    # method 3 below. Similarly the last two groups can be united by method 4.)

    # Simplification method 1:
    # Basic idea: if I have two different coordinate systems on the same rigid
    # body, an x-rotation seen from one coordinate system can be a y-rotation
    # in the other coordinate system.
    # The tool is a rotation matrix:
    #   rot120 = [
    #      0     0     1
    #      1     0     0
    #      0     1     0
    #   ];  (a shift of axis indices, or a 120 degree rotation about [1 1 1]).
    # For any sequence, the three Euler angles can be easily calculated using
    # the same program for any other sequence in the same group (as defined in
    # the "preliminary" section). For example (assuming M is 3x3):
    # getRotations(123, M) == getRotations(312, rot120.'*M*rot120)
    #                      == getRotations(231, rot120*M*rot120.');
    # getRotations(232, M) == getRotations(121, rot120.'*M*rot120)
    #                      == getRotations(313, rot120*M*rot120.').

    # Simplification method 2:
    # There are rules we can follow. For example, s2 or c2 (sine or cosine of
    # the 2nd rotation) is always at the (1st rot)(3rd rot) element in the
    # rotation matrix. The four groups we saw above are the only variations we
    # need to take care of.

    # Simplification method 3:
    # An A-B-C order Euler angle rotations of a body EE with respect to some
    # fixed body FF, can be seen as a C-B-A order rotations of the body FF with
    # respect to the body EE (but with the negative angles). For example
    # (assuming M is 3x3):
    # getRotations(123, M) == -getRotations(321, M.')(end:-1:1);
    # getRotations(312, M) == -getRotations(213, M.')(end:-1:1).

    # Simplification method 4:
    # Similar to method 1, a relabel of axes utilizing 90 degree rotations is
    # extremely useful for the symmetrical sequences. A 90 degree rotation
    # about either of the two axes in a symmetrical sequence would result in a
    # new valid sequence. For example:
    # getRotations(212, M) == getRotations(232, roty(90).'*M*roty(90))
    #                      == getRotations(313, rotx(-90).'*M*rotx(-90)).
    # This essentially unites all the symmetric sequences.
    # This method could also be used on the asymmetric sequences, but care
    # must be taken to negate the sign of individual angles.


def kin_rms(dataframe, list_of_diffs, cutoff):
    """
    This function calculates the rms values of each kinematic measurement when you feed in a dataframe
    """
    rms = []
    test_dict = {"place": "holder"}
    for idx, metric in enumerate(list_of_diffs):
        list = []
        for i in range(0, dataframe.shape[0]):
            list.append(dataframe["kin_diff"][i][idx])

        np_list = np.array(list)
        bool = abs(np_list) > cutoff
        test_dict.update({metric: bool})
        rms.append(np.sqrt(np.mean(np_list**2)))
    return rms, test_dict


def df_rms(dataframe, column, list_of_diffs, cutoff):
    rms = []
    test_dict = {"place": "holder"}
    for idx, metric in enumerate(list_of_diffs):
        list = []
        for i in range(0, dataframe.shape[0]):
            list.append(dataframe[column][i][idx])

        np_list = np.array(list)
        bool = abs(np_list) > cutoff
        test_dict.update({metric: bool})
        rms.append(np.sqrt(np.mean(np_list**2)))
    return rms, test_dict


# The ambiguous pose function - determines the offset
def ambiguous_pose(pose):
    xr = kinproc.rot.x(pose[4])
    yr = kinproc.rot.y(pose[5])
    zr = kinproc.rot.z(pose[3])

    R = np.matmul(np.matmul(zr, xr), yr)

    y_ax = R[:, 1]

    com = np.array(pose[0:3])
    normed = com / np.linalg.norm(com)

    amg_by_90 = abs(np.dot(y_ax, normed))
    angle_between = np.rad2deg(np.arccos(amg_by_90))

    return abs(angle_between - 90)


def blunder(rots, thresh):
    zr_diff = rots[0]
    xr_diff = rots[1]
    yr_diff = rots[2]
    if math.sqrt(xr_diff**2 + yr_diff**2 + zr_diff**2) < thresh:
        return False
    else:
        return True


def ambiguous_pose_x(pose):
    xr = kinproc.rot.x(pose[4])
    yr = kinproc.rot.y(pose[5])
    zr = kinproc.rot.z(pose[3])

    R = np.matmul(np.matmul(zr, xr), yr)

    x_ax = R[:, 0]

    com = np.array(pose[0:3])
    normed = com / np.linalg.norm(com)

    amg_by_90 = abs(np.dot(x_ax, normed))
    angle_between = np.rad2deg(np.arccos(amg_by_90))

    return abs(angle_between - 90)


def axis_angle_rotation_matrix(axis, angle):
    m = axis
    c = np.cos(angle)
    s = np.sin(angle)
    v = 1 - c

    return np.array(
        [
            [
                m[0] * m[0] * v + c,
                m[0] * m[1] * v - m[2] * s,
                m[0] * m[2] * v + m[1] * s,
            ],
            [
                m[0] * m[1] * v + m[2] * s,
                m[1] * m[1] * v + c,
                m[1] * m[2] * v - m[0] * s,
            ],
            [
                m[0] * m[2] * v - m[1] * s,
                m[1] * m[2] * v + m[0] * s,
                m[2] * m[2] * v + c,
            ],
        ]
    )


def angle_solver_312(rotation_matrix):
    R = rotation_matrix

    xr = np.rad2deg(np.arctan2(R[2, 1], math.sqrt(R[0, 1] ** 2 + R[1, 1] ** 2)))
    yr = np.rad2deg(np.arctan2(-R[2, 0] / np.cos(xr), R[2, 2] / np.cos(xr)))
    zr = np.rad2deg(np.arctan2(-R[0, 1] / np.cos(xr), R[1, 1] / np.cos(xr)))

    return xr, yr, zr


def sym_trap_dual(pose):
    # First, we find the Rotation matrix that describes the pose in space.
    xr = kinproc.rot.x(pose[4])
    yr = kinproc.rot.y(pose[5])
    zr = kinproc.rot.z(pose[3])

    R = np.matmul(np.matmul(zr, xr), yr)

    # We want to look at the z-axis
    z_ax = R[:, 2]

    # Compare the object's z-axis with the angle of the observer - > center of mass
    com = np.array(pose[0:3])

    normed = com / np.linalg.norm(com)

    # Transform the new vector so that it goes from COM -> observer
    normed = -normed

    # Find the angle between the two, take abs() to get total angle (we let cross product handle the direction of rotation)
    costh = abs(np.dot(z_ax, normed))

    angle_between = np.rad2deg(np.arccos(costh))
    #
    # Find the axis of rotation, M is notation from Crane and Duffy
    M = np.cross(z_ax, normed)

    # Need to normalize M
    M_norm = M / np.linalg.norm(M)

    # At this point, we know that the amount we want to rotate is double the angle between z_ax and V
    desired_rotation = 2 * angle_between
    des_rot_rad = np.deg2rad(desired_rotation)

    # Solve for the rotation matrix (above formula)
    sym_R = axis_angle_rotation_matrix(axis=M_norm, angle=des_rot_rad)

    # need to tinker with this a little bit
    new_pose = np.matmul(sym_R, R)
    zr, xr, yr = getRotations(sequence="312", matrix=new_pose)

    return pose[0], pose[1], pose[2], zr, xr, yr


def solid_angle_distance_to_dual(pose):
    # First, we find the Rotation matrix that describes the pose in space.
    xr = kinproc.rot.x(pose[4])
    yr = kinproc.rot.y(pose[5])
    zr = kinproc.rot.z(pose[3])

    # R = np.matmul(np.matmul(zr,xr),yr)
    R = zr @ xr @ yr
    # We want to look at the z-axis
    z_ax = R[:, 2]

    # Compare the object's z-axis with the angle of the observer - > center of mass
    com = np.array(pose[0:3])

    normed = com / np.linalg.norm(com)

    # Transform the new vector so that it goes from COM -> observer
    normed = -normed

    # Find the angle between the two, take abs() to get total angle (we let cross product handle the direction of rotation)
    costh = abs(np.dot(z_ax, normed))

    angle_between = np.rad2deg(np.arccos(costh))
    #
    # Find the axis of rotation, M is notation from Crane and Duffy
    M = np.cross(z_ax, normed)

    # Need to normalize M
    M_norm = M / np.linalg.norm(M)

    # At this point, we know that the amount we want to rotate is double the angle between z_ax and V
    desired_rotation = 2 * angle_between

    return desired_rotation


def sym_trap_solid_distance(pose):
    # First, we find the Rotation matrix that describes the pose in space.
    xr = kinproc.rot.x(pose[4])
    yr = kinproc.rot.y(pose[5])
    zr = kinproc.rot.z(pose[3])

    R = np.matmul(np.matmul(zr, xr), yr)

    # We want to look at the z-axis
    z_ax = R[:, 2]

    # Compare the object's z-axis with the angle of the observer - > center of mass
    com = np.array(pose[0:3])

    normed = com / np.linalg.norm(com)

    # Transform the new vector so that it goes from COM -> observer
    normed = -normed

    # Find the angle between the two, take abs() to get total angle (we let cross product handle the direction of rotation)
    costh = abs(np.dot(z_ax, normed))
    angle_between = np.rad2deg(np.arccos(costh))

    # Find the axis of rotation, M is notation from Crane and Duffy
    M = np.cross(z_ax, normed)

    # Need to normalize M
    M_norm = M / np.linalg.norm(M)

    # At this point, we know that the amount we want to rotate is double the angle between z_ax and V
    desired_rotation = 2 * angle_between

    return desired_rotation

def add_symmetry_trap_columns_to_dataframe(df):
    sym_trap_distance = []
    sym_trap_pose = []
    dual_pose = []
    for i in range(0,df.df.shape[0]):
        sym_trap_distance.append(solid_angle_distance_to_dual(df.df["tib"][i]))
        sym_trap_pose.append(np.round(sym_trap_dual(df.df["tib"][i]),3))
        tib_pose_dual = sym_trap_pose[i]
        fem_pose = df.df["fem"][i]
        side = df.df["Side"][i]
        dual_pose.append(
            np.round(kinproc.process.relative_kinematics(
                fem_pose=fem_pose,
                tib_pose=tib_pose_dual,
                side=side,
            ),
            3)
        )
    df.add_column("sym_trap_distance",sym_trap_distance)
    df.add_column("sym_trap_pose",sym_trap_pose)
    df.add_column("kin_dual",dual_pose)
    return df

def create_symmetry_trap_dataset_and_labels(df: Kinematics_Dataframe):
    kin_z = df.grab_index_from_column("kin",3)
    kin_x = df.grab_index_from_column("kin",4)
    kin_y = df.grab_index_from_column("kin",5)
    dual_z = df.grab_index_from_column("kin_dual",3)
    dual_x = df.grab_index_from_column("kin_dual",4)
    dual_y = df.grab_index_from_column("kin_dual",5)
    sym_dist = df.grab_data_from_column("sym_trap_distance")
    
    # random state for reproduceability
    np.random.seed(42)
    
    # Create an array of true kinematics, with 1 as the target
    X = np.array([kin_x, kin_y, kin_z, sym_dist]).T
    t = np.ones((X.shape[0],1))
    
    # append array of "Symmetry trap" kinematics with 0 as target
    X = np.append(X, np.array([dual_x, dual_y, dual_z, sym_dist]).T, axis=0)
    t = np.append(t, np.zeros((X.shape[0] - t.shape[0],1)), axis=0)
    t = t.ravel()
    
    # Randomize the order of the data
    
    idx = np.random.permutation(X.shape[0])
    X_rand, t_rand = X[idx], t[idx]
    
    # save data
    np.save("X.npy", X_rand)
    np.save("t.npy", t_rand)
    return X_rand, t_rand
    