import cv2
import numpy as np
import os
import random
# from matplotlib.pyplot import *
# # from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
# import matplotlib.pyplot as plt
# # PythonETpackage for xml file edition
import pickle


def self_check_path_create(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def save_a_pkl(dir,name,object):
    with open(dir + name +'.pkl', 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)
    pass

def read_a_pkl (dir,name ):
    object = pickle.load(open(dir + name +'.pkl', 'rb'), encoding='iso-8859-1')
    return object


def save_img_to_folder(this_save_dir,ID,img):
    # this_save_dir = Output_root + "1out_img/" + Model_key + "/ground_circ/"
    if not os.path.exists(this_save_dir):
        os.makedirs(this_save_dir)
    cv2.imwrite(this_save_dir +
                str(ID) + ".jpg", img)