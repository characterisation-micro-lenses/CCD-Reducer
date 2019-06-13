from FitsLoader import FitsLoader  # loads a singe file
from FitsReductionObject import FitsBias, FitsDark, FitsFlat  # creates and loads master calibration files
from FitsReducer import FitsReducer  # calibrates a single file
from FitsBackFocalPlaneAnalyser import FitsBackFocalPlaneAnalyser  # get the T parameter.
from FitsLaserReducer import FitsLaserReducer  # same
from FitsFolderLaserReducer import FitsFolderLaserReducer  # calibrates multiple files
from FitsFocus import FitsFocus  #calibrates and charactarizes a dataset
from FocusGenerator import FocusGenerator  # calculates an aberrated focus
from FitsFocusFitter import FitsFocusFitter  # fits the actual focus from FitsFocus with a calculated focus
import support_functions as sf  # mathematical support functions                              from FocusGenerator

import numpy as np
import os

# Most Darkened!!
# number of counts/s/pixel approx 529.247
I0_dark = 529.247

I0_dark2 = 5.88052e9 #counts/s/m^2

# Less Darkened!!
# number of counts/s/pixel 13888.148
I0_less_dark = 13888.148


schoolpath = "//data02.physics.leidenuniv.nl/pi-vanexter/Nanoscribe/2019 Nanoscribe/Measurements/"
homepath = "C:/Users/anjun/CloudStation/School/18-19 Natuur&Sterrenkunde 3/\
BRP/Nanoscribe/2019 Nanoscribe/Measurements/"

#mainpath = schoolpath
mainpath = homepath


cam2path = mainpath + "CCD Images/Camera 2 (Connection error) [main]/"
biaspath = cam2path + "BIAS/"
darkpath = cam2path + "DARK/"
flatpath = cam2path + "FLAT/"
masterpath = cam2path + "Master/"


setup2path = cam2path + "LIGHT/Setup 2 100x/"

set2_red_airpath = setup2path + "Red Lens 20x/Air/"
set2_red_gppath = setup2path + "Red Lens 20x/Glass Plate/"
set2_red_halfpath = setup2path + "Red Lens 20x/Air Half Illuminated/"

set2_cor_0path = setup2path + "Correction Lens 40x/0 Correction/"
set2_cor_4path = setup2path + "Correction Lens 40x/4 Correction/"
set2_cor_8path = setup2path + "Correction Lens 40x/8 Correction/"
set2_cor_12path = setup2path + "Correction Lens 40x/12 Correction/"
set2_cor_16path = setup2path + "Correction Lens 40x/16 Correction/"
set2_cor_20path = setup2path + "Correction Lens 40x/20 Correction/"
set2_cor_negpath = setup2path + "Correction Lens 40x/Negative Correction/"
set2_cor_1mmpath = setup2path + "Correction Lens 40x/1 mm Plate/"

set2_sample15path = setup2path + "Sample Batch 1/Sample 5/"
set2_sample21piezopath = setup2path + "Sample Batch 2/Piezo/Sample 1/"  # f = 40
set2_sample22piezopath = setup2path + "Sample Batch 2/Piezo/Sample 2/"  # D = 40
set2_sample24piezopath = setup2path + "Sample Batch 2/Piezo/Sample 4/"  # D = 100

set2_sample21galvopath = setup2path + "Sample Batch 2/Galvo/Sample 1/"  # D = 100
set2_sample22galvopath = setup2path + "Sample Batch 2/Galvo/Sample 2/"  # D = 40
set2_sample25galvopath = setup2path + "Sample Batch 2/Galvo/Sample 5/"  # f = 40

set2_sample31path = setup2path + "Sample Batch 3/Sample 1 (failed)/"
set2_sample32path = setup2path + "Sample Batch 3/Sample 2/"


paths = [set2_red_airpath, set2_red_gppath, set2_cor_0path, set2_cor_4path, set2_cor_8path, set2_cor_12path,
         set2_cor_16path, set2_cor_20path, set2_cor_negpath, set2_sample15path, set2_sample21piezopath,
         set2_sample22piezopath, set2_sample24piezopath, set2_sample21galvopath, set2_sample22galvopath,
         set2_sample25galvopath, set2_sample31path, set2_sample32path]


no_lenspath = setup2path+"No Lens/"
weak_focus = no_lenspath+"Weak Focus/"

folderpath = set2_red_airpath
filename = folderpath + sorted(os.listdir(folderpath))[4]
savepath = folderpath + "Pictures/"





# f = FitsLaserReducer(filename=filename, masterpath=masterpath, savepath=savepath)
# f.imshow(), f.imsave(), f.slicing, f.slicing_save(), f.power(), f.power_save()

# f = FitsFolderLaserReducer(folderpath, masterpath=masterpath, savepath=savepath)
# f.all_fit_save(), f.all_fit_reduced(), f.all_fit_power(), f.cube()

# f = FitsFocus(folderpath, masterpath=masterpath, savepath=savepath)
# f.show(), f.charactarize_focus(), f.charactarize_focus_save()
