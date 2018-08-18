import pySALEPlot as psp
import numpy as np
from pylab import figure, arange, colorbar, ma, sqrt, figtext
import bisect
import operator
from scipy.optimize import fsolve
import math
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib import ticker


# ---------------------------- #
#       LOAD JDATA FILE        #
# ---------------------------- #

####################################################################################################################################### USER
# Name the model directory
namemodel = 'MODEL'

# Define datafilename
datafilename = namemodel + '/jdata.dat'
####################################################################################################################################### USER

# Open the datafile
model=psp.opendatfile(datafilename)
    
# Set the distance units to km
model.setScale('km')

# ----------------------------------- #
# OBTAIN AND PRINT TRACER INFORMATION #
# ----------------------------------- #

# Number of tracer clouds
tr_cloud_num = model.tracer_numu

print "Number of tracer clouds in model is {}".format(tr_cloud_num)
print "==================================="

# Determine the first and last tracer numbers of each of the tracer clouds
cloud_start = []
cloud_end = []

for t in np.arange(0,tr_cloud_num,1):
    cloud_start.append(model.tru[t].start)
    cloud_end.append(model.tru[t].end)

# --------------------------------------------- #
# SELECT METHOD TO DETERMINE TRACER OF INTEREST #
# --------------------------------------------- #

print "How would you like to find the tracer of choice?"
print "-- Enter '1', '2', or '3' --"
choice = float(input("Final Location (1), Initial Location (2), or Tracer Number (3)? - "))

if choice == 1:

    # Input x-coordinate
    print "Input the x-coordinate (in km) of the tracer you are interested in:"
    x_coord = float(input("x-coordinate = "))

    # Input y-coordinate
    print "Input the y-coordinate (in km) of the tracer you are interested in (must be negative):"
    y_coord = float(input("y coordinate = "))

    print "==================================="
    print "Location of interest is ({},{}) during the final timestep".format(x_coord, y_coord)
    print "==================================="

elif choice == 2:
    
    # Input x-coordinate
    print "Input the x-coordinate (in km) of the tracer you are interested in:"
    x_coord = float(input("x-coordinate = "))
    
    # Input y-coordinate
    print "Input the y-coordinate (in km) of the tracer you are interested in (must be negative):"
    y_coord = float(input("y coordinate = "))
    
    print "==================================="
    print "Location of interest is ({},{}) during the initial timestep".format(x_coord, y_coord)
    print "==================================="

elif choice == 3:
    
    # Input x-coordinate
    print "Input the ID number of the tracer you are interested in:"
    TR_NUMBER = float(input("Tracer Number = "))
    
    print "==================================="
    print "Tracer of interest is ({})".format(TR_NUMBER)
    print "==================================="

else:
    print "ERROR! - FAILED TO CHOSE VALID SELECTION METHOD - PLEASE RESTART SCRIPT"

# -------------------------------------------------------------------------------- #
# PRINT THE TRACERS LOCATION AND CALCULATE THE TRACER IDs OF THE 4 NEAREST TRACERS #
# -------------------------------------------------------------------------------- #
# Only the 4 tracers that make the square cross-section of the Lagrangian          #
# pseudo-cell are required to calculate all strain data. Hoop strains are          #
# calculated by using the change in r-coordinate.                                  #

# Read the initial and final step
step_initial = model.readStep('Trd', 0)
step_final = model.readStep('Trd', model.nsteps-1)
print "==================================="

# Locate closest tracer and determine tracer ID
if choice == 1:
    TR = step_final.findTracer(x_coord,y_coord)
elif choice == 2:
    TR = step_initial.findTracer(x_coord,y_coord)
elif choice == 3:
    TR = int(TR_NUMBER)
else:
    print "ERROR"
    
TR_cl = bisect.bisect_left(cloud_start, TR)

print "Tracer of interest is #{}, within tracer cloud #{}".format(TR, TR_cl)

if choice == 1:
    print "Tracer of interest, #{}, was at ({}, {}) in the initial timestep".format(TR, step_initial.xmark[TR], step_initial.ymark[TR])
elif choice == 2:
    print "Tracer of interest, #{}, will be at ({}, {}) in the final timestep".format(TR, step_final.xmark[TR], step_final.ymark[TR])
elif choice == 3:
    print "Tracer of interest, #{}, was at ({}, {}) in the initial timestep".format(TR, step_initial.xmark[TR], step_initial.ymark[TR])
    print "Tracer of interest, #{}, will be at ({}, {}) in the final timestep".format(TR, step_final.xmark[TR], step_final.ymark[TR])
else:
    print "ERROR"
print "==================================="

# Determine the tracer IDs and location of the top-left (TL), top-right (TR), bottom-left (BL), and bottom-right (BR) tracers
tr_spacing = model.tru[TR_cl-1].d

TR_TL = step_initial.findTracer(step_initial.xmark[TR] - tr_spacing[0], step_initial.ymark[TR] + tr_spacing[1])
TR_TR = step_initial.findTracer(step_initial.xmark[TR] + tr_spacing[0], step_initial.ymark[TR] + tr_spacing[1])
TR_BL = step_initial.findTracer(step_initial.xmark[TR] - tr_spacing[0], step_initial.ymark[TR] - tr_spacing[1])
TR_BR = step_initial.findTracer(step_initial.xmark[TR] + tr_spacing[0], step_initial.ymark[TR] - tr_spacing[1])

# Determine which tracer, of TL, TR, BL, and BR, is closest to the input coordinates
TR_TL_x = step_final.xmark[TR_TL]
TR_TL_y = step_final.ymark[TR_TL]

TR_TR_x = step_final.xmark[TR_TR]
TR_TR_y = step_final.ymark[TR_TR]

TR_BL_x = step_final.xmark[TR_BL]
TR_BL_y = step_final.ymark[TR_BL]

TR_BR_x = step_final.xmark[TR_BR]
TR_BR_y = step_final.ymark[TR_BR]

if choice == 3:
    x_coord = step_initial.xmark[TR] + 0.00000000001
    y_coord = step_initial.ymark[TR] + 0.00000000001

# Calculate distance between prefered coordinates and TR_TL, TR_TR, TR_BL, TR_BR
TL_distance = sqrt(((x_coord - TR_TL_x)**2) + ((y_coord - TR_TL_y)**2))
TR_distance = sqrt(((x_coord - TR_TR_x)**2) + ((y_coord - TR_TR_y)**2))
BL_distance = sqrt(((x_coord - TR_BL_x)**2) + ((y_coord - TR_BL_y)**2))
BR_distance = sqrt(((x_coord - TR_BR_x)**2) + ((y_coord - TR_BR_y)**2))

distances = [TL_distance, TR_distance, BL_distance, BR_distance]

min_index, min_value = min(enumerate(distances), key=operator.itemgetter(1))

# Determine the IDs of the two remaining tracers that complete the square
if min_index == 0:
    tl = TR_TL
    tr = TR + 1
    bl = TR_TL - 1
    br = TR
elif min_index == 1:
    tl = TR + 1
    tr = TR_TR
    bl = TR
    br = TR_TR - 1
elif min_index == 2:
    tl = TR_BL + 1
    tr = TR
    bl = TR_BL
    br = TR - 1
elif min_index == 3:
    tl = TR
    tr = TR_BR + 1
    bl = TR - 1
    br = TR_BR
else:
    print "**** ERROR ****"

# Print output
print "The selected Lagrangian pseudo-cell is made of tracers:"
print "# {} --- {} #".format(tl, tr)
print "# --------------- #"
print "# {} --- {} #".format(bl,br)
print "==================================="

# Calculate Time Resolution
T_res = (step_final.time - step_initial.time)/model.nsteps

####################################################################################################################################### USER
# Create output directory
dirname = 'StressStrain_{}'.format(TR)
psp.mkdir_p(namemodel + '/' + dirname)
####################################################################################################################################### USER

# ---------------------------------------------------------------------------------- #
# LOOP OVER TIMESTEPS TO FIND LOCATION DATA AND STRESS DATA, THEN ORGANISE THAT DATA #
# ---------------------------------------------------------------------------------- #

# ------------------- NOTE ------------------- #
# DEVIATORIC STRESS TENSOR in 3D has the form #
# ( 'TrX'    'TrR'      0   )
# ( 'TrR'    'TrY'      0   )
# (   0        0      'TrH' )

# FULL STRESS TENSOR in 3D has the form #
# ( 'TrX'-'TrP'    'TrR'         0   )
# ( 'TrR'       'TrY'-'TrP'      0   )
# (   0              0      'TrH'-'TrP' )
# ------------------- NOTE ------------------- #

# Create lists to store data

## Location data
tl_xlocations = []
tl_ylocations = []

tr_xlocations = []
tr_ylocations = []

bl_xlocations = []
bl_ylocations = []

br_xlocations = []
br_ylocations = []

AVERAGE_X = []
AVERAGE_Y = []

x_loc = []
y_loc = []
t_av = []

## Pressure, Temperature, and Distension data
pre = []
temp = []
dist = []

## Stress data
sig_11 = []
sig_12 = []
sig_22 = []
sig_33 = []

sigSTAR_11 = []
sigSTAR_12 = []
sigSTAR_22 = []
sigSTAR_33 = []

FULL_STRESS_tensor = []
DEV_STRESS_tensor = []

STRESS_I1 = []
STRESS_I2 = []
STRESS_I3 = []

STRESS_e_val1 = []
STRESS_e_vec1x = []
STRESS_e_vec1y = []
STRESS_e_vec1z = []

STRESS_e_val2 = []
STRESS_e_vec2x = []
STRESS_e_vec2y = []
STRESS_e_vec2z = []

STRESS_e_val3 = []
STRESS_e_vec3x = []
STRESS_e_vec3y = []
STRESS_e_vec3z = []

# Loop over timesteps
for i in np.arange(0, model.nsteps,1):
    
    # Read step
    step = model.readStep(['Trp','TrX','TrY','TrR','TrH', 'Trt', 'TrA'], i)
    
    # Loop to calculate the "Pseudo-cell" centered and time-interpolated properties
    if i>0:
        step_n = model.readStep(['Trp','TrX','TrY','TrR','TrH', 'Trt', 'TrA'], i-1)
        
        x_loc.append((step.xmark[TR] + step_n.xmark[TR])/2)
        y_loc.append((step.ymark[TR] + step_n.ymark[TR])/2)
        t_av.append((step.time + step_n.time)/2)
        temp.append((step.data[5][TR] + step_n.data[5][TR])/2)
        dist.append((step.data[6][TR] + step_n.data[6][TR])/2)
    
        # Calculate the "Pseudo-cell" Centered and Time-interpolated Full and Deviatoric Stress Tensors
        p_tl = ((step.data[0][tl] + step_n.data[0][tl])/2)
        p_tr = ((step.data[0][tr] + step_n.data[0][tr])/2)
        p_bl = ((step.data[0][bl] + step_n.data[0][bl])/2)
        p_br = ((step.data[0][tr] + step_n.data[0][tr])/2)
        
        p_av = (p_tl + p_tr + p_bl + p_br) / 4
        pre.append(p_av)
        
        sig_11_tl = ((step.data[1][tl] + step_n.data[1][tl])/2)
        sig_11_tr = ((step.data[1][tr] + step_n.data[1][tr])/2)
        sig_11_bl = ((step.data[1][bl] + step_n.data[1][bl])/2)
        sig_11_br = ((step.data[1][tr] + step_n.data[1][tr])/2)
    
        sig_11_av = (sig_11_tl + sig_11_tr + sig_11_bl + sig_11_br) / 4
        sig_11.append(sig_11_av + p_av)
    
        sig_12_tl = ((step.data[3][tl] + step_n.data[3][tl])/2)
        sig_12_tr = ((step.data[3][tr] + step_n.data[3][tr])/2)
        sig_12_bl = ((step.data[3][bl] + step_n.data[3][bl])/2)
        sig_12_br = ((step.data[3][tr] + step_n.data[3][tr])/2)
    
        sig_12_av = (sig_12_tl + sig_12_tr + sig_12_bl + sig_12_br) / 4
        sig_12.append(sig_12_av)

        sig_22_tl = ((step.data[2][tl] + step_n.data[2][tl])/2)
        sig_22_tr = ((step.data[2][tr] + step_n.data[2][tr])/2)
        sig_22_bl = ((step.data[2][bl] + step_n.data[2][bl])/2)
        sig_22_br = ((step.data[2][tr] + step_n.data[2][tr])/2)
    
        sig_22_av = (sig_22_tl + sig_22_tr + sig_22_bl + sig_22_br) / 4
        sig_22.append(sig_22_av + p_av)
    
        sig_33_tl = ((step.data[4][tl] + step_n.data[4][tl])/2)
        sig_33_tr = ((step.data[4][tr] + step_n.data[4][tr])/2)
        sig_33_bl = ((step.data[4][bl] + step_n.data[4][bl])/2)
        sig_33_br = ((step.data[4][tr] + step_n.data[4][tr])/2)
    
        sig_33_av =(sig_33_tl + sig_33_tr + sig_33_bl + sig_33_br) / 4
        sig_33.append(sig_33_av + p_av)
    
        sigSTAR_11.append(sig_11_av)
        sigSTAR_12.append(sig_12_av)
        sigSTAR_22.append(sig_22_av)
        sigSTAR_33.append(sig_33_av)
    
    
        row1 = np.array([sig_11_av - p_av, sig_12_av, 0])
        row2 = np.array([sig_12_av, sig_22_av - p_av, 0])
        row3 = np.array([0, 0, sig_33_av - p_av])
    
        F_S_TENS = np.matrix( [row1 , row2 , row3] )
    
        FULL_STRESS_tensor.append(F_S_TENS)
    
    
        row1 = np.array([sig_11_av, sig_12_av, 0])
        row2 = np.array([sig_12_av, sig_22_av, 0])
        row3 = np.array([0, 0, sig_33_av])
    
        D_S_TENS = np.matrix( [row1 , row2 , row3] )
    
        DEV_STRESS_tensor.append(D_S_TENS)
    
        # Determine Stress Invariants
        STRESS_Inv1 = np.trace(D_S_TENS)
        STRESS_Inv2 = (np.trace(D_S_TENS**2) - (np.trace(D_S_TENS))**2)/2
        STRESS_Inv3 = np.linalg.det(D_S_TENS)
    
        STRESS_I1.append(STRESS_Inv1)
        STRESS_I2.append(STRESS_Inv2)
        STRESS_I3.append(STRESS_Inv3)

        # Determine Principal Stresses and Stress Directions
        STRESS_e_val, STRESS_e_vec = np.linalg.eig(D_S_TENS)
    
        sort = np.argsort(STRESS_e_val)
        STRESS_e_vec = STRESS_e_vec[sort]
        STRESS_e_val = STRESS_e_val[sort]
    
        STRESS_e_val1.append(-STRESS_e_val[0])
        STRESS_e_val2.append(-STRESS_e_val[1])
        STRESS_e_val3.append(-STRESS_e_val[2])
    
        STRESS_e_vec1x.append(STRESS_e_vec[0, 0])
        STRESS_e_vec1y.append(STRESS_e_vec[0, 1])
        STRESS_e_vec1z.append(STRESS_e_vec[0, 2])
    
        STRESS_e_vec2x.append(STRESS_e_vec[1, 0])
        STRESS_e_vec2y.append(STRESS_e_vec[1, 1])
        STRESS_e_vec2z.append(STRESS_e_vec[1, 2])
    
        STRESS_e_vec3x.append(STRESS_e_vec[2, 0])
        STRESS_e_vec3y.append(STRESS_e_vec[2, 1])
        STRESS_e_vec3z.append(STRESS_e_vec[2, 2])
    
    # Obtain Strain Information
    # Determine coordinates of tracers
    tl_x = step.xmark[tl]
    tl_y = step.ymark[tl]
    
    tr_x = step.xmark[tr]
    tr_y = step.ymark[tr]
    
    bl_x = step.xmark[bl]
    bl_y = step.ymark[bl]
    
    br_x = step.xmark[br]
    br_y = step.ymark[br]
    
    tracers_x = np.array([tl_x, tr_x, bl_x, br_x])
    tracers_y = np.array([tl_y, tr_y, bl_y, br_y])
    
    average_x = np.mean(tracers_x)
    average_y = np.mean(tracers_y)
    
    AVERAGE_X.append(average_x)
    AVERAGE_Y.append(average_y)
    
    # DETERMINE LOCATION OF TRACERS RELATIVE TO THE AVERAGE LOCATION AND NORMALISE TO CELL SPACING #
    
    tl_x = (tl_x - average_x)/ tr_spacing[0]
    tl_y = (tl_y - average_y)/ tr_spacing[1]
    
    tl_xlocations.append(tl_x)
    tl_ylocations.append(tl_y)
    
    tr_x = (tr_x - average_x)/ tr_spacing[0]
    tr_y = (tr_y - average_y)/ tr_spacing[1]
    
    tr_xlocations.append(tr_x)
    tr_ylocations.append(tr_y)
    
    bl_x = (bl_x - average_x)/ tr_spacing[0]
    bl_y = (bl_y - average_y)/ tr_spacing[1]
    
    bl_xlocations.append(bl_x)
    bl_ylocations.append(bl_y)
    
    br_x = (br_x - average_x)/ tr_spacing[0]
    br_y = (br_y - average_y)/ tr_spacing[1]
    
    br_xlocations.append(br_x)
    br_ylocations.append(br_y)

# Combine the x and y coordinate lists for each of the tracers
tl_locations = zip(tl_xlocations, tl_ylocations)
tr_locations = zip(tr_xlocations, tr_ylocations)
br_locations = zip(br_xlocations, br_ylocations)
bl_locations = zip(bl_xlocations, bl_ylocations)

# CREATE ARRAYS OF TRACER LOCATIONS ORGANISED BY TIMESTEP (CLOCKWISE AROUND THE SQUARE STARTING AT TOP-LEFT)
T_xlocations = []
T_ylocations = []
T_locations = []

for i in np.arange(0,model.nsteps,1):
    T_xloc = [tl_xlocations[i], tr_xlocations[i], br_xlocations[i], bl_xlocations[i]]
    T_yloc = [tl_ylocations[i], tr_ylocations[i], br_ylocations[i], bl_ylocations[i]]
    
    T_xlocations.append(T_xloc)
    T_ylocations.append(T_yloc)
    T_locations.append(zip(T_xloc, T_yloc))

# ----------------------------------------------------------------------------------------------- NOTE ----------------------------------------------------------------------------------------------- #
# Within the contribution, the Lagrangian pseudo-cell is described as possessing 8 nodes, and thus the matrix tM is an 3*8 matrix, here, tM has been simplified such that it only contains:
#
#      / r_a    r_b     r_c     r_d \
# tM = |                            |  = T_locations
#      \ z_a    z_b     z_c     z_d /
#
# This simplification can be used by ignoring the theta coordinates. Hoop strains are more simply calculated based on translations in the r-direction (x) and, as there cannot be any shear strains in the hoop direction, the other elements are not necessary. See L560. This simplification makes the following section considerably simpler.
# ----------------------------------------------------------------------------------------------- NOTE ----------------------------------------------------------------------------------------------- #

# -------------------------------- #
# SAVE LOCATIONS AT KEY TIME STEPS #
# -------------------------------- #
# Create arrays of tracer locations and areas for the key timesteps
####################################################################################################################################### USER
# The user may wish to add additional "key timesteps" #

Ti_xlocations = T_xlocations[0]
Tf_xlocations = T_xlocations[model.nsteps-1]

Ti_ylocations = T_ylocations[0]
Tf_ylocations = T_ylocations[model.nsteps-1]

Ti_locations = T_locations[0]
Tf_locations = T_locations[model.nsteps-1]

T_XLOCATIONS = [Ti_xlocations, Tf_xlocations]
T_YLOCATIONS = [Ti_ylocations, Tf_ylocations]
T_LOCATIONS = [Ti_locations, Tf_locations]

# -------------------------------------------------------------------------------------------------------------- #
# CALCULATE DEFORMATION AND DISPLACEMENT GRADIENT TENSORS, AND INFINITESIMAL STRAIN TENSORS AND ROTATION TENSORS #
# -------------------------------------------------------------------------------------------------------------- #

# Create lists to append to
I = []
TIME = []

DEF_GRAD_tensor = []

Distension = []

DIS_GRAD_tensor = []
VEL_GRAD_tensor = []

R_of_Def_tensor = []
Spin_tensor = []

EPS_tensor = []
ROT_tensor = []

rot = []
rot_cum = []

gam_xy = []
gam_xy_cum = []

I1 = []
I2 = []
I3 = []

R_I1 = []
R_I2 = []
R_I3 = []

e_val1 = []
e_vec1x = []
e_vec1y = []
e_vec1z = []

e_val2 = []
e_vec2x = []
e_vec2y = []
e_vec2z = []

e_val3 = []
e_vec3x = []
e_vec3y = []
e_vec3z = []

n_oct_unit_shear = []
lode = []
Wk = []
Wk_strain = []

# Calculate the incremental transformations
for i in np.arange(1,model.nsteps,1):
    I.append(i)
    TIME.append(i * T_res)
    
    locs_m = np.transpose(np.array([T_xlocations[i-1], T_ylocations[i-1]]))
    locs_n = np.transpose(np.array([T_xlocations[i], T_ylocations[i]]))
    tran = np.linalg.lstsq(locs_m, locs_n)
    tran = np.transpose(tran[0])
    
    e_hoop = AVERAGE_X[i]/AVERAGE_X[i-1]
    
    # DEFORMATION MATRIX in 3D has the form #
    # ( tran[0][0]    tran[0][1]      0  )
    # ( tran[1][0]    tran[1][1]      0  )
    # (     0             0       e_hoop )
    
    row1 = np.array([tran[0][0], tran[0][1], 0])
    row2 = np.array([tran[1][0], tran[1][1], 0])
    row3 = np.array([0, 0, e_hoop])
    
    TRAN = np.matrix( [row1 , row2 , row3] )
    
    Distension.append(np.linalg.det(TRAN))
    
    DEF_GRAD_tensor.append(TRAN)
    
    row1 = np.array([tran[0][0] - 1, tran[0][1], 0])
    row2 = np.array([tran[1][0], tran[1][1] - 1, 0])
    row3 = np.array([0, 0, e_hoop - 1])
    
    TRAN = np.matrix( [row1 , row2 , row3] )
    
    # D_grad is the displacement gradient tensor
    # V_grad is the velocity gradient tensor
    D_grad = TRAN
    V_grad = TRAN/T_res
    
    # EPS is the symmetric part of the displacement gradient tensor, The Strain Tensor
    # ROT is the anti-symmetric part of the displacement gradient tensor, The Rotation Tensor
    EPS = (D_grad + D_grad.T)/2
    ROT = (D_grad - D_grad.T)/2

    # RoDT is the symmetric part of the velocity gradient tensor, The Rate of Deformation Tensor
    # ST is the anti-symmetric part of the velocity gradient tensor, The Spin Tensor
    RoDT = (V_grad + V_grad.T)/2
    ST =(V_grad - V_grad.T)/2

    DIS_GRAD_tensor.append(D_grad)
    VEL_GRAD_tensor.append(V_grad)
    
    EPS_tensor.append(EPS)
    ROT_tensor.append(ROT)
    
    R_of_Def_tensor.append(RoDT)
    Spin_tensor.append(ST)
    
    #transformation is accessed by [time index], then [row, column]
#    print "................."
#    print transformations[0] = The first incremental deformation matrix
#    print "................."
#    print transformations[0][0] = The first row of first incremental deformation matrix
#    print "................."
#    print transformations[0][0,1] = The second element of the first row of first incremental deformation matrix
#    print "................."

    # Determine the Rotation Vector

    # in 3D, we would need to consider rotations about the x, and y axes too.
    # However, for iSALE2D, all rotations are, by definition, about the hoop direction.

    r_z = (-(ROT[0,1] - ROT[1,0]))/2

    # Where the rotation vector r = [0, 0, r_z],
    # and where the magnitude of r_z is the amount of rotation in radians
    
    rotation = r_z
    if i==1:
        cumulative_rotation = 0
    cumulative_rotation = cumulative_rotation + rotation
    
    rot.append(rotation)
    rot_cum.append(cumulative_rotation)

    # Determine the shear strain
    # Again, as there is only rotation about the hoop direction.
    # Gamma_xy is the only shear strain

    gamma_xy = EPS[0,1]

    if i==1:
        cumulative_gamma_xy = 0
    cumulative_gamma_xy = cumulative_gamma_xy + gamma_xy

    gam_xy.append(gamma_xy)
    gam_xy_cum.append(cumulative_gamma_xy)


    # Determine Strain Invariants
    Inv1 = np.trace(EPS)
    Inv2 = ((np.trace(EPS**2) - (np.trace(EPS))**2)/2)

    if Inv2 >= 0:
        rt_Inv2 = np.sqrt(Inv2)
    else:
        rt_Inv2 = -np.sqrt(-Inv2)

    Inv3 = np.linalg.det(EPS)
    
    R_Inv1 = np.trace(RoDT)
    R_Inv2 = (np.trace(RoDT**2) - (np.trace(RoDT))**2)/2

    if R_Inv2 >= 0:
        rt_R_Inv2 = np.sqrt(R_Inv2)
    else:
        rt_R_Inv2 = -np.sqrt(-R_Inv2)

    R_Inv3 = np.linalg.det(RoDT)
    
    I1.append(Inv1)
    I2.append(rt_Inv2)
    I3.append(Inv3)
    
    R_I1.append(R_Inv1)
    R_I2.append(R_Inv2)
    R_I3.append(R_Inv3)

    # Determine the Strain Ellipsoid
    e_val, e_vec = np.linalg.eig(EPS)

    sort = np.argsort(-e_val)
    e_vec = e_vec[sort]
    e_val = e_val[sort]
    
    e_val1.append(e_val[0])
    e_val2.append(e_val[1])
    e_val3.append(e_val[2])
    
    e_vec1x.append(e_vec[0, 0])
    e_vec1y.append(e_vec[0, 1])
    e_vec1z.append(e_vec[0, 2])
    
    e_vec2x.append(e_vec[1, 0])
    e_vec2y.append(e_vec[1, 1])
    e_vec2z.append(e_vec[1, 2])
    
    e_vec3x.append(e_vec[2, 0])
    e_vec3y.append(e_vec[2, 1])
    e_vec3z.append(e_vec[2, 2])
    
    e1 = 1 + e_val[0]
    e2 = 1 + e_val[1]
    e3 = 1 + e_val[2]

    ne1 = np.log(e1)
    ne2 = np.log(e2)
    ne3 = np.log(e3)

    natural_octahedral_unit_shear = np.sqrt((((ne1-ne2)**2) + ((ne2-ne3)**2) + ((ne3-ne1)**2))/3)
    Lode_number = ((2*ne2) - ne1 - ne3) / (ne1 - ne3)
    
    n_oct_unit_shear.append(natural_octahedral_unit_shear)
    lode.append(Lode_number)

    # Determine Kinematic Vorticity Number
    W_full_kin = np.sqrt( ( ((V_grad[2,1] - V_grad[1,2])**2) + ((V_grad[0,2] - V_grad[2,0])**2) + ((V_grad[1,0] - V_grad[0,1])**2) ) / ( (2 * (V_grad[0,0]**2 + V_grad[1,1]**2 + V_grad[2,2]**2)) + ( (V_grad[0,1] + V_grad[1,0])**2 + (V_grad[0,2] + V_grad[2,0])**2 + (V_grad[1,2] + V_grad[2,1])**2)) )
    
    W_strain_kin = np.sqrt( (RoDT[1,2]**2) + (RoDT[0,2]**2) + (RoDT[0,1]**2) ) / np.sqrt( (2 * ((RoDT[2,2]**2) + (RoDT[1,1]**2) + (RoDT[0,0]**2))) + (RoDT[1,2]**2) + (RoDT[0,2]**2) + (RoDT[0,1]**2))

    Wk.append(W_full_kin)
    Wk_strain.append(W_strain_kin)

DATA = zip(I, t_av, x_loc, y_loc, tl_xlocations, tl_ylocations, tr_xlocations, tr_ylocations, br_xlocations, br_ylocations, bl_xlocations, bl_ylocations, rot, rot_cum, gam_xy, gam_xy_cum, I1, I2, I3, R_I1, R_I2, R_I3, e_val1, e_vec1x, e_vec1y, e_vec1z, e_val2, e_vec2x, e_vec2y, e_vec2z, e_val3, e_vec3x, e_vec3y, e_vec3z, Wk, Wk_strain, n_oct_unit_shear, lode, Distension, STRESS_I1, STRESS_I2, STRESS_I3, STRESS_e_val1, STRESS_e_vec1x, STRESS_e_vec1y, STRESS_e_vec1z, STRESS_e_val2, STRESS_e_vec2x, STRESS_e_vec2y, STRESS_e_vec2z, STRESS_e_val3, STRESS_e_vec3x, STRESS_e_vec3y, STRESS_e_vec3z, pre, temp, dist)

np.savetxt('{}/{}/output.txt'.format(namemodel, dirname), np.array(DATA), delimiter=' ', header="TIME x y tl_x tl_y tr_x tr_y br_x br_y bl_x bl_y rot rot_cum gam_xy gam_xy_cum e_I1 e_I2 e_I3 e_R_I1 e_R_I2 e_R_I3 e_e_val1 e_e_vec1x e_e_vec1y e_e_vec1z e_e_val2 e_e_vec2x e_e_vec2y e_e_vec2z e_e_val3 e_e_vec3x e_e_vec3y e_e_vec3z Wk Wk_strain n_oct_unit_shear lode Distension S_I1 S_I2 S_I3 S_e_val1 S_e_vec1x S_e_vec1y S_e_vec1z S_e_val2 S_e_vec2x S_e_vec2y S_e_vec2z S_e_val3 S_e_vec3x S_e_vec3y S_e_vec3z P T Alpha")

print "SCRIPT IS COMPLETE - STRESS AND STRAIN DATA ARE NOW OUTPUT TO A .txt FILE, LOCATED IN THE USER-SPECIFIED OUTPUT DIRECTORY"

# ---------------------------------------------------- FINISH ---------------------------------------------------- #
