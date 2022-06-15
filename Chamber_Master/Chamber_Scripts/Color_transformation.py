import pickle
import numpy as np
import numpy.matlib as matlib
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color


# Input:
# - data: a np array with dimensions (n_samples, {optional
#   dimension: n_times}, n_color_coordinates=3) (e.g., a direct output of
#   'rgb_extractor()' or 'rgb_extractor_Xrite_CC()')
# - from_space: choose either 'RGB' or 'Lab'
# - to_space: choose either 'RGB' or 'Lab'
# Output:
# - converted: a np array with the same dimensions than in the input
def convert_color_space(data, from_space, to_space):
    # We need the code to work for inputs containing the optional dimension
    # n_times (i.e., many time points) and for inputs containing only one time
    # point.
    n_d = data.ndim
    if n_d == 2:
        data = np.expand_dims(data, 1)
    elif n_d != 3:
        raise Exception('Faulty number of dimensions in the input!')
    if (from_space == 'RGB') and (to_space == 'Lab'):
        # Values from rgb_extractor() are [0,255] so let's normalize.
        data = data/255
        # Transform to color objects (either sRGBColor or LabColor).
        data_objects = np.vectorize(lambda x,y,z: sRGBColor(x,y,z))(
            data[:,:,0], data[:,:,1], data[:,:,2])
        # Target color space
        color_space = matlib.repmat(LabColor, *data_objects.shape)
        # Transform from original space to new space 
        converted_objects = np.vectorize(lambda x,y: convert_color(x,y))(
            data_objects, color_space)
        # We got a matrix of color objects. Let's transform to a 3D matrix of floats.
        converted = np.transpose(np.vectorize(lambda x: (x.lab_l, x.lab_a, x.lab_b))(
            converted_objects), (1,2,0))
        # We want output to be in the same shape than the input.
    elif (from_space == 'Lab') and (to_space == 'RGB'):
        data_objects = np.vectorize(lambda x,y,z: LabColor(x,y,z))(
            data[:,:,0], data[:,:,1], data[:,:,2])
        color_space = matlib.repmat(sRGBColor, *data_objects.shape)
        converted_objects = np.vectorize(lambda x,y: convert_color(x,y))(
            data_objects, color_space)
        converted = np.transpose(np.vectorize(lambda x: (x.rgb_r, x.rgb_g, x.rgb_b))(
            converted_objects), (1,2,0))
        # Colormath library interprets rgb in [0,1] and we want [0,255] so let's
        # normalize to [0,255].
        converted = converted*255
    else:
        raise Exception('The given input space conversions have not been implemented.')
    if n_d == 2:
        converted = np.squeeze(converted)
    return (converted)


# Input:
# - data: a np array with dimensions (n_samples, {optional
#   dimension: n_times}, n_color_coordinates=3)
def haverford_color_calibration(sample_rgb):

    chamber_data_path = "C:/Users/Rod/Documents/Chamber_Test_Folder/"
    expt_run = "Color_Calibration_5_6_22" ### Enter in the experiment run you'd like to sync data for.
    
    xrite_array = pickle.load(open(chamber_data_path + expt_run + '/xrite_rgb.pkl', 'rb'))
    
    # Converting our xrite_array into the correct order for the color calibration code.
    # The color calibration code expects the xrite patchs to be in order from left to right,
    # Then down to the next, and then left to right.
    xrite_rgb_ordered = np.concatenate((xrite_array[0,0:7,:],xrite_array[1,0:7,:],xrite_array[2,0:7,:],xrite_array[3,0:7,:]), axis=0)
    
    
    ### Testing from color calibration script
    
    reference_CC_lab =np.array([[37.54,14.37,14.92],[62.73,35.83,56.5],[28.37,15.42,-49.8],
                                [95.19,-1.03,2.93],[64.66,19.27,17.5],[39.43,10.75,-45.17],
                                [54.38,-39.72,32.27],[81.29,-0.57,0.44],[49.32,-3.82,-22.54],
                                [50.57,48.64,16.67],[42.43,51.05,28.62],[66.89,-0.75,-0.06],
                                [43.46,-12.74,22.72],[30.1,22.54,-20.87],[81.8,2.67,80.41],
                                [50.76,-0.13,0.14],[54.94,9.61,-24.79],[71.77,-24.13,58.19],
                                [50.63,51.28,-14.12],[35.63,-0.46,-0.48],[70.48,-32.26,-0.37],
                                [71.51,18.24,67.37],[49.57,-29.71,-28.32],[20.64,0.07,-0.46]])
    # Reference data is in different order (from upper left to lower left, upper
    # 2nd left to lower 2nd left...). This is the correct order (The correct order 
    # for them is left to right, as you'd read):
    order = list(range(0,21,4)) + list(range(1,22,4)) + list(range(2,23,4)) + list(range(3,24,4))
    reference_CC_lab = reference_CC_lab[order]
    
    # For debugging purposes, let's convert to RGB.
    reference_CC_rgb = convert_color_space(reference_CC_lab, 'Lab', 'RGB')
    
    
    
    
    # Let's extract the rgb colors from our Xrite color passport picture.
    
    # Convert from RGB to Lab color space.
    CC_lab = convert_color_space(xrite_rgb_ordered , 'RGB', 'Lab')
    sample_lab = convert_color_space(sample_rgb, 'RGB', 'Lab')
    
    
    ###########################
    # Color calibration starts.
    
    # Number of color patches in the color chart.
    N_patches = CC_lab.shape[0]
    
    # Let's create the weight matrix for color calibration using 3D thin plate
    # spline.
    
    # Data points of our color chart in the original space.
    P = np.concatenate((np.ones((N_patches,1)), CC_lab), axis=1)
    # Data points of our color chart in the transformed space.
    V = reference_CC_lab
    # Shape distortion matrix, K
    K = np.zeros((N_patches,N_patches))
    for i in range(N_patches):
        for j in range(N_patches):
            if i != j:
                r_ij = np.sqrt((P[j,0+1]-P[i,0+1])**2 +
                               (P[j,1+1]-P[i,1+1])**2 +
                               (P[j,2+1]-P[i,2+1])**2)
                U_ij = 2* (r_ij**2)* np.log(r_ij + 10**(-20))
                K[i,j] = U_ij
    # Linear and non-linear weights WA:
    numerator = np.concatenate((V, np.zeros((4,3))), axis=0)
    denominator = np.concatenate((K,P), axis=1)
    denominator = np.concatenate((denominator,
                                  np.concatenate((np.transpose(P),
                                                  np.zeros((4,4))),axis=1)), axis=0)
    WA = np.matmul(np.linalg.pinv(denominator), numerator)
    
    # Checking if went ok. We should get the same result than in V (exept for
    # the 4 bottom rows)
    CC_lab_double_transformation = np.matmul(denominator,WA)
    print('Color chart patches in reference Lab:', reference_CC_lab,
          'Color chart patches transformed to color calibrated space and back - this should be the same than above apart from the last 4 rows',
          CC_lab_double_transformation, 'subtracted: ', reference_CC_lab-CC_lab_double_transformation[0:-4,:])
    # --> Went ok!
    
    # Let's perform color calibration for the sample points!
    N_samples = sample_lab.shape[0]
    N_times = sample_lab.shape[1]
    sample_lab_cal = np.zeros((N_samples,N_times+4,3))
    # We are recalculating P and K for each sample, but using the WA calculated above.
    for s in range(N_samples):
        # Data points of color chart in the original space.
        P_new = np.concatenate((np.ones((N_times,1)), sample_lab[s,:,:]), axis=1)
        K_new = np.zeros((N_times,N_patches))
        # For each time point (i.e., picture):
        for i in range(N_times):
            # For each color patch in Xrite color chart:
            for j in range(N_patches):
                #if i != j:
                r_ij = np.sqrt((P_new[i,0+1]-P[j,0+1])**2 + (P_new[i,1+1]-P[j,1+1])**2 + (P_new[i,2+1]-P[j,2+1])**2)
                U_ij = 2* (r_ij**2)* np.log(r_ij + 10**(-20))
                K_new[i,j] = U_ij
        #sample_lab_cal[s,:,:] = np.matmul(np.concatenate((K_new,P_new),axis=1), WA)
        dennom = np.concatenate((K_new,P_new),axis=1)
        denden = np.concatenate((np.transpose(P), np.zeros((4,4))), axis=1)
        sample_lab_cal[s,:,:] = np.matmul(np.concatenate((dennom, denden), axis=0), WA)
    # Remove zeros, i.e., the last four rows from the third dimension.
    sample_lab_cal = sample_lab_cal[:,0:-4,:]
    ################################
    # Color calibration is done now.
    
    # Let's transform back to rgb.
    sample_rgb_cal = convert_color_space(sample_lab_cal, 'Lab', 'RGB')
    


    # Let's return both lab and rgb calibrated values.
    return (sample_rgb_cal, sample_lab_cal)