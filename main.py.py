################################################################### IMPORTATIONS ###################################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random


################################################################### FUNCTIONS ###################################################################

#### Take a part of an image
def create_submatrix(matrix, center, submatrix_length, submatrix_width=None):
    length = round(submatrix_length/2)
    if submatrix_width == None:
        submatrix_width = submatrix_length
    width = round(submatrix_width/2)
    y = center[0]
    x = center[1]
    submatrix = matrix[int(x-length):int(x+length), int(y-width):int(y+width)]
    return submatrix

#### Create a histogram for each color
def histogrames_rgb(observ):

    total_pixels = observ.shape[0] * observ.shape[1]

    histogram_r = np.histogram(observ[:, :, 0], bins=256, range=(0, 256))[0] / total_pixels
    histogram_g = np.histogram(observ[:, :, 1], bins=256, range=(0, 256))[0] / total_pixels
    histogram_b = np.histogram(observ[:, :, 2], bins=256, range=(0, 256))[0] / total_pixels
    return histogram_r, histogram_g, histogram_b

#### Create n points around the center
def intialize_center(center, n):
    centers = []
    for k in range(n):
        x, y = np.random.normal(center[0], 15),np.random.normal(center[1], 15)
        centers.append([x, y])
    return centers

#### Initialize particule using the centers with equal weigth
def create_particules(centers):
    weighted_centers = []
    n = len(centers)
    for k in range(n):
        x, w = centers[k], 1/n
        weighted_centers.append((x, w))
    return weighted_centers

#### Create noise
def create_noise():
    return np.random.multivariate_normal(mean=[0, 0], cov=[[5, 0], [0, 5]])

#### "Sends" the particles around the previous position
def new_particules(particules):
    new_particules = particules
    for i in range(len(particules)):
        x, w = particules[i]
        new_particules[i] = (x + create_noise(), w)
    return new_particules

#### Calcul the distance between two histograms
def histogram_distance(hist1, hist2):
    distance = np.sqrt(1 - np.sum(np.sqrt(hist1 * hist2)))
    return distance

#### Calcul likelihood using histogram distance and return updated particles
def likelihood_function(particule_observation, href, particule, lambd):
    x, w = particule
    histogram_r, histogram_g, histogram_b = histogrames_rgb(particule_observation)
    distance = (histogram_distance(href[0], histogram_r) + histogram_distance(href[1], histogram_g) + histogram_distance(href[2], histogram_b))
    likelihood = np.exp(-lambd * distance**2)
    w2 = w*likelihood
    particule = (x, w2)
    return likelihood, particule

#### Resemple the particles
def systematic_resample(particules):

    N = len(particules)
    weights = np.zeros(N)

    for k in range(N):
        x, w = particules[k]
        weights[k] = w

    positions = (random() + np.arange(N)) / N
    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


################################################################### MAIN FUNCTIONS ###################################################################

#### Return the frames of the tracking of one object
def particle_filter(cap, observation, initial_center, num_particles, lambd, nb_frames):

    length_observ = len(observation)
    width_observ = len(observation[0])

    # Initialisation
    particules_centers = intialize_center(initial_center, num_particles)
    particules = create_particules(particules_centers)
    reference_histogram_r, reference_histogram_g, reference_histogram_b = histogrames_rgb(observation)
    href = [reference_histogram_r, reference_histogram_g, reference_histogram_b]

    # Process each frame in the video sequence
    for k in range(nb_frames):
        
        # Take frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        res, frame = cap.read()

        # MAJ particule motion
        particules = new_particules(particules)

        for i in range(len(particules)):
            
            # Infos on the particule
            x, w = particules[i]
            particule_observation = create_submatrix(frame, x, length_observ, width_observ)

            # Calculate likelihood for each particle
            likelihood, weighted_particule = likelihood_function(particule_observation, href, particules[i], lambd)
            particules[i] = weighted_particule
            

        # Normalize weights
        total_weight = sum(w for x, w in particules)
        particules = [(x, w / total_weight) for x, w in particules]

        # Resample particles
        indexes = systematic_resample(particules)
        particules = [particules[i] for i in indexes]

        
        # Plot
        if k % 10 == 0:
            liste_x = []
            liste_y = []
            for (x, y), w in particules:
                liste_x.append(x)
                liste_y.append(y)
            
            estimated_center = (np.average(liste_x, weights=[w for _, w in particules]), 
                                np.average(liste_y, weights=[w for _, w in particules]))

            plt.figure(figsize=(6, 6))
            plt.imshow(frame)
            plt.scatter(liste_x, liste_y)
            
            # Display bounding box around the estimated center
            plt.plot(estimated_center[0], estimated_center[1], 'yellow', markersize=10) # Point rouge
            plt.gca().add_patch(plt.Rectangle((estimated_center[0] - round((width_observ/2)), estimated_center[1] - round((length_observ/2))), width_observ, length_observ, 
                                            linewidth=2, edgecolor='yellow', facecolor='none')) # Carré jaune
            
            plt.title('Frame ' + str(k))
            plt.show()
    return 


#### Return the frames of the tracking of many objects
def particle_filter_multiple(cap, observations, initials_center, num_particles, lambd, nb_frames):

    observations_len = []
    observations_wid = []
    href_s = []
    particules_objects = []

    for i in range(len(observations)) :

        # Initialisations

        observations_len.append(len(observations[i]))
        observations_wid.append(len(observations[i][0]))

        particules_centers = intialize_center(initials_center[i], num_particles)
        particules = create_particules(particules_centers)
        particules_objects.append(particules)

        reference_histogram_r, reference_histogram_g, reference_histogram_b = histogrames_rgb(observations[i])
        href_s.append([reference_histogram_r, reference_histogram_g, reference_histogram_b])

    # observations_len/wid > taille des observation
    # href_s > histogrammes de refernce
    # particules_objects > particules des objets
    
    # Process each frame in the video sequence
    for k in range(nb_frames):
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        res, frame = cap.read()
        if k % 10 == 0:
            plt.imshow(frame)
        
        for i in range(len(observations)) :

            # MAJ particule motion
            particules = new_particules(particules_objects[i])

            for j in range(len(particules)):
                
                # Infos on the particule
                x, w = particules_objects[i][j]
                particule_observation = create_submatrix(frame, x, observations_len[i], observations_wid[i])

                # Calculate likelihood for each particle
                likelihood, weighted_particule = likelihood_function(particule_observation, href_s[i], particules_objects[i][j], lambd)
                particules_objects[i][j] = weighted_particule
                

            # Normalize weights
            total_weight = sum(w for x, w in particules_objects[i])
            particules_objects[i] = [(x, w / total_weight) for x, w in particules_objects[i]]

            # Resample particles
            indexes = systematic_resample(particules_objects[i])
            particules_objects[i] = [particules_objects[i][l] for l in indexes]

        
            # Plot
            if k % 10 == 0:
                liste_x = []
                liste_y = []
                
                for (x, y), w in particules_objects[i]:
                    liste_x.append(x)
                    liste_y.append(y)
                
                estimated_center = (np.average(liste_x, weights=[w for _, w in particules_objects[i]]), 
                                    np.average(liste_y, weights=[w for _, w in particules_objects[i]]))

                plt.scatter(liste_x, liste_y)
                
                # Display bounding box around the estimated center
                #plt.plot(estimated_center[0], estimated_center[1], 'ro', markersize=10) # Point rouge
                plt.gca().add_patch(plt.Rectangle((estimated_center[0] - round((observations_wid[i]/2)), estimated_center[1] - round((observations_len[i]/2))), observations_wid[i], observations_len[i], 
                                                linewidth=2, edgecolor='yellow', facecolor='none')) # Carré jaune
        if k % 10 == 0:    
            plt.title('Frame ' + str(k))
            plt.show()
    return 


################################################################### VIDEO TRACKING ###################################################################

#### I take the frames of the video
cap = cv2.VideoCapture("C:\\Users\\maeva\\OneDrive\\Documents\\ENSEA\\TP SIA\\TP34 IMAGE\\video sequences\\synthetic\\escrime-4-3.avi")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
res, frame = cap.read()

#### I define all objects I want to track
observe_center = frame[217:262, 299:344,:]
center_center = [321, 239]
observe_pate1 = frame[117:195, 310:330,:]
pate1_center = [320, 156]
observe_pate2 = frame[230:250, 350:440,:]
pate2_center = [395, 240]
observe_pate3 = frame[290:360, 310:330,:]
pate3_center = [320, 335]
observe_pate4 = frame[230:250, 200:280,:]
pate4_center = [240, 240]

#### Track the center only
#particle_filter(cap, observe_center, [323, 242], 70, 10, 150)

#### Track the upper leg only
#particle_filter(cap, observe_pate_1, [330, 167], 70, 10, 150

#### Track all the legs and the center
particle_filter_multiple(cap, [observe_center, observe_pate1, observe_pate2, observe_pate3, observe_pate4], [center_center, pate1_center, pate2_center, pate3_center, pate4_center], 70, 10, 150)
