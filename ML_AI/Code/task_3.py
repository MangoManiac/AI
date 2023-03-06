import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *

# File that contains the data
data_npy_file = 'data/PB_data.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full

X_full[:, 0] = f1
X_full[:, 1] = f2

########################################/
X_full = X_full.astype(np.float32)

# number of GMM components
k = 6 # change between 3 and 6

#########################################
# Write your code here

# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full

X_phonemes_1_2 = X_full[np.logical_or(phoneme_id ==1, phoneme_id ==2), :]



########################################/

# Plot array containing the chosen phonemes

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme 1 & 2'
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 & 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phonemes_1_2.png')
plt.savefig(plot_filename)


#########################################
# Write your code here
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
# Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar variable named "accuracy"

# Phoneme 1
p1_model = "data/GMM_params_phoneme_{:02}_k_{:02}.npy".format(1, k)
param_p1 = np.load(p1_model, allow_pickle=True).item()
copy = X_phonemes_1_2.copy()
Z_p1 = get_predictions(param_p1['mu'], param_p1['s'], param_p1['p'], copy)

p1_pred = Z_p1.sum(axis=1)

# phenome 2
p2_model = "data/GMM_params_phoneme_{:02}_k_{:02}.npy".format(2, k)
param_p2 = np.load(p2_model, allow_pickle=True).item()
copy = X_phonemes_1_2.copy()
Z_p2 = get_predictions(param_p2['mu'], param_p2['s'], param_p2['p'], copy)

p2_pred = Z_p2.sum(axis=1)


# Calculate Accuracy

preds = np.ones(len(copy)) * 2
preds[p1_pred >= p2_pred] = 1
labels = phoneme_id[np.logical_or(phoneme_id ==1, phoneme_id ==2)]
accuracy = np.sum(preds == labels) / copy.shape[0] * 100


########################################/

print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy))

################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()