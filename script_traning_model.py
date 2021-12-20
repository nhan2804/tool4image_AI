import os
import tensorflow as tf
import numpy as np
import math
import timeit
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random

# Create directories
dataset_home = 'train/'
subdirs = ['train/', 'val/', 'test/']
for subdir in subdirs:
    # Create label subdirectories
    labeldirs = ['dogs/', 'cats/']
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        makedirs(newdir, exist_ok=True)

# Copy dataset into folders
dog_files = []
cat_files = []

# Define ratio of pictures to use for testing
# Copy training dataset images into subdirectories
src_directory = dataset_home
for file in listdir(src_directory):
    if file.startswith('cat'):
        cat_files.append(file)
    elif file.startswith('dog'):
        dog_files.append(file)


def train_validate_test_split(data, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(np.arange(len(data)))
    m = len(data)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end

    train = np.array(data)[perm[:train_end]].copy()
    validate = np.array(data)[perm[train_end:validate_end]].copy()
    test = np.array(data)[perm[validate_end:]].copy()
    return train, validate, test


train_cats, val_cats, test_cats = train_validate_test_split(cat_files, seed=42, train_percent=.6, validate_percent=.2)
train_dogs, val_dogs, test_dogs = train_validate_test_split(dog_files, seed=42, train_percent=.6, validate_percent=.2)

list(map(lambda file: copyfile(src_directory + '/' + file, dataset_home + 'train/cats/' + file), train_cats))
list(map(lambda file: copyfile(src_directory + '/' + file, dataset_home + 'train/dogs/' + file), train_dogs))
list(map(lambda file: copyfile(src_directory + '/' + file, dataset_home + 'val/cats/' + file), val_cats))
list(map(lambda file: copyfile(src_directory + '/' + file, dataset_home + 'val/dogs/' + file), val_dogs))
list(map(lambda file: copyfile(src_directory + '/' + file, dataset_home + 'test/cats/' + file), test_cats))
list(map(lambda file: copyfile(src_directory + '/' + file, dataset_home + 'test/dogs/' + file), test_dogs))

print("Done!")
