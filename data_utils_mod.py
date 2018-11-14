import os
import numpy as np
import cv2
from scipy.ndimage import zoom

import keras
from keras import backend as K
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator

class DataGenerator(ImageDataGenerator):
    """
    Generate minibatches of images and labels with real-time augmentation.
    The only function that changes w.r.t. parent class is the flow that
    generates data. This function needed in fact adaptation for different
    directory structure and labels. All the remaining functions remain
    unchanged.
    """
    # SE REESCRIBE ESTA FUNCION, TODAS LAS DEMAS SE HEREDAN DE IMAGEDATAGENERATOR DE KERAS
    def flow_from_directory(self, directory, num_classes, target_size=(120,224,224),
                            img_mode='grayscale', batch_size=32, shuffle=True,
                            seed=None, follow_links=False):
        return DirectoryIterator(
                directory, num_classes, self, target_size=target_size, img_mode=img_mode,
                batch_size=batch_size, shuffle=shuffle, seed=seed,
                follow_links=follow_links)


class DirectoryIterator(Iterator):
    """
    Class for managing data loading of images and labels
    We assume that the folder structure is:
    root_folder/
        class_0/
            repetition_0/
                frame_00000000.png
                frame_00000001.png
                .
                .
                frame_00999999.png
                trajectories.csv
            repetition_1/
            .
            .
            repetition_m/
        class_1/
        .
        .
        class_n/

    # Arguments
       directory: Path to the root directory to read data from.
       num_classes: Output dimension (number of classes).
       image_data_generator: Image Generator.
       target_size: tuple of integers, dimensions to resize input images to.
       img_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
       batch_size: The desired batch size
       shuffle: Whether to shuffle data or not
       seed : numpy seed to shuffle data
       follow_links: Bool, whether to follow symbolic links or not

    # TODO: Add functionality to save images to have a look at the augmentation
    """
    def __init__(self, directory, num_classes, image_data_generator,
            target_size=(120,224,224), img_mode = 'grayscale',
            batch_size=32, shuffle=True, seed=None, follow_links=False):
        self.directory = os.path.realpath(directory)
        self.image_data_generator = image_data_generator
        self.target_size = target_size
        self.follow_links = follow_links
        
        # Initialize image mode
        if img_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', img_mode,
                             '; expected "rgb" or "grayscale".')
        self.img_mode = img_mode
        #if self.img_mode == 'rgb':
        #   self.image_shape = (self.target_size[1], self.target_size[2])
        #else:
        #   self.image_shape = self.target_size + (1,)'''
        
        # Initialize number of classes
        self.num_classes = num_classes

        # Number of samples in dataset
        self.samples = 0

        # Filenames of all samples/repetitions in dataset. 
        self.filenames = [] 
        
        # Labels (ground truth) of all samples/repetitions in dataset
        self.ground_truth = []
        
        # Number of samples per class
        self.samples_per_class = []
        
        # First count how many gestures there are
        gestures = []
        for subdir in sorted(os.listdir(directory)):
            if os.path.isdir(os.path.join(directory, subdir)):
                gestures.append(subdir)  

        # Decode dataset structure.
        # All the filenames and ground truths are loaded in memory from the
        # begining.
        # Images instead, will be loaded iteratively as the same time the
        # training process needs a new batch.
        for gesture_id, gesture in enumerate(gestures):
            print(gesture_id)
            gesture_path = os.path.join(directory, gesture)
            self.samples_per_class.append(len(os.listdir(gesture_path)))
            for rep_id, subdir in enumerate(sorted(os.listdir(gesture_path))):
                rep_path = os.path.join(gesture_path, subdir)
                if os.path.isdir(rep_path):
                    self.filenames.append(os.path.abspath(rep_path))
                    self.samples +=1
                    
                    # Generate associated ground truth
            labels = gesture_id*np.ones((self.samples_per_class[gesture_id]), dtype=np.int)
            self.ground_truth = np.concatenate((self.ground_truth,labels), axis=0)
                    
        
        # Check if dataset is empty            
        if self.samples == 0:
            raise IOError("Did not find any data")

        # Conversion of list into array
        self.ground_truth = np.array(self.ground_truth, dtype= K.floatx())

        print('Found {} samples belonging to {} classes.'.format(
                self.samples, self.num_classes))

        super(DirectoryIterator, self).__init__(self.samples,
                batch_size, shuffle, seed)


    def _recursive_list(self, subpath):
        return sorted(os.walk(subpath, followlinks=self.follow_links),
                key=lambda tpl: tpl[0])


    def next(self):
        """
        Public function to fetch next batch
        # Returns
            The next batch of images and commands.
        """
        with self.lock:
            index_array = next(self.index_generator)

        return self._get_batches_of_transformed_samples(index_array)


    def _get_batches_of_transformed_samples(self, index_array):
        """
        Public function to fetch next batch.
        Image transformation is not under thread lock, so it can be done in
        parallel
        # Returns
            The next batch of images and categorical labels.
        """
        current_batch_size = index_array.shape[0]
                    
        # Initialize batch of images
        batch_x = np.zeros((current_batch_size,) + self.target_size,
                dtype=K.floatx())
        
        # Initialize batch of ground truth
        batch_y = np.zeros((current_batch_size, self.num_classes,),
                                 dtype=K.floatx())
                                 
        grayscale = self.img_mode == 'grayscale'

        # Build batch of block-image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            x = load_imgs(os.path.join(self.directory, fname),
                                   grayscale=grayscale,
                                   target_size=self.target_size)
            # Data augmentation
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            
            batch_x[i] = x

        # Build batch of labels
        batch_y = np.array(self.ground_truth[index_array], dtype=K.floatx())
        batch_y = keras.utils.to_categorical(batch_y, num_classes=self.num_classes)

        return batch_x, batch_y
    
def adjust(data, size):
    factors = (size[0]/data.shape[0], size[1]/data.shape[1], size[2]/data.shape[2])
    new_array = zoom(data, factors)
    
    return new_array

def load_imgs(path, grayscale=False, target_size=None):
    """
    Load a block of images.
    # Arguments
        path: Path to image files.
        grayscale: Boolean, wether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or ints `(num_imgs, img_height, img_width)`.
    # Returns
        Block of images as numpy array.
    """
    
    # Read input images
    paths = [os.path.join(path, x) for x in os.listdir(path) if x.endswith('.png')]
    
    imgs = []
    
    for img_path in paths:
        img = cv2.imread(img_path)
        
        if grayscale:
            if len(img.shape) != 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        if grayscale:
            img = img.reshape((img.shape[0], img.shape[1]))
        imgs.append(img)
        
    if target_size:
        imgs = adjust(np.asarray(imgs, dtype=np.float32), target_size)
        
    return imgs
