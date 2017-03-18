# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'train.p'
validation_file = 'valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import random
import numpy as np

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
index = random.randint(0, len(X_train))
image = X_train[index]
image_shape = image.shape

# TODO: How many unique classes/labels there are in the dataset.
classes, classes_counts = np.unique(y_train, return_counts=True)
n_classes = len(classes)

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import csv 
# Visualizations will be shown in the notebook.
#%matplotlib inline

class_labels = [] 
with open('signnames.csv') as csvfile:
    signnames_reader = csv.DictReader(csvfile, delimiter=',')
    for row in signnames_reader:
        class_labels.append(row['ClassId'] + ": " + row['SignName'])
            
def graph_image_class_counts(classes, classes_counts, class_labels):
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(20,15))
    plt.margins(y=0) 
            
    ax.barh(classes, classes_counts, align='center', color='green')
    ax.set_yticks(classes)
    ax.set_yticklabels(class_labels)
    # Set the tick labels font
    for label in (ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(10)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Count')
    ax.set_title('Traffic Sign Counts')
    plt.show()
    
graph_image_class_counts(classes, classes_counts, class_labels)

#View a sample from the dataset.
index = random.randint(0, n_train)
sample_image = X_train[index]
plt.figure(figsize=(1,1))
plt.imshow(sample_image)
plt.show()
print(class_labels[y_train[index]])



### Preprocess the data here. Preprocessing steps could include normalization, converting to grayscale, etc.
### Feel free to use as many code cells as needed.
import cv2

### Define functions used to randomly augment training data by randomly rotating and translating existing images

def rotate_image(image_data, rotation_degrees):
    rows, cols = image_data.shape[:2]
    center = (cols / 2, rows / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_degrees, 1.0)
    return cv2.warpAffine(image_data, rotation_matrix, (cols, rows))

def randomly_rotate_image(image_data):
    random_rotation_degrees = random.randint(-15, 15)
    return rotate_image(image_data, random_rotation_degrees)

plt.figure(figsize=(1,1))
rotated_image = randomly_rotate_image(sample_image)
plt.imshow(rotated_image)
plt.show()
print("Randomly Rotated")

def translate_image(image_data, horizontal_translation_distance, vertical_translation_distance):   
    rows, cols = image_data.shape[:2]
    translation_matrix = np.float32([ [1,0,horizontal_translation_distance], [0,1,vertical_translation_distance] ])
    return cv2.warpAffine(image_data, translation_matrix, (cols, rows))

def randomly_translate_image(image_data):
    random_h_distance = random.randint(-5, 5)
    random_v_distance = random.randint(-5, 5)
    return translate_image(image_data, random_h_distance, random_v_distance)

plt.figure(figsize=(1,1))
rotated_image = randomly_translate_image(rotated_image)
plt.imshow(rotated_image)
plt.show()
print("Randomly Translated")


### Make sure each traffic sign has at least min_class_count samples
print("Augmenting Data...")
min_class_count = 2000
for i,class_count in enumerate(classes_counts):
    if (class_count < min_class_count):
        curr_class = classes[i]
        class_indexes = [j for j,image_class in enumerate(y_train) if image_class == curr_class]
        new_image_count = 0
        while (new_image_count < min_class_count - class_count):
            random_index = class_indexes[random.randint(0, len(class_indexes)-1)]
            X_train = np.append(X_train, [randomly_translate_image(randomly_rotate_image(X_train[random_index]))], axis=0) 
            y_train = np.append(y_train, [curr_class], axis=0)
            new_image_count += 1


n_train = len(X_train)
classes, classes_counts = np.unique(y_train, return_counts=True)
print("Number of augmented training examples =", n_train)
graph_image_class_counts(classes, classes_counts, class_labels)


### Normalize image data
is_data_normalized = False

def normalize_color(image_data):
    return (image_data - 128)/128.0

if not is_data_normalized:
    X_train = normalize_color(X_train)
    X_valid = normalize_color(X_valid)
    X_test = normalize_color(X_test)
    is_data_normalized = True

normalized_image = X_train[index]
plt.figure(figsize=(1,1))
plt.imshow(normalized_image)
plt.show()
print("Normalized")


### Shuffle training data.
from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)


### Define your architecture here.
### Feel free to use as many code cells as needed.

import tensorflow as tf

EPOCHS = 20
BATCH_SIZE = 128

from tensorflow.contrib.layers import flatten

def ConvNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.2
    
    layer_width = {
        'layer_1': 6,
        'layer_2': 16,
        'fully_connected_1': 120,
        'fully_connected_2': 84,
        'output': 43
    }
    
    weights = {
        'layer_1': tf.Variable(tf.truncated_normal(
            shape = [5, 5, 3, layer_width['layer_1']], mean = mu, stddev = sigma)),
        'layer_2': tf.Variable(tf.truncated_normal(
            shape = [5, 5, layer_width['layer_1'], layer_width['layer_2']], mean = mu, stddev = sigma)),
        'fully_connected_1': tf.Variable(tf.truncated_normal(
            shape = [5*5*layer_width['layer_2'], layer_width['fully_connected_1']], mean = mu, stddev = sigma)),
        'fully_connected_2': tf.Variable(tf.truncated_normal(
            shape = [layer_width['fully_connected_1'], layer_width['fully_connected_2']], mean = mu, stddev = sigma)),
        'output': tf.Variable(tf.truncated_normal(
            shape = [layer_width['fully_connected_2'], layer_width['output']], mean = mu, stddev = sigma))
    }
    
    biases = {
        'layer_1': tf.Variable(tf.zeros(layer_width['layer_1'])),
        'layer_2': tf.Variable(tf.zeros(layer_width['layer_2'])),
        'fully_connected_1': tf.Variable(tf.zeros(layer_width['fully_connected_1'])),
        'fully_connected_2': tf.Variable(tf.zeros(layer_width['fully_connected_2'])),
        'output': tf.Variable(tf.zeros(layer_width['output']))
    }
    
    filter_strides = [1, 1, 1, 1]
    padding = 'VALID'
    ksize = [1, 2, 2, 1]
    pooling_strides = [1, 2, 2, 1]
    convolutional_keep_prob = 0.7
    fully_connected_keep_prob = 0.5
    
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    layer1 = tf.nn.conv2d(x, weights['layer_1'], filter_strides, padding) + biases['layer_1']
    layer1 = tf.nn.relu(layer1)
    layer1 = tf.nn.dropout(layer1, convolutional_keep_prob)
    # Pooling. Input = 28x28x6. Output = 14x14x6.
    layer1 = tf.nn.max_pool(layer1, ksize, pooling_strides, padding)

    # Layer 2: Convolutional. Output = 10x10x16.
    layer2 = tf.nn.conv2d(layer1, weights['layer_2'], filter_strides, padding) + biases['layer_2']
    layer2 = tf.nn.relu(layer2)
    layer2 = tf.nn.dropout(layer2, convolutional_keep_prob)
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    layer2 = tf.nn.max_pool(layer2, ksize, pooling_strides, padding)

    # Flatten. Input = 5x5x16. Output = 400.
    layer2_flat = tf.reshape(layer2, [-1, 5*5*layer_width['layer_2']])
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fully_connected1 = tf.add(tf.matmul(layer2_flat, weights['fully_connected_1']), biases['fully_connected_1'])
    fully_connected1 = tf.nn.relu(fully_connected1)
    fully_connected1 = tf.nn.dropout(fully_connected1, fully_connected_keep_prob)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fully_connected2= tf.add(tf.matmul(fully_connected1, weights['fully_connected_2']), biases['fully_connected_2'])
    fully_connected2 = tf.nn.relu(fully_connected2)
    fully_connected2 = tf.nn.dropout(fully_connected2, fully_connected_keep_prob)

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = tf.add(tf.matmul(fully_connected2, weights['output']), biases['output'])
    
    return logits


### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

# x is a placeholder for a batch of input images. y is a placeholder for a batch of output labels.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)


# Create a training pipeline that uses the model to classify traffic sign data
rate = 0.001

logits = ConvNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


# Evaluate how well the loss and accuracy of the model for a given dataset.
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


### Run the training data through the training pipeline to train the model.
### Before each epoch, shuffle the training set.
### After each epoch, measure the loss and accuracy of the validation set.
### Save the model after training.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './traffic_sign_convnet')
    print("Model saved")
    