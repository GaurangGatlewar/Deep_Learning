# Importing the requirements
import pandas as pd
import urllib.request
import os
import requests
import tensorflow as tf
import glob
import csv
import time

# The files containing the images and their labels
'''train = pd.read_csv('train_subset.csv')
test = pd.read_csv('test_subset.csv')
validate = pd.read_csv('validate_subset.csv')
'''
train = pd.read_csv('total_10.csv')

#Major defines
# The images have to be resized to fit the following dimensions
N_INPUTS  = 32*32*3
N_CLASSES = 10
BATCH_SIZE = 64

# Initializing a global counter for images
counter = 0

# Download and save an image locally
def get_image(url,number):
    filename = 'train/' + str(number).zfill(4)+'.jpg'
    urllib.request.urlretrieve(url,filename)
    return (filename)

# Deleting images to clear space
def delete_images(self):
    for file in os.listdir():
        if(file.endswith('.jpg')):
            os.remove(file)

# Download entire batch of images
def download_batch(counter = counter):
    filenames = []
    labels = [0] * BATCH_SIZE
    n = 0
    while(n < BATCH_SIZE):
        try:
            url = train['url'][counter]
            print (url)
            filename = get_image(url,n)
            filenames.append(filename)
            labels[i] = train['landmark_id'][counter]
            print ("Downloaded " + str(counter))
            n += 1
        except:
            print ("Skipped " + str(counter))
        counter += 1
    print ("The images were downloaded")
    return (filenames,labels)

# Reset
tf.reset_default_graph()

# Create input dataset and generate batches of data
def _parse_function():
    for image in filenames:
        tf.image.resize_image_with_crop_or_pad(image,32,32)
    img_string = tf.read_file(filenames)
    img_decoded = tf.image.decode_png(img_string, channels=3)

    #Reserved for image augmentation
    float_image = tf.image.per_image_standardization(img_decoded)
    
#     float_image = tf.image.random_flip_left_right(float_image, seed = 1)
#     float_image = tf.contrib.image.rotate(float_image, np.random.randint(-15,16)) 
#     float_image = tf.image.random_brightness(float_image, 0.3)
#     float_image = tf.image.random_contrast(float_image, 0.1, 0.9)
#     float_image = tf.image.random_saturation(float_image, 0.1, 0.9)
    
    img_decoded = tf.reshape(float_image , [-1])  #flatten
    return img_decoded, labels

# Getting the training batches ready
def get_train_data():
    filenames,labels = download_batch()
    train = tf.data.Dataset.from_tensor_slices(filenames, labels)
    train = train.map(_parse_function, num_parallel_calls = 3)
    delete_images()
    train_iterator = train.make_initializable_iterator()
    train_batch = train_iterator.get_next()  #next_element is tensor of (img_train, y_train)
    return (train_batch)

# Getting the validation batches ready
def get_validation_data(self):
    filenames,labels = download_batch()
    val = tf.data.Dataset.from_tensor_slices(filenames, labels)   #validation batch
    val = val.map(_parse_function, num_parallel_calls = 3)
    delete_images()
    val_iterator = val.make_initializable_iterator()
    validation_batch = val_iterator.get_next()
    return (validation_batch)

# Parsing for test data
def _parse_function_test(sefl):
    img_string = tf.read_file(filenames)
    img_decoded = tf.image.decode_png(img_string, channels=3)
    
    #Reserved for image augmentation
    float_image = tf.image.per_image_standardization(img_decoded)
    img_decoded = tf.reshape(float_image , [-1])  #flatten
    
    return img_decoded

download_test_images(self)
filenames_test = [file_t for file_t in glob.glob(data_path_test + '*/*')]  
filenames_test.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

#For test data
data_test = tf.data.Dataset.from_tensor_slices(filenames_test)
data_test = data_test.map(_parse_function_test)
delete_images()
data_test = data_test.batch(5000)
test_iterator = data_test.make_initializable_iterator()
X_t = test_iterator.get_next()

# The CNN architecture
def conv_model(X, N_CLASSES, reuse, is_training):
    
    with tf.variable_scope('Conv', reuse = reuse): #to reuse weights and biases for testing
        
        input_layer = tf.reshape(X, [-1, 32,32,3])

        conv0 = tf.layers.conv2d(
          inputs=input_layer,
          filters=32,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu)
        
        batchnorm0 = tf.layers.batch_normalization(conv0)
        
        pool0 = tf.layers.max_pooling2d(
            inputs = batchnorm0,
            pool_size = 2,
            strides = 2,
            padding = "same",
            data_format='channels_last')
        
        conv1 = tf.layers.conv2d(
          inputs=conv0,
          filters=64,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu) 
       
        pool1 = tf.layers.max_pooling2d(
            inputs = conv1,
            pool_size = 2,
            strides = 2,
            padding = "same",
            data_format='channels_last')
        
        batchnorm3 = tf.layers.batch_normalization(pool1)
        
        dropout0 = tf.layers.dropout(
            inputs = batchnorm3,
            rate=0.30,
            training = is_training)
        
        conv2 = tf.layers.conv2d(
            inputs=dropout0,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)

        batchnorm1 = tf.layers.batch_normalization(conv2)
        
        pool2 = tf.layers.max_pooling2d(
            inputs = batchnorm1,
            pool_size = 2,
            strides = 2,
            padding="same",
            data_format='channels_last')
       
        conv3 = tf.layers.conv2d(
          inputs=pool2,
          filters=256,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu)
        
        batchnorm2 = tf.layers.batch_normalization(conv3)
        
        pool3 = tf.layers.max_pooling2d(
            inputs = batchnorm2,
            pool_size = 2,
            strides = 2,
            padding="same",
            data_format='channels_last')
        
        dropout1 = tf.layers.dropout(
            inputs = pool3,
            rate=0.25,
            training = is_training)

        flatten = tf.layers.flatten(dropout1)

        dense1 = tf.layers.dense(
            inputs = flatten,
            units = 1024,
            activation= tf.nn.relu)

#         dropout2 = tf.layers.dropout(
#             inputs = dense1,
#             rate=0.25,
#             training = is_training)

        dense2 = tf.layers.dense(
            inputs = dense1,
            units = 1024,
            activation= tf.nn.relu)

        dropout2 = tf.layers.dropout(
            inputs = dense2,
            rate=0.35,
            training = is_training)
        
        dense4 = tf.layers.dense(
            inputs = dropout2,
            units = N_CLASSES)
        
        if is_training:
            last_layer = dense4     #using sparse cross entropy so no need to apply softmax here
        else:
            last_layer = tf.nn.softmax(dense4)   #for inference

        #FOR FUTURE
    #     predictions = {
    #       # Generate predictions (for PREDICT and EVAL mode)
    #       "classes": tf.argmax(input=logits, axis=1),
    #       # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
    #       # `logging_hook`.
    #       "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    #       }

    #     if mode == tf.estimator.ModeKeys.PREDICT:
    #         return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        return last_layer

# Running tensorflow
# X,y = train_batch
X,y = get_train_data()
valX, valY = get_validation_data()

global_step = tf.Variable(0, dtype=tf.int32, trainable = False, name='global_step')

train_output = conv_model(X, N_CLASSES, reuse = False, is_training = True)

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits (labels = y, logits = train_output))
learning_rate = tf.placeholder(tf.float32)
optimizer = tf.train.AdamOptimizer().minimize(cost, global_step = global_step)

#NOTE: THIS IS VERY INEFFICIENT. IMPROVE THIS BY USING feed_dict

# Evaluate model with train data
test_output = conv_model(X, N_CLASSES, reuse=True, is_training=False)
correct_pred = tf.equal(tf.argmax(test_output, 1, output_type=tf.int32), y)
train_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# # Evaluate model with Validation data
val_test_output = conv_model(valX, N_CLASSES, reuse=True, is_training=False)
val_pred = tf.equal(tf.argmax(val_test_output, 1, output_type=tf.int32), valY)
val_accuracy = tf.reduce_mean(tf.cast(val_pred, tf.float32))

#for test data
test_r = conv_model(X_t, N_CLASSES, reuse=True, is_training=False)
test_pred = tf.argmax(test_r, 1 , output_type=tf.int32)


init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)   #initialize variables
    sess.run(train_iterator.initializer)
    
    
    epochCount = 1
        
    while (epochCount < N_EPOCHS):
        startTime = time.time()
        while True:
            try:
                sess.run(optimizer)
                
            except tf.errors.OutOfRangeError:
                sess.run(train_iterator.initializer)
                tr_loss, tr_acc = sess.run([cost, train_accuracy])
                
                sess.run(val_iterator.initializer)
                val_acc = sess.run(val_accuracy)
                print("Epoch {}    Loss: {:,.4f}    Train Accuracy: {:,.2f}    Val Accuracy: {:,.2f}    Time: {:,.2f}"         
                      .format(epochCount, tr_loss, tr_acc, val_acc, time.time() - startTime))
               
                epochCount += 1
                break
                
            except KeyboardInterrupt:   #use this to close the program and save the model as well as a file for predictions on the test set
                print ('\nTraining Interrupted at epoch %d' % epochCount)
                epochCount = N_EPOCHS + 1
                break

    print ('Done Training')
    #Save the model
    save_path = saver.save(sess, 'checkpoints/', global_step = global_step )
    print ('Model saved at %s' % save_path)  
    
    sess.run(test_iterator.initializer)
    test_predictions = sess.run(test_pred)
    predictions_csv = open('model4_1_'+str(epochCount)+'.csv', 'w')
    header = ['id','label']
    with predictions_csv:
        writer = csv.writer(predictions_csv)
        writer.writerow((header[0], header[1]))
        for count, row in enumerate(range(test_predictions.shape[0])):
            writer.writerow((count, id2label[test_predictions[count]]))
        print("Writing complete")