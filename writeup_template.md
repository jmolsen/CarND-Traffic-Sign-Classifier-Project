#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/visualization.png "Visualization"
[image2]: ./writeup_images/random_image.png "Random Image"
[image3]: ./writeup_images/random_rotated.png "Random Image Rotated"
[image4]: ./writeup_images/random_rotated_translated.png "Random Rotated Image Translated"
[image5]: ./writeup_images/augmented_visualization.png "Augmented Visualization"
[image6]: ./writeup_images/random_image_grayscale.png "Random Image Grayscaling"
[image7]: ./examples/random_noise.jpg "Random Noise"
[image8]: ./examples/placeholder.png "Traffic Sign 1"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/jmolsen/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the standard python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The size of the validation set is 4410
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how many examples of each traffic sign classification exist in the training set.

![Data Visualization][image1]


###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth through eighth code cells of the IPython notebook.

First, in the fourth code cell, I defined some functions to randomly rotate and translate images to be used later to augment the data. I'll discuss that more in the next section.

Next, in the fifth, sixth, and seventh code cells, I handled data augmentation which I'll talk about more in the next section.

I wanted to handle the data augmentation before the image preprocessing so that I could preprocess all the images at once.

Finally, in the eighth code cell, I converted the images to grayscale and ran min-max scaling on them.  In some of my earlier versions I was using the RGB images, but thought that I might be making it too complicated.  I thought that since signs should be recognizable mostly by their shape and symbols and in varying lighting conditions that it would better to try to learn them as normalized grayscale.

Here is an example of a traffic sign image before and after grayscaling.
![Random Image Before Grayscaling][image2]
![Random Image After Grayscaling][image6]


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

First, in the fourth code cell, I defined some functions to randomly rotate and translate images to be used later to augment the data.  I didn't want to rotate or translate them too much, but just enough to get slightly different images.  So, I randomly rotated between -15 and 15 degrees, and I randomly translated both horizontally and vertically between -5 and 5 pixels.

Using a randomly chosen sample image (pictured first), I printed out examples of a random rotation of the sample image and then a random translation of the rotated image.

![Random Image Before Rotatation and Translation][image2]
![Random Image Rotated][image3]
![Random Rotated Image Translated][image4]

In the fifth code cell, I augmented the given training data so that every class had at least 2000 training images.  I decided to augment the data because there was a huge disparity in how many images existed for each class.  Some classes only had a couple hundred images whereas others had a thousand or two.  For the classes with too few examples I was concerned that it might cause those classes to overfit.  
So, I randomly chose images for each class and then applied the random rotation and translation functions from above to create new images to add to the training set.  The total size of the training set after augmentation was 86010.

I printed another bar graph as before, confirming that that each class had met enough augmented images to meet the minimum of 2000.

![Augmented Visualization][image5]

In the sixth code cell, I pickled the augmented data.  And in the seventh code cell, I loaded the pickled augmented data so I didn't have to run the augmentation every time.
In the ninth code cell, I shuffled the training data.

For the validation and testing data I just used what was given.

My final training set had 86010 images. My validation set and test set had 4410 and 12630 images respectively.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the tenth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 normalized grayscale image  			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x9 	|
| RELU					|												|
| Dropout				| Using convolutional keep probablity 			|
| Max pooling	      	| 2x2 stride,  outputs 14x14x9  				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x24 	|
| RELU					|												|
| Dropout				| Using convolutional keep probablity 			|
| Max pooling   		| 2x2 stride,  outputs 5x5x24   				|
| Flatten       	    | outputs 600  									|
| Fully connected		| outputs 180 									|
| RELU					|												|
| Dropout				|Using keep probablity              			|
| Fully connected		| outputs 126 									|
| RELU					|												|
| Dropout				|Using keep probablity              			|
| Fully connected		| outputs 43 									|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eleventh cell of the ipython notebook. 

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 