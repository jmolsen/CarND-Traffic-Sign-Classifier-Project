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
[image7]: ./writeup_images/no_stopping_sign_32x32.jpg "Traffic Sign 1: No Stopping"
[image8]: ./writeup_images/30km_sign_32x32.jpg "Traffic Sign 2: 30 km/h"
[image9]: ./writeup_images/60km_sign_32x32.jpg "Traffic Sign 3: 60 km/h"
[image10]: ./writeup_images/children_crossing_sign_32x32.jpg "Traffic Sign 4: Children Crossing"
[image11]: ./writeup_images/stop_sign_32x32.jpg "Traffic Sign 5: Stop"


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

Finally, in the eighth code cell, I converted the images to grayscale, ran histogram equalization, and ran min-max scaling on them.  In some of my earlier versions I was using the RGB images, but thought that I might be making it too complicated.  I thought that since signs should be recognizable mostly by their shape and symbols and in varying lighting conditions that it would better to try to learn them as normalized grayscale.

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
| ELU					|												|
| Dropout				| Using convolutional keep probablity 			|
| Max pooling	      	| 2x2 stride,  outputs 14x14x9  				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x24 	|
| ELU					|												|
| Dropout				| Using convolutional keep probablity 			|
| Max pooling   		| 2x2 stride,  outputs 5x5x24   				|
| Flatten       	    | outputs 600  									|
| Fully connected		| outputs 180 									|
| ELU					|												|
| Dropout				|Using keep probablity              			|
| Fully connected		| outputs 126 									|
| ELU					|												|
| Dropout				|Using keep probablity              			|
| Fully connected		| outputs 43 									|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is primarily located in the eleventh cell of the ipython notebook with a few parameters defined earlier in the tenth cell. I trained the model in batches of size 128 for 40 epochs.  I used mostly the same training method used in the LeNet lab, using softmax cross entropy as the basis for calculating the loss, and minimizing the loss using the AdamOptimizer.  However, I used ELU for activation instead of RELU, and I supplemented the loss operation by adding a L2 Regularization calculation to the cross entropy when reducing the mean.  I decided to try The ELU (Exponential Linear Unit) base on a paper I was referred to which suggested that ELUs can be faster and provide higher accuracy than RELUs.  I added in the L2 Regularization in order to penalize high weights to prevent overfitting.

For the random generation of weights and biases I used 0 for mu and 0.1 for sigma. - it made sense to keep the starting weights small since I didn't want large weights causing overfitting on any particular feature. I used a learning rate of 0.001 and a beta value for L2 Regularization also of 0.001.


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the training and validation accuracy of the model is located in the eleventh cell of the Ipython notebook.  The code for calculating the test accuracy is in the twelfth cell.

My final model results were:
* training set accuracy of 98.0%
* validation set accuracy of 95.6% 
* test set accuracy of 92.8

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

I started out with LeNet but adjusted to handle color images and the larger number of output logits because I knew it worked well for the MNIST numbers dataset, and I though the traffic signs might only be a little more complicated to classify.

* What were some problems with the initial architecture?

I wasn't getting high enough validation accuracy and so I think the model needed to be adjusted both to capture the more complex nature of the traffic signs, and also to include features to prevent overfitting.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I made several adjustments to the architecture and tuned and tweaked the architecture and the hyperparameters over several runs in order to increase both training and validation accuracy. I also tried to keep the difference between training and validation accuracy small since a larger difference (assuming the training accuracy is larger) suggests overfitting to the training data.
In an attempt to capture the more complex nature of this dataset in comparison to the MNIST number set, I increased the output depth at each layer.  I experimented with several different values, but ended up with values that are about 50% larger than those from LeNet.  I also experimented with additional convolutional layers with dropout, but without pooling when I was originally attempting to use color images.

I added a dropout function call for each layer using a separate keep probability for the convolutional layers and the fully connected layers.  I added those in to allow me additional ways to prevent overfitting.  And I had two separate keep probabilities because I thougth I might want to keep the dropout less aggressive in the convolutional layers since I was already using max pooling and I didn't want to lose too much information on those layers.  I ultimately ended up using a keep probability of 1.0 for the convolutional layers so that I was effectively not doing dropout on those layers, but I was glad to have the parameter to tweak.

Finally, as mentioned above, I also added L2 Regularization in my loss calculation as yet another way to prevent overfitting by penalizing large weights.

* Which parameters were tuned? How were they adjusted and why?

I tried different values of sigma during some early runs to see if I might train faster assuming that smaller values would likely get me closer to what I wanted.  I ultimately just increased my epochs to ensure I had enough iterations to learn the weights that get me the best accuracy.

As mentioned above, I experimented with different values for the layer widths to see if having more features might increase overall accuracy.  Even though my final accuracy for training and validation was decent, I wonder if perhaps I should have increased the values here more based on the results I got from test validation and the web images.

As far as the learning rate, I experimented with some slightly smaller learning rates to see if I might increase my accuracy in case my learning rate was so big that it was causing my model to jump over weights that might lead to better accuracy.  Since I was using a fairly large epoch size I thought it would also afford the model the time to learn at a slower rate.  However, I didn't find any real benefit so I put the learning rate back to what we had for LeNet.

The other three hyperparameters I fiddled with are the beta for L2 Regularization, the keep probability (for the fully connected layers), and the convolutional keep probability.  Whenever I saw a disparity of a few more more percentage points between my training accuracy and validation accuracy (and the training accuracy was larger -- which it always was by the end of training) suggesting overfitting, I would try different combinations of adjustments to these values since they all can help prevent said overfitting.  So, that would include increasing the beta if I wanted to more harshly penalize large weights during L2 Regularization or decreasing either or both keep probabilities to drop a higher percentage of weights during dropout.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

I used convolutions since I knew the images we were training on might have the signs or symbols on the signs showing up in different locations on each image and I wanted to capture those features regardless of their exact location within the image.
I increased the layer widths compared to LeNet because I thought the traffic signs were slightly more complicated than the MNIST numbers.  I chose to add in dropout to each layer as well as L2 Regularization in order to give me ways to prevent overfitting.  In general I wanted to have options to tweak in case of any overfitting that was evident based on the comparison of training and validation accuracy. In addition, I was concerned about overfitting because some traffic signs had very few examples, and there's only so much additional variety that I could artificially generate through image manipulation as I did with rotations and translations for training set augmentation.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![No Stopping][image7] ![30 km/h][image8] ![60 km/h][image9] 
![Children Crossing][image10] ![Stop][image11]

The first image might be difficult to classify because it has a circular border like several other signs and the X in the middle could be similar to the angled arrows in the keep right or keep left signs.
The second and third images might be difficult to classify again because of several signs have the same circular border and all of the speed limit signs having at least a zero in them.
The fourth image might be difficult to classify because it has a triangle border like several other signs and a few of those also have small details within the triangle like this children crossing one which might be hard to distinguish.
The fifth image might be difficult to classify because it has an "O" in it which might look like the "0"s in the speed limit signs.


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the fourteenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No Stopping      		| Road work   									| 
| 30 km/h     			| 20 km/h 										|
| 60 km/h				| 20 km/h										|
| Children Crossing		| Road narrows on the right		 				|
| Stop      			| Stop              							|


The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. This does not compare favorably to the accuracy on the test set of 92.8%
On the 4 signs that the model got wrong I can see how the predictions are similar to the correct sign.  Coupled with the test set accuracy of 92.8%, it suggests that my model was a bit overfitted to the training data.  In addition to tweaking some parameters which help prevent overfitting such as L2's beta and the dropout keep probabilities, I also think that some things I could try to improve the model might be to increase the the layer widths to capture more detail and possibly to make my convolutional filter larger to capture more features together (in case some of the missed predictions are a case of not seeing the forest for the trees).

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for analyzing performance on my final model is located in the fifteenth and sixteenth cells of the Ipython notebook.

Based on the probabilities for each sign, it seems that generally speaking, the predictions were pretty certain even when they were wrong.  Out of the four that were predicted incorrectly, at least for Children Crossing, the correct sign was the second choice even though the model was the most certain of it's wrong choice.

For the No Stopping sign, the top 5 probability values and associated predictions were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 61.3      			| Road work             						| 
| 14.6      			| Priority road     							|
| 11.3  				| Bumpy road        							|
| 8.9       			| Keep right        			 				|
| 3.5           	    | Bicycles crossing     						|


For the 30 km/h sign, the top 5 probability values and associated predictions were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 77.8      			| Speed limit (20km/h)							| 
| 18.1      			| Speed limit (70km/h)							|
| 2.5       			| Stop                      					|
| 0.62       			| General caution				 				|
| 0.61           	    | Speed limit (120km/h)  						|


For the 60 km/h sign, the top 5 probability values and associated predictions were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 78.2      			| Speed limit (20km/h)							| 
| 17.2       			| Double curve      							|
| 1.5       			| Road narrows on the right    					|
| 0.58      			| Speed limit (120km/h)      	 				|
| 0.47          	    | Wild animals crossing    						|


For the Children Crossing sign, the top 5 probability values and associated predictions were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.9      			| Road narrows on the right						| 
| 0.0298    			| Children crossing								|
| 0.0078    			| Bicycles crossing 							|
| 0.0069    			| Beware of ice/snow			 				|
| 0.0011        	    | Pedestrians             						|


For the Stop sign, the top 5 probability values and associated predictions were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 74.2      			| Stop      									| 
| 22.4      			| General caution								|
| 2.1       			| Speed limit (20km/h) 							|
| 1.0       			| Speed limit (70km/h)  		 				|
| 0.056         	    | Speed limit (120km/h)    						|




