# **Traffic Sign Recognition** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output/dataset-distribution.png "Visualization"
[image2]: ./output/additional-data.png "Additional test signs"
[image3]: ./output/additional-tests-outputs.png "Softmaxes"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/MarkoGodra/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb).

### **Data Set Summary & Exploration**

Numpy library was used in order to calculate basic dataset stats such as it's size, size of single image, size of validation/training/test set and number of total classes

* The size of training set is 34799 images
* The size of the validation set is 4410 images.
* The size of test set is 12630 images.
* The shape of a traffic sign image is 32x32x3 (width = 32, height = 32, color channels = 3 - RGB)
* The number of unique classes/labels in the data set is 43.

On image bellow we can see total number of each classes in each segment of data set (training, validation, test subsets respectivly). Each bar represents total number of occurances for each unique label in particular part of dataset.

![alt text][image1]

From these graphs we can conclude that data set is not evenly balanced, but distribution of unique classes is similar between training, validation and test set.

### **Design and Test a Model Architecture**

### Dataset Preprocessing

Main preprocessing step is normalizing dataset. With this we achieve mean value of each image of dataset to be close to 0, and have similar variances. This is important step, since it sagnificantly facilitates training process. Unscaled input can result in unstable learning process or 'exploding' gradients. Grayscale was not included as preprocessing technique since image color could prove to be useful information for sign classification, and with applying grayscale we lose this 'valuable' information.

Prior to dataset normalization, mean value of input was 36.352 and variance was 359.278 and after normalization step input mean was brought down to -0.716 and variance was 0.022. Basic calculation was used for data normalization where each pixel value was recalulated as:

`new_value = (old_value - 128) / 128`

### Model Architecture

Final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							
| Convolution(1) 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	
| RELU					|												
| Max pooling	      	| 2x2 stride,  outputs 14x14x6
| Convolution(2) 5x5	    | 1x1 stride, valid padding, outputs 10x10x16  
| RELU          		|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6
| Convolution(3) 3x3       | 1x1 stride, valid padding, outputs 3x3x32
| RELU                  |
| Concatenate      		| Takes flattened output from conv2 and conv3 (5x5x16) + (3x3x32) -> 400 + 288 = 688
| Fully connected(1)    | Input 688, output 120
| RELU                  |
| Dropout               | 50% Chance
| Fully connected(2)    | Input 120, output 84
| RELU                  |
| Dropout               | 50% Chance
| Fully connected(3)    | Input 84, output 43
| Softmax               |

### Model training

In order to train model, Adam optimizer was used with learning rate of 0.001 and batch size of 128. Total number of epochs used for model training was 20. Before each epoch, training dataset was randomly shuffeled. In earlier stages of training model suffered from underfitting (low training and validation accuracy ~89%), but adding additional convolutional layer helped solve that problem. Overfitting problem was not evident (training accuracy was not drastically higher than validation accuracy, and training accuracy never reached 100%) but anyway it was addressed with adding dropout for fully connected layers in order to improve generalization. 

### Model result and discussion

Final model results were:
* training set accuracy of 99.5%
* validation set accuracy of 95.6% 
* test set accuracy of 94.4%

Training set accuracy and validation accuracy is calculated on each epoch of training phase. The code for training phase is located in 13th cell of notebook.

Final model design was reached in iterative process. I have choosen LeNet-5 architecture as starting point. With initial architecture i achieved validation accuracy of roughly 89% - 91%. Second step included adding additional convolution layer, where output from this layer and output from previous layers are both flattened and combined before connecting with fully connected layer. This approach was inspired by [this](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) paper and 'Inception module' lesson from theoretical part of the course.
After these modifications model was able to achieve accuracy greater than 93% on validation set.

Additional tweak on model architecture was made by adding dropout at fully connected layer to eleminate potential overfitting issue, but also to help improve model's generalization. As already mentioned in 'Model training' section, total number of epoch choosen was 20 and model proved to have sufficient accuracy with this amount of epochs.

### Test a Model on New Images

Here are five German traffic signs that are used as additional testing inputs for parameters. These images were cropped from different images found on web and hopefully are not part of dataset. They were cropped and scaled manually.

![alt text][image2]

First image of this additional test set seemed like problematic image to predict, since there is part of another sign present in background. Other images seemed like straightforward cases for prediction. Only catch is slight twist n 3rd and 5th image.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road Work      		|  Right-of-way at next intersection
| Yield					| Yield											
| General caution       | General caution 					 			
| No Entry  			| No Entry   
| Speed Limit (30km/h)  | Speed Limit (30km/h)


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set which was 94.4%. Only false prediction was made on most 'difficult' image of all five images.

Top five softmax values for each of the images in additional test set is shown on following image:

![alt text][image3]

The code for making predictions on my final model is located in the 18th cell of the Ipython notebook.

In case of all images where prediction was correct, model showed relatively high certanity with minmum of 96.9% for case of 3rd image (General caution). In cases of 2nd (Yield), 4th (No Entry) and 5th (Speed limit 30 km/h) model was 100% certain of his prediction which seems like good thing.
In case of false prediction (1st image - Road work) model showed high certanity that sign shown in this image is 'Right-of-way at next intersection' which does not differ drastically from 'Road work' traffic sign, since they are both triangle-shaped and have red outline with white background and black symbol. This false prediction had preety high certanity which is not ideal. In addition to this, 'Road Work' sign is not in top five softmax values which also proves to be problematic.

Common thing for both of theses, 'Road work' and 'Right-of-way', signs is that they are not most frequent signs in dataset (aprox. 1000 occurances). Dataset augmentation could prove to be right way to handle this issue, where it could be used to try to balance out dataset, by generating augmented (rotated, nosiy) images for these less frequent classes.
