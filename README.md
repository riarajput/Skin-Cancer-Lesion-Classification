## Skin-Lesion-Image-Classification-using-CNN-and-Transfer-Learning

#### Dataset chosen : SKIN CANCER MNIST : HAM10000

#### Link to the dataset :

https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000

#### Number of classes present in the dataset :

#### a. Melanocytic Nevi
#### b. Benign Keratosis-like lesions
#### c. Basal cell carcinoma
#### d. Actinic keratoses
#### e. Vascular lesions
#### f. Dermatofibroma
#### g. Melanoma

This repository consists of a total of 4 jupyter notebooks which are as follows :

#### 1. Exploratory Data Analysis (EDA) and data preprocessing :- 
- This jupyter notebook consists of different methods employed in order to explore the data such as creation of a dictionary where each type of skin lesion has been assigned a unique label depending upon whether it is bening or malignant, the benign skin lesions are assigned with a '0' while the malignant skin lesions are assigned with a '1'. 
- As well as the dataset which is loaded is in the form of a directory which was created by joining the absolute path of the complete folder to the absolute path to the metadata csv file of the dataset. This was done in order to concatenate the csv data to the actual image data in order to proceed with the final mapping of different columns to the pixel values of the existing images. 
- Finally this jupyter notebook consists of all the resized images for different purposes which will be discussed in the further jupyter notebooks.

#### 2. Building of a Convolutional Neural Network (CNN) and hyperparameter tuning :-
- In this jupyter notebook, I have built a simple convolutional neural network by the method of sequential modelling by using the keras library. The CNN consists a total of 11 layers with the last layer being the output layer specifically used for classification purposes. 
- The loss function used in order to build this neural network is categorical cross entropy as the dataset contains a total of 7 different classes of skin lesions.
- The dataset used in this jupyter notebook is the resized image file of the size 64x64x3 where '3' signifies that the images are coloured images and are a combination of 'RGB'.
- A CNN is built in order to classify these skin lesion images with the best precision as possible. The accuracy achieved by the CNN built using sequential modelling is equal to 77.85% at a batch size of 64 and epoch value as 30.
- Later in the jupyter notebook, a keras classifier is instantiated and wrapped in the scikit learn library package in order to perform the hyperparameter tuning using the GridSearchCV option of the sklearn library. The model definition of the keras classifier consists of the same exact steps as the baseline CNN model definition. Here the hyperparameters taken into consideration are the batch size and epoch values.
- Best accuracy attained after hyperparameter tuning is equal to 75.25% at an epoch value of 10 and batch size of 30 after iterating the model with the values as batch_size = [10,20,30] and epochs = [10,25,40].
- Finally the jupyter notebook consists of a function to plot the confusion matrix in order to find out the most misclassified classes and a function to plot the bar graph highlighting the most misclassified classes.

#### 3. Building of a DCNN using the technique of transfer learning using MobileNet architecture with Imagenet as the input
- In this jupyter notebook, I have downloaded the MobileNet architecture consisting a total of 23 layers but with the input weight as "Imagenet" due to which the final model consists of a total of 87 layers in general. This is what is the first step of transfer learning.
- After setting up the model from the pretrained weights, a few more layers are added to build the actual DCNN model which is later trained with an optimizer function of 'Adam' and loss function as 'Categorical cross entropy", the training set accuracy received just by training this model is equal to 73.65% when the last layer is taken into consideration.
- As the training set accuracy is not upto the mark, the model is fine-tuned by just taking any random layer into consideration and the fine tuned model achieved an accuracy of 79.06% which was higher than the trained model.

#### 4. Retraining the complete MobileNet architecture in order to get the maximum training set accuracy.
- In this jupyter notebook, the initial steps to build a DCNN model are the same as the previous jupyter notebook concentrating on fine tuning of the DCNN model built with the only difference as the complete MobileNet architecture is retrained i.e. by not choosing a single random layer but by choosing all the layers of the architecture. 
- This technique was faster and gave the best performing model accuracy of 91.20% at a batch_size of 32 and epoch value of 15.

#### 5. A Power Point presentation consisting of the results in the complete project

#### 6. A report consisting on various definition of terms used in the complete project such as Convolutional Neural Networks, transfer learning, architectural description of MobileNet model as well as final comparison of the performance of all the techniques employed in order to achieve the best working accuracies.
