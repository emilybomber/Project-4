# Spaceship Titanic
## <u>Overview                                                                                                                              </u>
It's our mission to predict if passenger was transported to an alternate dimension during the Spaceship Titanic's collision with the spacetime anomaly.

_"Welcome to the year 2912, where your data science skills are needed to solve a cosmic mystery. We've received a transmission from four lightyears away and things aren't looking good._

_The Spaceship Titanic was an interstellar passenger liner launched a month ago. With almost 13,000 passengers on board, the vessel set out on its maiden voyage transporting emigrants from our solar system to three newly habitable exoplanets orbiting nearby stars._

_While rounding Alpha Centauri en route to its first destination—the torrid 55 Cancri E—the unwary Spaceship Titanic collided with a spacetime anomaly hidden within a dust cloud. Sadly, it met a similar fate as its namesake from 1000 years before. Though the ship stayed intact, almost half of the passengers were transported to an alternate dimension!"_

This project is a culmination of knowledge utilizing tools and techniques learned during the course. Chosen form Kaggle.com in the competition section, Spaceship Titanic helps demonstrate machine learning

## <u>Tools                                                                                                                                        </u>

<div align=center><a href="https://colab.research.google.com" target="_blank"><img src="https://img.shields.io/badge/Google Colab-F9AB00?style=for-the-badge&logo=Google Colab&logoColor=white"></a>
<a href="http://python.org" target="_blank"><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://tableau.com" target="_blank"><img src="https://img.shields.io/badge/Tableau-E97627?style=for-the-badge&logo=tableau&logoColor=white"></a><a href="https://pandas.pydata.org" target="_blank"><img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/></a>
<a href="https://scikit-learn.org" target="_blank"><img src="https://img.shields.io/badge/scikit learn-F7931E?style=for-the-badge&logo=scikit learn&logoColor=white"/></a><a href="https://tensorflow.org" target="_blank"><img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white"/></a>
<a href="https://keras.io" target="_blank"><img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white"/></a>
<a href="https://matplotlib.org" target="_blank"><img src="https://img.shields.io/badge/Matplotlib-800080?style=for-the-badge&logoColor=white"></a></div>


### Data Field Descriptions

PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.

HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.

CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.

Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.

Destination - The planet the passenger will be debarking to.

Age - The age of the passenger.

VIP - Whether the passenger has paid for special VIP service during the voyage.

RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.

Name - The first and last names of the passenger.

Transported - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

## Data Model Optimization


The provided Python script utilizes TensorFlow's Keras API to create a neural network model for binary classification. It begins by preparing the dataset, splitting it into training and testing sets using **train_test_split**, and scaling the features using **StandardScaler**. The neural network architecture is defined with two hidden layers, the first with 80 neurons and the second with 30, both utilizing ReLU activation functions. The output layer consists of a single neuron with a sigmoid activation function, suitable for binary classification tasks. The model is compiled with binary cross-entropy loss and the Adam optimizer, and accuracy is chosen as the evaluation metric.

During training, the model undergoes 100 epochs on the scaled training data (**X_train_scaled** and **y_train**). After training, the model's performance is evaluated on the scaled testing data (**X_test_scaled** and **y_test**). The evaluation results display the loss and accuracy achieved by the model on the testing set. In this instance, the model achieves an accuracy of approximately 79.12%, surpassing the required threshold of 75% for meaningful predictive power.

The iterative optimization process, crucial for enhancing the model's performance, can be documented in a CSV/Excel table or within the Python script itself. This documentation may include variations in hyperparameters, such as different layer sizes, activation functions, optimizers, or learning rates, along with their corresponding effects on the model's accuracy and loss. Such systematic experimentation and documentation facilitate understanding the model's behavior and guide future improvements to enhance its predictive power further.

![image](https://github.com/emilybomber/Project-4/assets/144865763/c7d0b67e-3b2f-4981-afce-af8aa0e3c47d)


![image](https://github.com/emilybomber/Project-4/assets/144865763/143a836d-b640-4612-b45d-107434ad82e0)

Print out of the results as follows:

68/68 - 0s - loss: 0.4423 - accuracy: 0.7912 - 98ms/epoch - 1ms/step
Loss: 0.4422522485256195, Accuracy: 0.7911683320999146

## Resources

Addison Howard, Ashley Chow, Ryan Holbrook. (2022). Spaceship Titanic. Kaggle. https://kaggle.com/competitions/spaceship-titanic

Image in tableau obtained from - https://neural.love/ai-art-generator/1eda7871-ebcd-6310-a0e2-8b95972b3201/titanic-in-space
