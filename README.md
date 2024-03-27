# Spaceship Titanic
## <u>Overview                                                                                                                              </u>
It's our mission to predict if a passenger was transported to an alternate dimension during the Spaceship Titanic's collision with the spacetime anomaly.

_"Welcome to the year 2912, where your data science skills are needed to solve a cosmic mystery. We've received a transmission from four lightyears away and things aren't looking good._

_The Spaceship Titanic was an interstellar passenger liner launched a month ago. With almost 13,000 passengers on board, the vessel set out on its maiden voyage transporting emigrants from our solar system to three newly habitable exoplanets orbiting nearby stars._

_While rounding Alpha Centauri en route to its first destination—the torrid 55 Cancri E—the unwary Spaceship Titanic collided with a spacetime anomaly hidden within a dust cloud. Sadly, it met a similar fate as its namesake from 1000 years before. Though the ship stayed intact, almost half of the passengers were transported to an alternate dimension!"_

This project is a culmination of knowledge utilizing tools and techniques learned during the course. Chosen form Kaggle.com in the competition section, Spaceship Titanic helps demonstrate machine learning with the use of the following tools.

## <u>Tools                                                                                                                                        </u>

<div align=center><a href="https://colab.research.google.com" target="_blank"><img src="https://img.shields.io/badge/Google Colab-F9AB00?style=for-the-badge&logo=Google Colab&logoColor=white"></a>
<a href="http://python.org" target="_blank"><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://tableau.com" target="_blank"><img src="https://img.shields.io/badge/Tableau-E97627?style=for-the-badge&logo=tableau&logoColor=white"></a>
<a href="https://pandas.pydata.org" target="_blank"><img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/></a>
<a href="https://scikit-learn.org" target="_blank"><img src="https://img.shields.io/badge/scikit learn-F7931E?style=for-the-badge&logo=scikit learn&logoColor=white"/></a>
<a href="https://tensorflow.org" target="_blank"><img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white"/></a>
<a href="https://keras.io" target="_blank"><img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white"/></a>
 
<a href="https://numpy.org" target="_blank"><img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white"></a>
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

### In these first sets of graphs we looked at a break down of the ships populous.

![image](https://github.com/emilybomber/Project-4/assets/146396417/aed9faf0-c151-4a9a-af8f-3a133804df46)

### This graph shows the age of the ship's passengers.

![image](https://github.com/emilybomber/Project-4/assets/146396417/35646685-ba0a-49a2-a81f-98378e7eec4c)




## Data Model Optimization


The provided Python script utilizes TensorFlow's Keras API to create a neural network model for binary classification. It begins by preparing the dataset, splitting it into training and testing sets using **train_test_split**, and scaling the features using **StandardScaler**. The neural network architecture is defined with two hidden layers, the first with 80 neurons and the second with 30, both utilizing ReLU activation functions. The output layer consists of a single neuron with a sigmoid activation function, suitable for binary classification tasks. The model is compiled with binary cross-entropy loss and the Adam optimizer, and accuracy is chosen as the evaluation metric.

During training, the model undergoes 100 epochs on the scaled training data (**X_train_scaled** and **y_train**). After training, the model's performance is evaluated on the scaled testing data (**X_test_scaled** and **y_test**). The evaluation results display the loss and accuracy achieved by the model on the testing set. In this instance, the model achieves an accuracy of approximately 79.12%, surpassing the required threshold of 75% for meaningful predictive power.

The iterative optimization process, crucial for enhancing the model's performance, can be documented in a CSV/Excel table or within the Python script itself. This documentation may include variations in hyperparameters, such as different layer sizes, activation functions, optimizers, or learning rates, along with their corresponding effects on the model's accuracy and loss. Such systematic experimentation and documentation facilitate understanding the model's behavior and guide future improvements to enhance its predictive power further.

![image](https://github.com/emilybomber/Project-4/assets/144865763/c7d0b67e-3b2f-4981-afce-af8aa0e3c47d)


![image](https://github.com/emilybomber/Project-4/assets/144865763/143a836d-b640-4612-b45d-107434ad82e0)

#### Print out of the results as follows:

68/68 - 0s - loss: 0.4423 - accuracy: 0.7912 - 98ms/epoch - 1ms/step Loss: 0.4422522485256195, Accuracy: 0.7911683320999146

### From the trained data we see the following breakdowns of passengers transported.

![image](https://github.com/emilybomber/Project-4/assets/146396417/0b81337d-f876-4691-a64a-0d966bfd07c8)

After running an initial model, we the opted to use **Keras Tuner**

The provided code demonstrates the utilization of Keras Tuner, a tool for hyperparameter tuning in TensorFlow, to optimize the neural network model for binary classification. Instead of manually specifying hyperparameters like activation functions and layer sizes, Keras Tuner automates this process by searching through a predefined search space for the best set of hyperparameters that maximize the chosen objective metric. In this case, the objective is set to maximize validation accuracy.

The **create_model** function defines the architecture of the neural network model with tunable hyperparameters using Keras Tuner's API. It allows Keras Tuner to choose the activation function for hidden layers, the number of neurons in the first layer, the number of hidden layers, and the number of neurons in each hidden layer. These hyperparameters are then used to construct the model within the search space defined by the user.

The Keras Tuner instance, configured with the Hyperband algorithm, conducts a hyperparameter search over a specified number of epochs and iterations. The search is performed using the training data (**X_train_scaled** and **y_train**) while validating against the testing data (**X_test_scaled** and **y_test**). After the search, the best hyperparameters and the corresponding model are obtained. The evaluation of the best model on the testing data reveals its performance, which in this case achieves a validation accuracy of approximately 80.36%, demonstrating the effectiveness of using Keras Tuner for optimizing neural network models.

#### Best model hyperparameters from the keras tuner:

{'activation': 'tanh',
 'first_units': 10,
 'num_layers': 6,
 'units_0': 2,
 'units_1': 5,
 'units_2': 6,
 'units_3': 2,
 'units_4': 7,
 'tuner/epochs': 50,
 'tuner/initial_epoch': 17,
 'tuner/bracket': 3,
 'tuner/round': 3,
 'units_5': 1,
 'tuner/trial_id': '0049'}

![image](https://github.com/emilybomber/Project-4/assets/144865763/a240b4e2-1f3d-4bb0-8c91-28a827b88de7)

![image](https://github.com/emilybomber/Project-4/assets/144865763/a6039bed-2924-4845-9506-6a7d6b7fb319)

#### Print out of the results as follows:

68/68 - 0s - loss: 0.4686 - accuracy: 0.8059 - 383ms/epoch - 6ms/step Loss: 0.46856844425201416, Accuracy: 0.805887758731842

## Final Analysis

The project aims to predict whether a passenger was transported to an alternate dimension during the collision of the Spaceship Titanic with a spacetime anomaly. Through the use of machine learning techniques and tools, specifically TensorFlow's Keras API and Keras Tuner, the project successfully creates and optimizes a binary classification model.

The data starts with information about passengers and gains insights into the age distribution, VIP status, and other demographic characteristics.

The data model optimization process involves the development of a neural network model using TensorFlow's Keras API. The model architecture consists of two hidden layers with ReLU activation functions, culminating in a sigmoid activation function for binary classification. Through iterative optimization, the model achieves an accuracy of approximately 79.12% on the testing set, surpassing the required threshold of 75%.

To further enhance model performance, Keras Tuner is applied for hyperparameter tuning. Keras Tuner automates the search for optimal hyperparameters, such as activation functions and layer sizes, maximizing the validation accuracy. The best model obtained through Keras Tuner achieves a validation accuracy of approximately 80.36%, demonstrating the effectiveness of this approach in optimizing neural network models.

Overall, the project showcases the application of machine learning techniques to address a "real-world" problem in a futuristic context, highlighting the importance of data analysis, model development, and optimization in solving complex challenges.

## Resources

Addison Howard, Ashley Chow, Ryan Holbrook. (2022). Spaceship Titanic. Kaggle. https://kaggle.com/competitions/spaceship-titanic

Image in tableau obtained from - https://neural.love/ai-art-generator/1eda7871-ebcd-6310-a0e2-8b95972b3201/titanic-in-space

![image](https://github.com/emilybomber/Project-4/assets/46686019/19c0d2be-ec5f-4d24-b0ed-efeeb13226fb)

