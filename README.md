# Modelling the Solar System Numerically
## Solar System Model - RK4
### 07/10/2021

solar_system_model_rk4.py is my most recent coding project to date. This program was written for my 3rd year Computational Projects module. It is currently in an intermediate state as extra functionality shall need to be added in order to complete my project goals. 

This program improves upon my previous solar system model (solar_system_model.py) as it incorporates the Runge-Kutta method which has improved accuracy over the Euler method. This allows for larger values of dt, resulting in faster computation times. This program also includes N-body calculations which considers the gravitational forces from all other planets in the solar system.

In this project, I furthered my knowledge of object-oriented programming, allowing me to write more robust code which can be used for other applications. This was necessary as the project requirements were quite broad, meaning a purely procedural approach would not be as effective.

An interesting result of this model was that, with the addition of the Moon orbiting the Earth, the resulting Earth-Moon system took longer to orbit the Sun than it takes the Earth to orbit without its Moon. This is a concept I would like to investigate further to fully understand the physics behind this.

## Solar System Model - Euler
### 12/03/2020
solar_system_model.py is a program I wrote for a 1st year uni assignment with which I received a grade of 99%. The question was as follows:

Implement the forward Euler method, using the difference equations above, to model the orbits of the four closest planets to the Sun.  Here we are assuming that each planet only feels the Sun's gravitational pull (and not that of the other planets) and that motion is confined to the xy-plane.

I decided to make use of object-oriented programming in order to simplify my code and reduce its length by almost half. This was a great learning experience as I had never used objects before and to achieve an almost perfect score was extremely satisfying.

# Investigation into Machine Learning
## Decision Tree Regressor
### 09/09/2021

snClassifier.ipynb is my most recent investigation into ML with the intention of applying my machine learning knowledge to my area of study; astrophysics.

This program uses a database of supernova and their classifications which includes information about the supernova, its environment and the space between the earth and the supernova.

Using a decision tree regressor, I was able to achieve an accuracy of 75% in determining the type of a supernova.

I believe that this low success rate is due to the modelling approach chosen. A more in-depth investigation into the pros and cons of different models should significantly improve the accuracy in the future.

This project exposed me to the sklearn library and the convenience of using Jupyter Notebook when working on machine learning projects, allowing databases to be shown concisely and removes the necessity to retrain a model each time the code is run.

## Sequential Model
### 01/07/2020

Movie_Reviews.py is a program which uses the sequential model to determine whether a movie review is positive or negative. 

This is the result of a series of YouTube tutorials I followed during the summer of 2020 while in self isolation due to the COVID-19 pandemic. The following links are to the playlists of videos I followed: 

https://www.youtube.com/playlist?list=PLzMcBGfZo4-mP7qA9cagf68V06sko5otr  
https://www.youtube.com/playlist?list=PLzMcBGfZo4-lak7tiFDec5_ZMItiIIfmj

I was very interested in this project as I believe that machine learning can be invaluable in the world of physics and that the possibilities that machine learning provides are seemingly endless. I very much enjoyed testing the final result of this project with reviews of my favourite movies and even my own, slightly misleading, movie reviews to really test the limitations of the program. I am excited to discover the applications of machine learning not in use today!
