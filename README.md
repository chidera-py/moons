Background
Kepler's Laws of planetary motion describe the orbits of planetary bodies around the Sun. According to Kepler's Third Law, the square of a planet's orbital period (
T
) is proportional to the cube of the length of the semi-major axis (
a
) of its orbit:

T
2
∝
a
3
.

This relationship is described by:

T
2
=
4
π
2
G
M
a
3
,

where 
M
 is the mass of the central body and 
G
 is the gravitational constant 
6.67
×
10
−
11
 
m
3
k
g
−
1
s
−
2
.

For the case of the Earth, orbitting around the sun, the mass 
M
 would be the mass of the Sun. If we were to apply Kepler's Law to the motion of moon's around a planet then 
M
 would be the mass of the planet (central body).

In this repository there is a dataset containing some physical information about a subset of Jupiter's moons. Jupiter is currently thought to have 95 moons in total. The largest and most well-studied of Jupiter's moons are the four Galilean moons: Callisto, Europa, Ganymede, and Io.

Data
You can find data on a large subset of Jupiter's moons in the Jupiter's moons dataset, which is provided in a .db file in the data folder. This SQLite database contains only one table called moons. There is no metadata provided alongside this dataset, so some exploratory work to find out exactly what is contained in the moons table. One thing to note is that this dataset contains a distance_km field (specifying the distance between each given moon and the planet Jupiter), which we will use as a proxy for semi-major axis in this task.


Task 1
You have been employed as a Research Software Engineer to help an astronomy group store and analyse data on Jupiter's moons. You have been asked to deliver a piece of software with two components:

a module called jupiter.py containing a Moons class
a Jupyter notebook that shows the researchers how to use the module, and takes them through an analysis of the dataset
The Moons class should load the data from the Jupiter's moons database provided. It is suggested that:

you store the full dataset as a data attribute of the class, along with any other attributes you think might be helpful to the researchers
you develop a set of methods that perform exploratory analysis of the data, for example, returning summary statistics, correlations between variables, plots
the class contains methods that conveniently extract particular pieces of information from the dataset (e.g., a method that extracts data for one moon)
When exploring the dataset, you might find it helpful to think about some of the following points:

How many fields and records are in the dataset?
How complete is the dataset?
Are there any remarkable trends, correlations or features in the data?
Does the dataset look reasonable, given what you might expect?
Annotate your notebook with markdown cells to explain your work, providing a clear narrative to help the researchers reuse your code.

Task 2
Use the Jupiter's Moons dataset and the equation from Kepler's Third Law to calculate an estimate for the mass of the planet Jupiter.

To do this you will need to prepare a linear regression model that relates 
T
2
 and 
a
3
. Remember to demonstrate that a linear model is appropriate to model the relationship between 
T
2
 and 
a
3
. Use markdown cells to explain your process and choices as you setup, train and test your model.

Ideally, the training, testing and prediction steps of your model should be added as methods of your Moons class.

Using the attributes from your linear model and the equation for Kepler's Third Law, estimate the mass of Jupiter in Kg. How does your estimate compare to the value from literature?

Remember the following:

Pay attention to the units quoted for different variables in the moons table and also for the gravitational constant 
G
. You may need to make some unit conversions when creating new columns in your DataFrame for 
T
2
 and 
a
3
.

Hint: 
G
 is in units of 
m
3
k
g
−
1
s
−
2
, so you should make sure that the inputs for your model use metres, kg and seconds too.

Think about whether you should set any hyperparameters when constructing the linear model and explain your reasoning.
Think about how you can validate the model.
Use in-code comments to explain the steps that you take to demonstrate that you are able to explain what you are doing.
