# Analysing and applying ML methods to a salary database
The purpose of this project is to apply data transformations and machine learning techniques in order to predict the expected salary of people person given a few descriptors such as their job title, their education and their experience.

## The Method
The analysis is performed through the method provided by a utility class, SupervisedDataframe, which has been created to manage the pandas data structure for a typical EDA+fast-modeling job.
SupervisedDataframe has tools to operate on the training and the validation set at the same time for data transformation and feature engineering. At the same time, the training and the validation set are properly isolated when special transformations (such as grouped statistics) are applied.
- Acquiring data from csv files and segmenting the dataframe into training and validation sets
- Plotting variables against specific variables
- Checking for null values and optionally deleting rows with null values
- Creating grouped statistics (mean, median, var) based on the categorical columns of the training set
- One hot encoding and scaling

## The Data

The data we will work on in this project are:

- 1,000,000 training examples, complete with salary information
- 1,000,000 test examples, with no salary information attached

While the dataset examined here is a simplified data model of a real-world job market, the procedures implemented here should be easily transferable to more complex scenarios and with a richer variety of categories.

As we will see when looking at the properties of each feature, at a first look the data seem to have been machine-generated over a uniform distribution. We will expect a lot of noise barring us from predicting the target with very high accuracy. Nevertheless, the data do show some interesting pattern that will eventually drive our hypotheses in building an appropriate model.
