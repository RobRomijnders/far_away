# Introduction
In this project, we explore how three linear classifiers behave for input points that are _far away_ from the typical data. We will compare Logistic regression and Gaussian process classifier. We compare what a trained model predicts on input points that are _far away_ from the data points of the train set

# Simplicity	
We often herald Logistic regression for its simplicity. Logistic regression is easily explainable, allows for easy inference, and almost every out-of-the-box machine learning library in any language implements the algorithm. However, I see one drawback. Points _far away_ from the typical training data get a high or low output. This might be a problem in production environments. As the input data distribution shifts, the user might be tricked into interpreting the high outputs as confident classifications. 

# Example
In the following example, we make two point clouds. One cloud of positive examples and one cloud of negative examples. We fit both a logistic regression model (LR) and a gaussian process classifier (GP). As the clouds are fairly distinct, both models achieve higher than 95% accuracy. 

The background of the scatter plots colors according to probability output of the model. Both LR and GP output a number between 0 and 1, which we interpret as a probability. 

Here are both plots
![plot](https://github.com/RobRomijnders/far_away/blob/master/far_away/im/heatmap_logreg_gp.png?raw=true)

For logistic regression: the further we go right, the higher the probability becomes. This means that an input at `x=(4, 1)` will output a very high probability. (and vice versa to the far left)

But how justified is this high probability?

We have not seen any data around the `x=(4,1)` inputs. Our model outputs a high probability, but later on we might as well observe negative examples in this area.

For the gaussian process classifier, however, the probability tends to `0.5` as we go far away from the observed data. If we now consider the input `x=(4, 1)`, then the output will be around `0.5`. This feels justified, as we have no reason to assume other knowledge.

# Side notes
Of course, the gaussian process classifier also has disadvantages to logistic regression. Making a prediction requires the inverse of the Gram matrix, requiring `O(n**3)` computation. Moreover, explaining gaussian processes will probably take more time than explaining logistic regression.

As always, I am curious to any comments and questions. Reach me at romijndersrob@gmail.com

