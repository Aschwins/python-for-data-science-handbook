# Chapter 5: Machine Learning

The term “machine learning” is sometimes thrown around as if it is some kind of magic pill: apply machine learning to your data, and all your problems will be solved! As you might expect, the reality is rarely this simple. While these methods can be incredibly powerful, to be effective they must be approached with a firm grasp of the strengths and weaknesses of each method, as well as a grasp of general concepts such as *bias* and *variance*, *overfitting* and *underfitting*, and more.

## Other Resources Machine Learning in Python

To learn more about machine learning in Python, I’d suggest some of the following resources:

[The Scikit-Learn website](http://scikit-learn.org)
The Scikit-Learn website has an impressive breadth of documentation and exam‐ ples covering some of the models discussed here, and much, much more. If you want a brief survey of the most important and often used machine learning algo‐ rithms, this website is a good place to start.

**SciPy, PyCon, and PyData tutorial videos**
Scikit-Learn and other machine learning topics are perennial favorites in the tutorial tracks of many Python-focused conference series, in particular the PyCon, SciPy, and PyData conferences. You can find the most recent ones via a simple web search.

[Introduction to Machine Learning with Python](http://bit.ly/intro-machine-learning-python)
Written by Andreas C. Mueller and Sarah Guido, this book includes a fuller treat‐ ment of the topics in this chapter. If you’re interested in reviewing the fundamen‐ tals of machine learning and pushing the Scikit-Learn toolkit to its limits, this is a great resource, written by one of the most prolific developers on the Scikit-Learn team.

[Python Machine Learning](http://bit.ly/2eLDR7c)
Sebastian Raschka’s book focuses less on Scikit-Learn itself, and more on the breadth of machine learning tools available in Python. In particular, there is some very useful discussion on how to scale Python-based machine learning approaches to large and complex datasets.

## General Machine Learning
Of course, machine learning is much broader than just the Python world. There are many good resources to take your knowledge further, and here I highlight a few that I have found useful:

[Machine Learning](https://www.coursera.org/learn/machine-learning)
Taught by Andrew Ng (Coursera), this is a very clearly taught, free online course covering the basics of machine learning from an algorithmic perspective. It assumes undergraduate-level understanding of mathematics and programming, and steps through detailed considerations of some of the most important machine learning algorithms. Homework assignments, which are algorithmically graded, have you actually implement some of these models yourself.

[Pattern Recognition and Machine Learning](http://www.springer.com/us/book/9780387310732)
Written by Christopher Bishop, this classic technical text covers the concepts of machine learning discussed in this chapter in detail. If you plan to go further in this subject, you should have this book on your shelf.

[Machine Learning: A Probabilistic Perspective](https://mitpress.mit.edu/books/machine-learning-0)
Written by Kevin Murphy, this is an excellent graduate-level text that explores nearly all important machine learning algorithms from a ground-up, unified probabilistic perspective.

These resources are more technical than the material presented in this book, but to really understand the fundamentals of these methods requires a deep dive into the mathematics behind them. If you’re up for the challenge and ready to bring your data science to the next level, don’t hesitate to dive in!

# Machine Learning

With Machine Learning one can learn a computer to find patterns in data. And make predictions on these pattern for new not yet found data. One of the most famous examples of data where machine learning can be applied is the iris dataset, found in the seaborn library.

``` python
import seaborn as sns
sns.set()
iris = sns.load_dataset('iris')

sns.pairplot(iris, hue = 'species', size = 1.5)
```

<img src="./static/images/ml1.png" width="400px" />

This iris dataset contains the features of the plants, like the length and width of their petal. And it also contains it's species. With the features known one can make predictions on what kind of species of iris the plant is. This is the essence of machine learning.

Finding patterns in a features matrix on a target array in a training set. To make superb predictions in a target set. Since we have a training and a test set, this is called supervised learning. We're learning the algorithm what is good and bad. The other part of machine learning is unsupervised machine learning where it doesn't have a target vector and is just asked to find patterns.

## Scikit Learn

A great machine learning library is Scikit Learn, or sklearn in short. Using this library of API always goes about in similar fashion.

* Pick a model
* Define a features matrix X and target vector y
* Fit the model to your data by calling the `.fit()` method.
* Apply the model to new data, by call the `.predict()`
* Evaluate the model

### Supervised Learning Example: Simple linear regression

<img src="./static/images/ml2.png" width="400px" />

Let's start of with a small dataset configured with the following code. There is a linear correlation between the x and y data (axis) for all points in the dataset. The goal of linear regression is to build a model which simulates this correlation so we can make predictions on datapoints not in the original dataset.

``` python
# importing the linear regression module
from sklearn.linear_model import LinearRegression

# Creating some data with a linear correlation
rng = np.random.RandomState(40)
x = 10 * rng.rand(50)
y = 2 * x + 1 - rng.rand(50)

# Features matix X has to be a matrix
X = x[:, np.newaxis]

# Initiate a linear model
model = LinearRegression(fit_intercept = True)

# Fit the model to our data
model.fit(X,y)

# Predict values not in X
xfit = np.linspace(-1, 11)
Xfit = xfit[:, np.newaxis]

yfit = model.predict(Xfit)

# Plot the results
plt.scatter(x,y)
plt.plot(xfit, yfit);
```

<img src="./static/images/ml3.png" width="400px" />

With our model being represented with the blue line. Which has a intercept and coefficient.

``` python
In [27]: model.coef_
Out[27]: array([1.9755672])

In [28]: model.intercept_
Out[28]: -0.3853261759566511
```

And we've build our first Machine Learning Model! Oh my god!

### Supervised Learning Example: Iris Classification

After building out first model, we're (of course) very excited to build our second model. But for this example we should use something more exciting and make real predictions! Let's go back to our trusty iris dataset and see if we can build a model on one part of the dataset and with this model make predictions on the other part of data. Later we can compare results and see how good the model actually is.

For this we're going to use a very simple classification model called Gaussian Naive Bayes.

``` python
# Load the iris dataset and define the matrix of features and target vector
iris = sns.load_dataset('iris')
X_iris = iris.drop('species', axis = 1)
y_iris = iris['species']

# import function that automatically splits the data in a training and a test set.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris)

# import a simple classification model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

# train the model
model.fit(X_train, y_train)

# fit the model to the test-set
y_model = model.predict(X_test)

# evaluate the model
from sklearn.metrics import accuracy_score
accuracy_score(y_model, y_test)
Out[69]: 0.9473684210526315
```

Already getting a whopping 95% accuracy!!

### Unsupervised Learning Example: Dimensionality Reduction

Now since we've already seen the iris dataset a couple of times, we're not going in to much dept. The iris dataset contains four dimensions in the feature space: `sepal_length  sepal_width  petal_length  petal_width` and of course it's target vector of one dimension `species`. Visualising this 4/5 dimensional data in a two dimensional plot will be very hard. But with dimensionality reduction this is possible.

We'll compress the four dimensions in two dimensions trying to maintain as much of the variance of the data as possible with Principle Component Analysis or PCA for short.

``` python
import seaborn as sns
iris = sns.load_dataset('seaborn')
X_iris = iris.drop('species', axis = 1)
y_iris = iris['species']

from sklearn.decomposition import PCA
model = PCA(n_components = 2)
model.fit(X_iris)
X_2d = model.transform(X_iris)

iris['PCA1'] = X_2d[:, 0]
iris['PCA2'] = X_2d[:, 1]

sns.lmplot("PCA1", "PCA2", data = iris, hue = 'species', fit_reg = False)
```

Giving us the nice two dimensional plot with a third dimension of species on the colors.

<img src="./static/images/ml4.png" width="400px" />

### Unsupervised Learning Example: Iris Clustering

Next we'll try a clustering algorithm on the data. A clustering algorithm tries to find distict groups of data without having a target vector. For the iris dataset we'll use the Gaussian Mixture Model or GMM for short. A GMM tries to model the data as a collection of Gaussian blobs.

``` python
from sklearn.mixture import GMM
model = GMM(n_components = 3, covariance_type = 'full')
model.fit(X_iris)
y_gmm = model.predict(X_iris)   #predicting y labels

iris['cluster'] = y_gmm
sns.lmplot("PCA1", "PCA2", data = iris, hue = 'species', col = 'cluster', fit_reg = False)
```

This way we've told an algorithm to find a model that splits the data in three distinct groups without telling the algorithm anything about the target vector `species`. Because we've splitted the plot among the clusters we can very well see how well the model actually did. We can see that it was hundred percent accurate on find the cluster 0 species, while there is some mixture between the 1st and 2nd cluster. Supercool stuff!!

<img src="./static/images/ml5.png" width="800px" />


<img src="./static/images/ml6.png" width="500px" />
