# Chapter 4: Visualization with Matplotlib
*Other libraries or plotting methods include: D3js, seaborn, ggplot, HoloViews, Altair*

## General Matplotlib Tips
Let's start of with a little bit of basic code, to set everything up right!

``` Python
import matplotlib as mpl
import matplotlib.pyplot as plt

# pick your style!
plt.style.use('classic') #others: 'seaborn-whitegrid'

# enabling mpl interactively in ipython:
%matplotlib

# basic plotting example
x = np.linspace(0, 10, 100)
fig = plt.figure()
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--');

plt.savefig('my_figure.png')

from IPython.display import Image
Image('my_figure.png')
```

## Two interfaces for the Price of One

Getting figures containing two plots is relatively easy with matplotlib, but there are two face to do it. With the MATLAB-Style interface and the Object-oriented interface.

Where the object-oriented is preferable if plots get more complicated since you're able to edit multiple plots afterwards, while the MATLAB-style is more or less 'permanent'.

### MATLAB-style interface
``` Python
plt.figure() #create a plotting figure

plt.subplot(2,1,1) #(rows, columns, panel number)
plt.plot(x, np.sin(x))

plt.subplot(2,1,2)
plt.plot(x, np.cos(x))
```

### Object-oriented interfaces
``` Python
# First create a grid of plots
# ax will be an array of two Axes objects
fig, ax = plt.subplots(2)

# Call plot() method on appropriate objects
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x));
```

## Simple line plots
Again there are two ways to plot a simple graph with matplotlib, and guess what it's easier than plotting two figures.

``` Python
#object oriented
plt.style.use('seaborn-whitegrid')

fig = plt.figure()
ax = plt.axes()

ax.plot(x, np.sin(x))

#matlab-Style
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
```

### Adjusting the Plot: Line Colors and Style

Adjusting the line colors can be done in several ways.

``` Python
In[6]:
plt.plot(x, np.sin(x - 0), color='blue')          # Specify color by name
plt.plot(x, np.sin(x - 1), color='g')             # short color code (rgbcmyk)
plt.plot(x, np.sin(x - 2), color='0.75')          # Grayscale between 0 and 1
plt.plot(x, np.sin(x - 3), color='#FFDD44')       # Hex code (RRGGBB from 00 to FF)
plt.plot(x, np.sin(x - 4), color=(1.0,0.2,0.3))   # RGB tuple, values 0 and 1
plt.plot(x, np.sin(x - 5), color='chartreuse');   # all HTML color names supported
```

Similarly you can adjust the line style using the linestyle keyword.

``` Python
plt.plot(x, x+0, linestyle='solid')
plt.plot(x, x+1, linestyle='dashed')
plt.plot(x, x+2, linestyle='dashdot')
plt.plot(x, x+3, linestyle='dotted');

# For short you can use the following codes:
plt.plot(x, x + 4, linestyle='-')   # solid
plt.plot(x, x + 5, linestyle='--')  # dashed
plt.plot(x, x + 6, linestyle='-.')  # dashdot
plt.plot(x, x + 7, linestyle=':')   # dotted
```

If you're really the stylish type these linestyle and color codes can be combined.

``` Python
plt.plot(x, x + 0, '-g')    # solid green
plt.plot(x, x + 1, '--c')   # dashed cyan
plt.plot(x, x + 2, '-.k')   # dashdot black
plt.plot(x, x + 3, ':r')    # dotted red
```

### Adjusting the Plot: Axes Limits

Adjusting the axes, adding labels, adding a legend is all done in a very easy fashion with matplotlib. The following codes shows how to do several of these things.

``` Python
plt.xlim(-1,1)
plt.ylim(-1.5,1.5)

plt.axis([xmin, xmax, ymin, ymax])
plt.axis('tight')
plt.axis('equal')

plt.title('A Sine Curve')
plt.xlabel('x')
plt.ylabel('sin(x)')

plt.plot(x, np.sin(x), '-g', label='sin(x)')
plt.plot(x, np.cos(x), ':b', label='cos(x)')
plt.legend()
```

In an object oriented plot everything is done in a similar fashion:

``` python
ax = plt.axes()

plt.plot() = ax.plot()
plt.legend() = ax.legend()
plt.xlabel() = ax.set_xlabel()
plt.ylabel() = ax.set_ylabel()
plt.xlim() = ax.set_xlim()
plt.ylim() = ax.set_ylim()
plt.title() = ax.set_title()

ax = plt.axes()
ax.plot(x, np.sin(x))

# Setting all options all at once:
ax.set( xlim=(0, 10), ylim=(-10,10),
        xlabel = 'x', ylabel='sin x'
        title = 'Sinus of the x'
      )
```

In the above plots we've taken a lot of (100) data points so all the Sine lines look pretty smooth. But something is going on under the hood. It actually making a scatterplot and chosing a default symbol value '-' to connect the dots.

### Simple Scatter Plots

There are several nice markers you can use to scatter you data among a plot:

``` Python
rng = np.random.RandomState(0)
markers = ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']

for marker in markers:
  plt.plot(rng.rand(5), rng.rand(5), marker,
  label = "marker = {0}".format(marker))

plt.legend()
plt.xlim(0, 1.8)
```

Colors and markers can be combined!

``` Python
plt.plot(x, y, '-ok'); # line (-), circle marker (o), color black (k)
```

And plots can be tuned of course:

``` Python
plt.plot(x, y, color = 'gray'
marker = 'p', markersize = 15,
linewidth = 4,
markerfacecolor = 'white'
markeredgecolor = 'grey'
markeredgewidth = '2'
)
```

### Scatter Plots with plt.scatter

``` Python
plt.scatter(x, y, marker = 'o')
```

The primary difference between `plt.plot` and `plt.scatter` is that with plt.scatter each individual point (size, face, color, edge color, etc.) can be individually controlled or mapped to data.

``` Python
rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)

sizes = 1000 * rng.rand(100)
colors = rang.rand(100)

plt.scatter(x, y, c = colors, s = sizes, alpha = 0.3, cmap = 'viridis')
plt.colorbar()
```

``` Python
from sklearn.dataset import load_iris
iris = load_iris()

features = iris.data.T

plt.scatter(features[0], features[1], alpha = 0.2,
s=100 * features[3], color = iris.target, cmap='viridis',
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1]))

```

## Visualizing Errors

Some variables are not really certain that's why the called variables. Sometime we can say a variable has a value x with max error dx. Visualising values with error rates can be done with:

``` Python
plt.errorbar(x, y, error = dy, fmt = 'o', color = 'black', ecolor = 'lightgray', elinewidth = 2, capsize = 3)
```

There are a lot of other ways to plot error bars. Horizontal ones, `plt.xerr()`, one sided ones and a ton of other variants. See docstring of plt.errorbar
