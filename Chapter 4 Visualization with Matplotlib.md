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
