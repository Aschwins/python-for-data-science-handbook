# Chapter 2: Introduction to NumPy

## Data types in Python

Python in short is actually a smart wrapper around C so dynamic programming can be applied.  Arrays in Python can contain different kind of datatypes, int, float, bool, string, etc. But this comes at a cost: efficiency. Now np.array() solves this issue once again since it can only contain one datatype!

## Creating arrays from scratch

``` Python
#length 10 integer array filled with zeros
np.zeros(10, dtype = int)

# 3 by 5 float array filled with ones
np.ones((3,5), dtype=float)

# 3 by one array filled with 3.14
np.full((3,5), 3.14)

# array with linear sequence starting at 0, ending at 20, stepping by 2
np.arange(0, 20, 2)

# array of five values evenly spaced between 0 and 1
np.linspace(0, 1, 5)

np.random.random((3,3))

np.random.normal(0,1,(3,3))

np.random.randint(0,10,(3,3))

np.eye(3)

np.empty(3)
```

## The Basics of NumPy Arrays

``` Python
np.random.seed(0)

x1 = np.random.randint(10, size=6)

x2 = np.random.randint(10, size= (3,4))

x3 = np.random.randint(10, size = (3,4,5))

print("x3 ndim: ", x3.ndim)
print("x3 shape: ", x3.shape)
print("x3 size: ", x3.size)
print("x3 itemsize: ", x3.itemsize)
print("x3 nbytes: ", x3.nbytes)
```

## Array slicing: Accessing subarrays

``` Python
x[start:stop:step]
x[::-1] # all elements reversed
```

Changing subarrays results in changing the real array, use copy() if you don't want this. Use .reshape() to reshape arrays, size must be the same and uses nocopy by default.

## Array Concatenation and Splitting

``` Python
# Let x,y,z be one dimensional arrays
np.concentate([x,y]) # joining two arrays
np.concentate([x,y,z]) # joining more than two arrays

# Let grid be a two dimensional array
grid = np.array([[1,2,3],[3,4,5]])
np.concentate([grid, grid]) # axis = 0, puts the grids under eachother
np.concentate([grid, grid], axis=1) #puts the grids next to eachother

# so pasting along the axis.

# If they have different dimensions use np.vstack or np.hstack to stack vertically resp. horizontally.
# Use np.dstack for stacking along a third axis.

# Opposite of Concatenation is splitting used by: split, vsplit, hsplit
np.split(1darray, [splitpoint1,splitpoint2,...,splitpointn])
np.vsplit(2darray,[splitpoint1,splitpoint2,...,splitpointn])
np.hsplit(2darray,[splitpoint1,splitpoint2,...,splitpointn])

#np.dsplit will split arrays along the third axis.
```

## Computation on NumPy Arrays: Universal Functions
The key to making it fast is to use vectorized operations, implemented through Numpy's universal functions (ufuncs).

Python is slow, using C or Fortran is fast. For this problem various attempts to solve this weakness are: PyPy Project, Cython Project, Numba Project. But well use the standard CPython engine!

If you see python loops replace them with numpy vectors if you can.

NumPy's Ufuncs exists in two flabors: unary ufuncs, which operate on a single input, and binary ufuncs, which operate on two inputs.

### Ufuncs
``` python
+, -, *, /, //
-x, x **2, x%2
abs()
sin(), cos(), tan()
arcsin(), arccos(), arctan()
np.exp(), np.exp2(), np.power(3,x) #3^x
np.log(), np.log2(), np.log10()
np.expm1(x) #exp(x)-1
np.log1p(x) #log(1+x)
```

### More advanced UFuncs.
``` Python
# Defining the ouput beforehand can also speed up the process
x = np.arange(5)
y = np.empty(5)
np.multiply(x,10,out=y)

y = np.zeros(10)
np.power(2,x,out=y[::2])
```

### Aggregates, vereniging/ verzameling
``` Python
# applying UFuncs on an entire array
x = np.arange(1,6)

# .reduce()
np.add.reduce(x) #gives the sum of all elements: 15
np.multiply.reduce(x) # gives the product of all elements

# .accumulate()
np.add.accumulate(x) #gives the sum of alle elements and storing intermediate results
np.multiply.accumulate(x) # gives the product of all elements and stores intermediate results
```

### Outer Products
Pairwise multiplication/addition/etc. is done with .outer(x,x). Returns a matrix of size(x) by size(x).

### Sum, Min, Max,
Using the np.sum, np.min and np.max functions is a lot faster. Can also be called quicker via: .min(), .max(), .sum().

Can also be used on arrays. By default takes the entire array, otherwise input axis=0,1. Where the axis = 0 means the x-axis will collapse and axis = 1, means the y axis will collapse. np.

### Comparison

Numpy also has a lot of logica operators. np.equal (==), np.not_equal(!=), np.less (<), etc.

### Working with bool Arrays
np.count_nonzero(), np.sum(), np.any(), np.all()

### Bitwise logic operators
&,|,^,~ == np.bitwise_and(), np.bitwise_or(), np.bitwise_xor(), np.bitwise_not()

print("Number of days without rain: ", np.sum(insights == 0))
print("Number of days with rain: ", np.sum(insights != 0))
print("Number of days with rain <0.1: ", np.sum((insights > 0) & (insights <=0.1)))ins

Also combine these to make masks on arrays to get what insights you want.

### Masks
It is often very wise to create different masks for your data, so you can ask it alot of questions very efficiently. Consider an example of a weather dataset. One can make a mask of the summer, one can make a mask of rainy days. And after that you can ask yourself questions about rainy days in the summer very easily. data[summer & rainy] or data[rainy & ~summer]. np.max(data[summer]), etc.

### Fancy indexing
In numpy you can use fancy indexing to get subsets from your data very easily. data[array], gives you an array back with the values of your data on the index of the array for example. Of course you can also modify your data with this accordingly. data[i] = x, for indices i, and constant x.

Sometimes the modifying behaves oddly, because iterations are done immediately. x[i] += 1, is actually x[i] = x[i]+ 1. So if you have indices i, which contains the same constant c twice. One would think 2 would be added to x[c], but that's not correct, because computations are done immediately the old value of x[c] is used twice. So 1 only gets added once.

To avoid this use .at() function of numpy. np.add.at(x,i,1) = np.add.at(data,indices,value).

### Binning data
Sometime you want to split your data in several bins to see where most values are contained. Also called a histogram. Numpy can also do this manually with bins = np.linspace(lower,upper,step) and i = np.searchsorted(bins,x). Which sorts all values of x in the bins. So it returns a matrix i containing indices of the bins where the value of x is sorted, so i has the same value of x.

## Sorting Arrays
Starting a new header here, because this is an important subject in all of computer science. Sorting array algoritms are: insertion sorts, selection sorts, merge sorts, quick sorts, bubble sorts, and many many more. Sort out that data!

One can always write a sorting algorithm manually, for example, bogosort or swapsort, but NumPy has some pretty efficient sorting algoritms built in already:

np.sort(), np.argsort().

Which can also have matrices as an input. Where it will sort the colums for axis =0 and the rows for axis =1 (sorting along the axis). Keep in mind that all relations will be lost.


### Partitioning arrays
Partitioning can be done with np.partition(array, k, axis=0). Where k stands for where the partition is supposed to be. The result is the same array with the k smallest values on the left and the rest on the right.
np.argpartition() is also available.
