# Chapter 3: Data manipulation with Pandas

## Pandas Series:
Defining a series with pandas usually is of the form pd.Series(data, index=index). Where one can define the data in several ways. It can be done by defining a dict, where indices are still being folowed ordinally. One can also input data and specify an index where it only takes the data wherefore indices are specified.

## Pandas DataFrame
While a pandas series is a one dimensional array with indices the pandas DataFrame is a little upgrade. It's the two dimensional array with indices .index and .columnds.

## Pandas Index
Pandas dataframe have an index which is an interesting object by itself. One can define one by pd.Index([1,2,3,4]). Or by calling it from a dataframe with .index. Index objects behave a lot like numpy arrays, although they're innerchangeble. This makes sure data is not easily scrambled.
