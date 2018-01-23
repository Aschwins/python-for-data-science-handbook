# Chapter 3: Data manipulation with Pandas

## Pandas Series:
Defining a series with pandas usually is of the form pd.Series(data, index=index). Where one can define the data in several ways. It can be done by defining a dict, where indices are still being folowed ordinally. One can also input data and specify an index where it only takes the data wherefore indices are specified.

## Pandas DataFrame
While a pandas series is a one dimensional array with indices the pandas DataFrame is a little upgrade. It's the two dimensional array with indices .index and .columnds.

## Pandas Index
Pandas dataframe have an index which is an interesting object by itself. One can define one by pd.Index([1,2,3,4]). Or by calling it from a dataframe with .index. Index objects behave a lot like numpy arrays, although they're innerchangeble. This makes sure data is not easily scrambled.

The cool thing about indices is that it helps you merge, compare en aggregate datasets together alot better. If one has two different indices: indA and indB for example, one can take &, the intersection, |, the union or ^ the symmetric difference between the two indexes.

These operations are also available via object methods: indA.intersection(indB), for example.

## Pandas Columns
Pandas columns can be accessed via data['column'], or via attribute style acces: data.column.

New columns can be forged with one line via operations. data['column3'] = data['column1']/data['column2']

## DataFrame as a two-dimensional array
Just use data.values. Sometimes it's convenient to look at the transposed form of the data via data.T,

For array style indexing one has to use data.iloc[:i,:j] for numerical indexing and data.loc['rows','columns'] for name indexing. One can use a hybrid of the two with data.ix[]

With the indexers we can always use the same methods we use with numpy to acces subset of the data. Masks, slicing, facy indexing, etc.

## Ufuncs: Index Preservation
Using np ufuncs on DataFrames or Series, preserves indices and just changes the values, like you would expect...

When using operations on 2 Series or Dataframes it adds/substracts/divides, etc on the indices. It take the union of both indexes and if one is missing in one of the matrices it produces a NaN. By calling the appropriate object methods one can play around with the results. For example: A.add(B, fill_value = 0), let's a NaN count as a zero instead.

The same thing happens with columns.

So a small example: If one takes the subset of a dataset, halfrow = df.iloc[0,::2], only the first row with half of the values. And one tries df - halrow. We get a result! Pandas knows all the column names so substracts what is given and produces NaN's for what is not!

This means pandas is awesome! The data will always maintain its context, and silly alignments mistakes are in the past!

## Handling missing data
Pandas recognizes two characters as missing data: 'None', the python standard which is recognized as an object. The np.nan: 'NaN', acronym for Not a Number which is used by all IEEE programming languages. Since numpy uses the standard floating type format for a missing number this supports fast computations.

Aggregates over 'None' result in error as python can only recognize it as a python object, but since NaN is recognized as a floating point, computations can be done. They're not always usefull since, min/max/sum with only only NaN value will result in NaN. If you want to ignore the nan one can use nansum, nanmin, nanmax.

## Operating on Null values.
isnull(), dropna(), notnull(), fillna()
