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
isnull(), dropna(), notnull(), fillna(). with parameters how='any', 'all'; thresh=3, axis = 'rows', 'columns'. We can also fill in the NaN values with fillna(), method = 'bfill', 'ffill', np.mean(data), axis=0. Easy.

## Hierarchical indexing
Hierarchical indexing is used to store higher (than 2) dimensional data in a pandas dataframe. Hierachical indexing can be recognized by pandas dataframes who have a MultiIndex() instead of a regular Index. Creating MultiIndex can be done in different ways. index = pd.MultiIndex() or just defining and index with two dimensions in the index parameter in pandas. For example:
``` python
df = pd.DataFrame(np.random.rand(4, 2),
                  index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                  columns=['data1', 'data2'])
```
Gives a 4 by 2 matrix with indices: a1,a2,b1,b2 (multilayer). Here the work of creating a MultiIndex is done in the background.

If you pass a dictionary with appropriate tuples as keys, Pandas will automatically recognize this and use MultiIndex by default. So data = {(ind1,ind2): value, (ind1,ind2): value, ...} pd.Series(data)

Nevertheless it's sometimes usefull to create a MultiIndex explicitly. There are several methods for this:

``` python
pd.MultiIndex.from_arrays([[a,a,b,b],[1,2,1,2]])
pd.MultiIndex.from_tuples([(a,1),(a,2),(b,1),(b,2)])
pd.MultiIndex.from_product([[a,b],[1,2]]) #here is takes the cartesian product.
pd.MultiIndex(levels==[[a,b],[1,2]], labels =[[0,0,1,1],[0,1,0,1]])
```

Any of the above objects can be passed as the index argument when creating a Series or Dataframe, or be passed to reindex method of a existing Series of DataFrame.

Sometimes it is convenient to name the levels of the MultiIndex. This can be accomplished by passing the names argument to any of the above MultiIndex constructors, or by setting the names attribute of the index after the fact:
pop.index.names = ['state', 'year']. This way all the indices are aggregated in one name or group, this can be usefull to keep track of what all the indices mean!

All of the above is described for indices and indexes for rows, but since the rows and columns in a DataFrame are entirely symmetrical all of the above can also be applied for the columns of a Dataframe.

With all these indices in place there are a ton of ways to acces the data. For fancy slicing it's wise to use: pd.IndexSlice.

## Rearranging Multi-indices
Many of the MultiIndex slicing operations will fail if the index is not sorted! So use .sort_index(). Now you can slice away all you want!

Another way to rearrange hierarchical data is to turn the index labels into columns; this can be accomplished with the reset_index method. Calling this on the population dictionary will result in a DataFrame with a state and year column holding the information that was formerly in the index. For clarity, we can optionally specify the name of the data for the column representation.

And the other way around van be done with the set_index method of the DataFrame, which returns a multiply indexed DataFrame.

## Index setting and resetting
If one wants to reset all indices as columns:

``` python
pop_flat = pop.reset_index(name='population')
```

But normally the data is already in this format and you want to change it to Multiindex. This can be done with set_index

``` python
pop_flat.set_index(['state', 'year'])
```

## Data aggregations on Multi indices
One can use aggregations across indices for calculating mean, sum, max, min, etc.
``` python
data.mean(level='index')
data.mean(axis=1, level='type')
```

# Combining Datasets: Concat and Append
Remember:
``` python
np.concatenate([a,b,c], axis=0)
```

About the same in Pandas:

``` python
pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False,
keys=None, levels=None, names=None, verify_integrity=False,
copy=True)
```

Where objs = [obj1,obj2,...]. Where verify_integrity=True, will verify if indices overlap. Where ignore_index=True will ignore overlapping indices and just make new ones. Where keys=['obj1', 'obj2'] will create a MultiIndex so overlapping indices are impossible. Where join='outer' means it will take the union of both datasets and produce NaN values for columns or indices which aren't in both datasets. join='inner' will take the interesection of both datasets and will remove columns and indices which arent in both datasets. If you wan't to specify yourself what rows and columns will be kept, one can do this with join_axes = [obj1.columns]. Where in this case it will only take the columns of the first object.

By definition the concatenation takes place row-wise with the DataFrame (axis=0).

Because merging datasets together is so common on can use the object function df1.append(df2) for convenience. .append on a pandas dataframe does not alter the original dataframe, but just makes a copy.

## Combining datasets: Merge & Join

While concatenating two datasets is just taking the union or the intersection it is often desired to combine two datasets. If two datasets have some values in common one does not want to add another row or column for both of these entries. You want to merge them! Enter pd.merge()

The signature is as follows:

``` python
pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)
```
Where on specifies the column on which both DataFrames have to merge. This only works if both dataframes have this column.
Where left_on, right_on specify columns on which you want to merge if both datasets don't have the same column-name. This makes merging possible in al sorts of datasets.

One can also take a dataframe df, and join it with another df2 with df.join(df2, left_index=True, right_on = 'name') for example.

The how is very important in merges and join. The default how is 'inner' so it will take the intersection. If one does not want to leave several variables begin you should use 'outer' to take the union of both datasets!

Sometimes two datasets have the same column name with different data in it. Specifying on where you want to merge automatically creates a suffix ```_x, _y```

df.query()

## Aggregation and Grouping

Pandas has the same aggregation methods, sum, max, min, etc. as the numpy library for arrays. It also has a very usefull .describe() method which shows several of these aggregations. Other usefull aggregations are count(), first(), last(), mean(), median(), std(), var(), prod()

## GroupBy: Split, Apply, Combine

* The split step involves breaking up and grouping a DataFrame depending on the value of the specified key.
* The apply step involves computing some function, usually an aggregate, transformation or filtering withing the individual groups.
* The combine step merges the results of these operations into an output array.

.groupby('key').min(), max() etc. Lets you compute aggregation alot quicker over several groups found in the data set.

.aggregate() is also an aggregation method, but a lot more versital. It can take strings, lists, dicts and functions as input. df.groupby('key').aggregate(['min', np.median, max]). No problem.

Filtering like the english word already says is used in databases to filter out data you don't need. filter() takes a function as argument where you can filter out, what you don't want.

One can also tranform the data by using the df.groupby('key').transform(lambda x: x - np.mean(x)), or inputting any other transformation function.

You can input a list, a dict or a key in a groupby

## Vectorized String Operations

With the Pandas library one can call on a dataframe containing strings with df.str.[method]. Using the .str.[TAB] one can call onto all vectorized string methods.

### Miscellaneous methods
Finally, there are some miscellaneous methods that enable other convenient operations:

| Method | Description |
|--------|-------------|
| ``get()`` | Index each element |
| ``slice()`` | Slice each element|
| ``slice_replace()`` | Replace slice in each element with passed value|
| ``cat()``      | Concatenate strings|
| ``repeat()`` | Repeat values |
| ``normalize()`` | Return Unicode form of string |
| ``pad()`` | Add whitespace to left, right, or both sides of strings|
| ``wrap()`` | Split long strings into lines with length less than a given width|
| ``join()`` | Join strings in each element of the Series with passed separator|
| ``get_dummies()`` | extract dummy variables as a dataframe |

### Methods using regular expressions

In addition, there are several methods that accept regular expressions to examine the content of each string element, and follow some of the API conventions of Python's built-in ``re`` module:

| Method | Description |
|--------|-------------|
| ``match()`` | Call ``re.match()`` on each element, returning a boolean. |
| ``extract()`` | Call ``re.match()`` on each element, returning matched groups as strings.|
| ``findall()`` | Call ``re.findall()`` on each element |
| ``replace()`` | Replace occurrences of pattern with some other string|
| ``contains()`` | Call ``re.search()`` on each element, returning a boolean |
| ``count()`` | Count occurrences of pattern|
| ``split()``   | Equivalent to ``str.split()``, but accepts regexps |
| ``rsplit()`` | Equivalent to ``str.rsplit()``, but accepts regexps |

## Working with time Series

We have time stamps, time intervals & series, and time deltas. Where time stamps equal an exact point in time, time intervals equal a time interval with a fixed beginning and endpoint and time deltas and exact time length.

Userfull packages for time series object are the **datetime**, python native package. The **dateutil**, 3rd party package, and if you're ready for some advanced time zone problems try **pytz**.

All of the above time crunching methods are python native so relatively slow compared to a language like C. This is why the numpy-team added a 64bit time variable called datetime64. Which can be called with:

```
date = np.array('2017-4-7', datetype = np.datetime64) #or
np.datetime64('2017-4-7 12:00')
```

### Pandas, always winning
Now of course our lovely pandas library also made its own addition to the numpy datetime64 with for example:
```
pd.to_datetime("23rd of july 2016")
pd.to_timedelta(np.arange(12), 'D')
pd.DatetimeIndex(['2015-7-4', '2015-8-5', '2016-3-4'])
```
Where the first creates a date out of a string where it is able to. The second creates a time delta and the third one creates a pandas index which can be used in Series and DataFrame pandas objects.

216
