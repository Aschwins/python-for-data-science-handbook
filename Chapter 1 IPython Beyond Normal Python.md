# Chapter 1: IPython: Beyond Normal Python

## Keyboard Shortcuts in the IPython Shell

Ctrl-a  - Move curser to the beginning of the line
Ctrl-e  - Move curser to the end of the line

Ctrl-d  - Delete next character in line
Ctrl-k  - Cut text from the cursor to the end of line
Ctrl-u  - Cut text from beginning to line to cursor
Ctrl-y  - Yank (i.e. paster) text that was previously Cut
Ctrl-r  - Reverse-search through command history

## Miscellaneous Shortcuts

Ctrl-l  - Clear terminal screen
Ctrl-c  - Interrupt current Python command
Ctrl-d  - Exit IPython Session

## IPython Magic Commands, % and %%

### Running External Code: %Run

For example say we've created a myscript.py file with the following contents
``` Python
#-------------------
# file: myscript.py

def square(x):
  """square a number"""
  return x ** 2

for N in range(1,4):
  print(N, "squared is", square(N))  
```

You can execute this from your IPython session as follows:

```
In [42]: %run myscript.py
1 squared is 1
2 squared is 4
3 squared is 9
```

Functions defined in the script are now available as well.

## Timing Code Execution: %timeit

%timeit for a single line
%%timeit for turning it into cell magic.
%time for a cell in Jupyter

## In and Out objects
IPython stores a In and Out variable which can be used to save computation time of intensive commands. The underscore contains the previous output. Double underscore contains the output before that and so on.
A shortcut for this is the underscore aswell underscore 2 equals Out[2].

Ending with a semicolon, ;, supresses the output.

## Using Shell commands in IPython
One can acces the shell or terminal commands in IPython by using the exclamation mark! Everything behind an exclamation mark is interpreted as a shell command.

You can also pass values to and from the shell, for example contents = !ls.

The other way around, passing IPyhton objects to the shell van be done with the {varname} syntax.

Magic commands with the percentage can also be used to acces shell commands. %cd ..

##  Debugging
A lot goes wrong during programming in Python. With the %xmode one can acces different debugging methods, Plain, Context and Verbose. The default is Context.

If the traceback is not enough one can use the magic %debug for more interactive debugging.

You can also run a script interactively with the magic %run -d command. Use next to navigate.
