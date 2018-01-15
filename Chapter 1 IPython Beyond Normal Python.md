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
