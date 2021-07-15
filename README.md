# RNNExploration4SymbolicTS

Library to explore the hyper-paramenter space in RNNS

This repository contains a library for the creation of strings of determinate complexity using
LZW compressing method as base. It also cointains the tools for the exploration of the 
hyperparameter space of commonly used RNNS as well as novel ones.

### Prerequisites
Pandas 1.2.3, NumPy 1.19.2, TensorFlow 2.4.1, and TextDistance 4.2.0

## Example

```python
>>> path = './Code'
>>> sys.path.append(path)
>>> from LZWStringGenerator import *
>>> df_strings = LZWStringLibrary(symbols=3, complexity=10)
>>> df_strings
Processing: 1 of 1
  nr_symbols	LZW_complexity	length	string
0	   3	              10	    12	  ABCACBBBCAAA
```
