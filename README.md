# RNNExploration4SymbolicTS

Library to explore the hyper-paramenter space in RNNS

This repository contains a library for the creation of strings of determinate complexity using
LZW compressing method as base. It also cointains the tools for the exploration of the 
hyperparameter space of commonly used RNNS as well as novel ones.

### Prerequisites
Pandas 1.2.3, NumPy 1.19.2, TensorFlow 2.4.1, and TextDistance 4.2.0

## Example

```python
>>> import sys
>>> sys.path.append('./Code')
>>> from LZWStringGenerator import *
>>> from RNNExploration4SymbolicTS import *
>>> df_strings = LZWStringLibrary(symbols=3, complexity=[3, 9])
>>> df_strings
```
Processing: 2 of 2
| nr_symbols | LZW_complexity | length  | string |
| ----------:| --------------:| -------:| ------:|
| 0 | 3 | 3 | 3 | BCA |
nr_symbols  LZW_complexity  length        string
0        3               3       3           BCA
1	       3	             9	    12	ABCBBCBBABCC
```
>>> df_iters = pd.DataFrame()
>>> for i, string in enumerate(df_strings['string']):
>>>     kwargs = df_strings.iloc[i,:-1].to_dict()
>>>     seed_string = df_strings.iloc[i,-1]
>>>     df_iter = RNN_Iteration(seed_string, iterations=2, architecture='LSTM', **kwargs)
>>>     df_iter.loc[:, kwargs.keys()] = kwargs.values()
>>>     df_iters = df_iters.append(df_iter)
>>> df_iter.reset_index(drop=True, inplace=True)
...
>>> df_iters.reset_index(drop=True, inplace=True)
>>> df_iters
jw	dl	total_epochs	seq_test	seq_forecast	total_time	nr_symbols	LZW_complexity	length
0	1.000000	1.0	12	ABCABCABCA	ABCABCABCA	2.685486	3	3	3
1	1.000000	1.0	14	ABCABCABCA	ABCABCABCA	2.436733	3	3	3
2	0.657143	0.5	36	CBBCBBABCC	AABCABCABC	3.352712	3	9	12
3	0.704762	0.4	36	CBBCBBABCC	ABCBABBBBB	3.811584	3	9	12
```
