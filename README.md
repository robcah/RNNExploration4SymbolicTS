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
>>> df_strings = LZWStringLibrary(symbols=(10,52,3), complexity=(100,1000,3), save_csv=True,
                  symbols_range_distribution='geometrical', complexity_range_distribution='geometrical')
>>> df_strings
Processing: 9 of 9
String library saved in file: StrLib_Symb(10-52_03)_LZWc(100-1000_3)_Iters(1).csv
  nr_symbols	LZW_complexity	length	string
0	        10	           100	   139	FEBIJGAHCDAGGGAGIFECHGJDJIAFFIDCBCAAAGAADFJDGI...
0	        10	           316	   556	CFHGIABJEDAHEBJEAEJBBBEJFIAAJBIDFIGFDDABFIBBBE...
0	        10	           1000	  2213	BAEICFGJDHDBAJAHAHEJHEJJFHDDEJDEAGAFDHCJJFFHHI...
0	        23	           100	   110	WIOFJTBMLRKNCUGSDEHVQAPFEVHLLAKGWAGNPWVPCSVQHW...
0	        23	           316	   385	HJLSWKFOTRNCIAQBEUMVGDPWCQKBJECCWKGTKHNEHCMGLS...
0	        23	           1000	  1565	TFIPDNMSVECRKLOBAJUGWQHJDSCPFSNGTJURQIWWKPTWBA...
0	        52	           100	   100	QqHXdSeUIsfivYZOxrTBgthlAjwoVJaunyWKPcCmbNkERp...
0	        52	           316	   336	PjZwzitQkAoWVYsHbNDJIxrGpOElKTUvhaMmFycfBLdnSq...
0	        52	           1000	  1163	gLnemIbJGTrpNUuhxBsCOfAcjlRPWwHMYSzvEZDKadQkVo...
```
