# Most_probable_number_MPN_for_Python
The most probable number (MPN) is a statistical method to estimate bacterial cell counts from a 10-fold dilution series. This repository contains functions to calculate the MPN estimates and confidence intervals.

# Requirements
The following modules need to be installed to use functions from this repository:
- numpy
- pandas
- scipy

# How to use
Import the function "calculate_mpn" from mpn.py and call the function giving your data (pandas.DataFrame), the number of replicates per dilution step (int), and a list of column names (list[str]) for (1) experiment identifier, (2) inoculum used in g or ml, and (3) the number of positive outcomes as arguments.

Make sure to specify

# Example code

```
import pandas as pd
from mpn import calculate_mpn

dict_example = {
    "experiment": [1, 1, 1, 2, 2, 2, 3, 3, 3],
    "inoculum_amounts": [0.1, 0.01, 0.001, 0.1, 0.01, 0.001, 0.1, 0.01, 0.001],
    "positives": [5, 2, 0, 5, 0, 0, 3, 1, 0]
}

df_example = pd.DataFrame(dict_example)

df_mpn = calculate_mpn(df_example, 5, ["experiment", "inoculum_amounts", "positives"])
print(df_mpn)
```
