<b>BIC_TCP Project<b>

- Explaination of each file:
- `test_cubic.c`: is a given original file about BIC_TCP problem
- `int_original.c`: this file calculates the bic_target using the condensed or simplified intermediate variables in integer form
- `float_original.c`: this file calculates the bic_target using the condensed or simplified intermediate variables in floating form
- `export_data.py`: this file helps to generate data (input: last_max_cwnd, rtt; output: bic_target)
- `output.csv`: this is data file created by the `export_data.py`
- `equal_range.py`: this file is equal range method when trying to divide the range of the piecewise linear approximation
- `exponential_range.py`: this file is exponential range method when trying to divide the range of the piecewise linear approximation
- `adaptive_range.py`: this file is adaptive range method when trying to divide the range of the piecewise linear approximation.
