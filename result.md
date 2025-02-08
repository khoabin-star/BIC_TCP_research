Number of interval: 3 \* 3 (9 total) (range last-max-cwnd: 1000, rtt: 1000)

1. Exponential range

Subregion: last*max_cwnd=[1, 10), rtt=[1, 10)
Equation: result = 0.700000 * last*max_cwnd + -0.000000 * rtt + 0.500000
Mean Squared Error (MSE): 0.066667
R² Score: 0.980000

---

Subregion: last*max_cwnd=[1, 10), rtt=[10, 100)
Equation: result = 0.700000 * last*max_cwnd + -0.000000 * rtt + 0.500000
Mean Squared Error (MSE): 0.066667
R² Score: 0.980000

---

Subregion: last*max_cwnd=[1, 10), rtt=[100, 1000)
Equation: result = 0.892000 * last*max_cwnd + 0.001303 * rtt + -0.509769
Mean Squared Error (MSE): 0.139592
R² Score: 0.974887

---

Subregion: last*max_cwnd=[10, 100), rtt=[1, 10)
Equation: result = 0.699222 * last*max_cwnd + 0.003889 * rtt + 0.565542
Mean Squared Error (MSE): 0.090084
R² Score: 0.999727

---

Subregion: last*max_cwnd=[10, 100), rtt=[10, 100)
Equation: result = 0.707973 * last*max_cwnd + 0.010622 * rtt + -0.175238
Mean Squared Error (MSE): 0.166427
R² Score: 0.999508

---

Subregion: last*max_cwnd=[10, 100), rtt=[100, 1000)
Equation: result = 0.788073 * last*max_cwnd + 0.010070 * rtt + -3.912565
Mean Squared Error (MSE): 1.204958
R² Score: 0.997179

---

Subregion: last*max_cwnd=[100, 1000), rtt=[1, 10)
Equation: result = 0.700503 * last*max_cwnd + 0.048852 * rtt + 0.123394
Mean Squared Error (MSE): 0.186831
R² Score: 0.999994

---

Subregion: last*max_cwnd=[100, 1000), rtt=[10, 100)
Equation: result = 0.704985 * last*max_cwnd + 0.064156 * rtt + -2.574285
Mean Squared Error (MSE): 0.525385
R² Score: 0.999984

---

Subregion: last*max_cwnd=[100, 1000), rtt=[100, 1000)
Equation: result = 0.744141 * last*max_cwnd + 0.055573 * rtt + -22.165586
Mean Squared Error (MSE): 28.341476
R² Score: 0.999247

---

Average Mean Squared Error (MSE) across all subregions: 3.420898

2. equal range

Subregion: last*max_cwnd=[1, 334), rtt=[1, 334)
Equation: result = 0.720648 * last*max_cwnd + 0.026763 * rtt + -3.246247
Mean Squared Error (MSE): 1.557478
R² Score: 0.999676

---

Subregion: last*max_cwnd=[1, 334), rtt=[334, 667)
Equation: result = 0.760117 * last*max_cwnd + 0.023419 * rtt + -8.701936
Mean Squared Error (MSE): 1.682461
R² Score: 0.999685

---

Subregion: last*max_cwnd=[1, 334), rtt=[667, 1000)
Equation: result = 0.796370 * last*max_cwnd + 0.020146 * rtt + -12.576170
Mean Squared Error (MSE): 2.049513
R² Score: 0.999651

---

Subregion: last*max_cwnd=[334, 667), rtt=[1, 334)
Equation: result = 0.713940 * last*max_cwnd + 0.059375 * rtt + -6.708543
Mean Squared Error (MSE): 0.735974
R² Score: 0.999845

---

Subregion: last*max_cwnd=[334, 667), rtt=[334, 667)
Equation: result = 0.740450 * last*max_cwnd + 0.053927 * rtt + -18.140532
Mean Squared Error (MSE): 0.680944
R² Score: 0.999866

---

Subregion: last*max_cwnd=[334, 667), rtt=[667, 1000)
Equation: result = 0.765677 * last*max_cwnd + 0.048714 * rtt + -27.277848
Mean Squared Error (MSE): 0.637159
R² Score: 0.999883

---

Subregion: last*max_cwnd=[667, 1000), rtt=[1, 334)
Equation: result = 0.711820 * last*max_cwnd + 0.084229 * rtt + -9.319681
Mean Squared Error (MSE): 0.686427
R² Score: 0.999855

---

Subregion: last*max_cwnd=[667, 1000), rtt=[334, 667)
Equation: result = 0.734298 * last*max_cwnd + 0.077662 * rtt + -25.851312
Mean Squared Error (MSE): 0.625994
R² Score: 0.999876

---

Subregion: last*max_cwnd=[667, 1000), rtt=[667, 1000)
Equation: result = 0.755891 * last*max_cwnd + 0.071368 * rtt + -39.638554
Mean Squared Error (MSE): 0.577267
R² Score: 0.999892

---

Average Mean Squared Error (MSE) across all subregions: 1.025913

3. Adaptive method

Subregion 1: last*max_cwnd=(500, 1000), rtt=(1, 500)
Equation: result = 0.718483 * last*max_cwnd + 0.076617 * rtt + -13.183615
Mean Squared Error (MSE): 2.438356

---

Subregion 2: last*max_cwnd=(500, 1000), rtt=(500, 1000)
Equation: result = 0.752697 * last*max_cwnd + 0.067361 * rtt + -34.215614
Mean Squared Error (MSE): 2.186469

---

Subregion 3: last*max_cwnd=(1, 250.5), rtt=(500, 750.0)
Equation: result = 0.780918 * last*max_cwnd + 0.017828 * rtt + -8.263009
Mean Squared Error (MSE): 0.921692

---

Subregion 4: last*max_cwnd=(1, 250.5), rtt=(750.0, 1000)
Equation: result = 0.809554 * last*max_cwnd + 0.015673 * rtt + -10.234025
Mean Squared Error (MSE): 1.133953

---

Subregion 5: last*max_cwnd=(250.5, 500), rtt=(500, 750.0)
Equation: result = 0.754720 * last*max_cwnd + 0.042111 * rtt + -17.633765
Mean Squared Error (MSE): 0.340856

---

Subregion 6: last*max_cwnd=(250.5, 500), rtt=(750.0, 1000)
Equation: result = 0.775071 * last*max_cwnd + 0.038628 * rtt + -22.651427
Mean Squared Error (MSE): 0.323848

---

Subregion 7: last*max_cwnd=(1, 250.5), rtt=(1, 250.5)
Equation: result = 0.717113 * last*max_cwnd + 0.022178 * rtt + -1.962991
Mean Squared Error (MSE): 0.720182

---

Subregion 8: last*max_cwnd=(1, 250.5), rtt=(250.5, 500)
Equation: result = 0.750159 * last*max_cwnd + 0.020128 * rtt + -5.556932
Mean Squared Error (MSE): 0.748500

---

Subregion 9: last*max_cwnd=(250.5, 500), rtt=(1, 250.5)
Equation: result = 0.711481 * last*max_cwnd + 0.049375 * rtt + -4.152291
Mean Squared Error (MSE): 0.385102

---

Subregion 10: last*max_cwnd=(250.5, 500), rtt=(250.5, 500)
Equation: result = 0.733540 * last*max_cwnd + 0.045714 * rtt + -11.494149
Mean Squared Error (MSE): 0.357685

---

Average Mean Squared Error (MSE) across all subregions: 0.955664

Number of interval: 4 \* 4 (16 total) (range last-max-cwnd: 10000, rtt: 1000)

1. Exponential range
   Average Mean Squared Error (MSE) across all subregions: 49.671925

2. Equal range
   Average Mean Squared Error (MSE) across all subregions: 13.109311

3. Adaptive range
   Average Mean Squared Error (MSE) across all subregions: 13.109311

Test case 3 (5 \* 5) (10,000 - 1000)

1. Equal range
   Average Mean Squared Error (MSE) across all subregions: 8.783567
2. Adaptive range
   Average Mean Squared Error (MSE) across all subregions: 7.055983
