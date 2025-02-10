1. (range last-max-cwnd: 1000, rtt: 1000)
   **Number of interval: 3 x 3 (9 total)**

- Exponential range: Average Mean Squared Error (MSE) across all subregions: 3.420898
- equal range: Average Mean Squared Error (MSE) across all subregions: 1.025913
- Adaptive method: Average Mean Squared Error (MSE) across all subregions: 0.955664

**Number of interval: 4 x 4 (16 total)**

- Exponential range: Average Mean Squared Error (MSE) across all subregions: 0.654593
- equal range: Average Mean Squared Error (MSE) across all subregions: 0.478370
- Adaptive method: Average Mean Squared Error (MSE) across all subregions: 0.478370

2. (range last-max-cwnd: 10000, rtt: 1000)
   **Number of interval: 3 \* 3 (9 total)**

- Exponential range: Average Mean Squared Error (MSE) across all subregions: 80.775948
- Equal range: Average Mean Squared Error (MSE) across all subregions: 26.287462
- Adaptive range: Average Mean Squared Error (MSE) across all subregions: 21.955418

**Number of interval: 4 \* 4 (16 total)**

- Exponential range: Average Mean Squared Error (MSE) across all subregions: 49.671925
- Equal range: Average Mean Squared Error (MSE) across all subregions: 13.109311
- Adaptive range: Average Mean Squared Error (MSE) across all subregions: 13.109311

**Number of interval: 5 \* 5 (25 total)**

- Exponential range: Average Mean Squared Error (MSE) across all subregions: 19.929133
- Equal range: Average Mean Squared Error (MSE) across all subregions: 8.783567
- Adaptive range: Average Mean Squared Error (MSE) across all subregions: 7.055983

3.  (range last-max-cwnd: 100,000 rtt: 1000)
    **Number of interval: 3 \* 3 (9 total)**

- Exponential range: Average Mean Squared Error (MSE) across all subregions: 1732.853410
- Equal range: Average Mean Squared Error (MSE) across all subregions: 1029.581423
- Adaptive range: Average Mean Squared Error (MSE) across all subregions: 703.360857

  **Number of interval: 4 \* 4 (16 total)**

- Exponential range: Average Mean Squared Error (MSE) across all subregions: not count
- Equal range: Average Mean Squared Error (MSE) across all subregions: 728.956338
- Adaptive range: Average Mean Squared Error (MSE) across all subregions: 728.956338

Accuracy:

\*\*Scalability
Original:
(last-max-cwnd, rtt) Time
(klee-out-0) (10, 10): 144.57
(klee-out-1) (50, 50): 67.74
(klee-out-2) (100, 100): 32.45
(klee-out-3) (200, 200): 82.97
(klee-out-4) (500, 500): 72.36
(klee-out-5) (1000, 1000): 118.79
(klee-out-6) (10000, 1000): 293.89

With the 25 number of interval in adaptive method
(last-max-cwnd, rtt) Time
(klee-out-7) (10, 10): 0.45
(klee-out-8) (50, 50): 0.45
(klee-out-9) (100, 100): 0.46
(klee-out-10) (200, 200): 0.47
(klee-out-11) (500, 500): 0.86
(klee-out-12) (1000, 1000): 1.16
