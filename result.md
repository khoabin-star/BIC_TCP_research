1. (range last-max-cwnd: 1000, rtt: 1000)
   **Number of interval: 3 x 3 (9 total)**

- Exponential range:
  Average Mean Squared Error (MSE): 3.439415
  Average Mean Absolute Error (MAE): 0.803726
  Average Mean Absolute Percentage Error (MAPE): 3.495595%

  Max MSE = 28.379778 at subregion: last_max_cwnd=100-1000, rtt=100-1000
  Max MAE = 4.017299 at subregion: last_max_cwnd=100-1000, rtt=100-1000
  Max MAPE = 9.471849% at subregion: last_max_cwnd=1-10, rtt=100-1000

  Min MSE = 0.066667 at subregion: last_max_cwnd=1-10, rtt=1-10
  Min MAE = 0.222222 at subregion: last_max_cwnd=1-10, rtt=1-10
  Min MAPE = 0.130523% at subregion: last_max_cwnd=100-1000, rtt=1-10

- equal range:
  Average Mean Squared Error (MSE) across all subregions: 1.435873
  Average Mean Absolute Error (MAE) across all subregions: 0.921457
  Average Mean Absolute Percentage Error (MAPE) across all subregions: 1.422437%

  Max MSE: 2.600141 at subregion last_max_cwnd=[1, 334), rtt=[667, 1000))
  Max MAE: 1.353146 at subregion last_max_cwnd=[1, 334), rtt=[667, 1000))
  Max MAPE: 4.238200% at subregion last_max_cwnd=[1, 334), rtt=[667, 1000))

  Min MSE: 0.734401 at subregion last_max_cwnd=[334, 667), rtt=[667, 1000))
  Min MAE: 0.596732 at subregion last_max_cwnd=[334, 667), rtt=[667, 1000))
  Min MAPE: 0.121071% at subregion last_max_cwnd=[667, 1000), rtt=[1, 334))

- Adaptive method:

  Average MSE across all subregions: 1.27
  Average MAE across all subregions: 0.86
  Average MAPE across all subregions: 1.40%

  Max MSE: 2.319288 at subregion last_max_cwnd=(1, 250.75], rtt=(500.5, 1000])
  Max MAE: 1.212664 at subregion last_max_cwnd=(1, 250.75], rtt=(500.5, 1000])
  Max MAPE: 5.070245% at subregion last_max_cwnd=(1, 250.75], rtt=(500.5, 1000])

  Min MSE: 0.858360 at subregion last_max_cwnd=(750.25, 1000], rtt=(500.5, 1000])
  Min MAE: 0.673088 at subregion last_max_cwnd=(750.25, 1000], rtt=(1, 500.5])
  Min MAPE: 0.101914% at subregion last_max_cwnd=(750.25, 1000], rtt=(500.5, 1000])

**Number of interval: 4 x 4 (16 total)**

- Exponential range:
  Average Mean Squared Error (MSE): 1.420982
  Average Mean Absolute Error (MAE): 0.557534
  Average Mean Absolute Percentage Error (MAPE): 3.711138%
  Max MSE = 18.316899 at subregion: last_max_cwnd=178-1000, rtt=178-1000
  Max MAE = 3.258468 at subregion: last_max_cwnd=178-1000, rtt=178-1000
  Max MAPE = 15.104089% at subregion: last_max_cwnd=1-6, rtt=178-1000
  Min MSE = 0.060000 at subregion: last_max_cwnd=1-6, rtt=32-178
  Min MAE = 0.200000 at subregion: last_max_cwnd=1-6, rtt=1-6
  Min MAPE = 0.096339% at subregion: last_max_cwnd=178-1000, rtt=1-6

- equal range:
  Average Mean Squared Error (MSE) across all subregions: 0.751400
  Average Mean Absolute Error (MAE) across all subregions: 0.621049
  Average Mean Absolute Percentage Error (MAPE) across all subregions: 0.975970%
  Max MSE: 1.236864 at subregion last_max_cwnd=[1, 251), rtt=[750, 1000))
  Max MAE: 0.864851 at subregion last_max_cwnd=[1, 251), rtt=[251, 500))
  Max MAPE: 4.283225% at subregion last_max_cwnd=[1, 251), rtt=[750, 1000))
  Min MSE: 0.408755 at subregion last_max_cwnd=[500, 750), rtt=[251, 500))
  Min MAE: 0.391823 at subregion last_max_cwnd=[500, 750), rtt=[251, 500))
  Min MAPE: 0.076018% at subregion last_max_cwnd=[750, 1000), rtt=[251, 500))

- Adaptive method:
  Average Mean Squared Error (MSE) across all subregions: 0.70
  Average Mean Absolute Error (MAE) across all subregions: 0.60
  Average Mean Absolute Percentage Error (MAPE) across all subregions: 1.04%
  Max MSE: 1.091840 at subregion last_max_cwnd=(1, 250.75], rtt=(1, 250.75])
  Max MAE: 0.838464 at subregion last_max_cwnd=(1, 125.875], rtt=(500.5, 750.25])
  Max MAPE: 4.029431% at subregion last_max_cwnd=(1, 125.875], rtt=(750.25, 1000])
  Min MSE: 0.274336 at subregion last_max_cwnd=(125.875, 250.75], rtt=(750.25, 1000])
  Min MAE: 0.273440 at subregion last_max_cwnd=(125.875, 250.75], rtt=(750.25, 1000])
  Min MAPE: 0.092673% at subregion last_max_cwnd=(500.5, 750.25], rtt=(250.75, 500.5])

  **Number of interval: 5 \* 5 (25 total)**

  - exponential range
    Average Mean Squared Error (MSE): 0.722114
    Average Mean Absolute Error (MAE): 0.404566
    Average Mean Absolute Percentage Error (MAPE): 1.189852%
    Max MSE = 12.222946 at subregion: last_max_cwnd=251-1000, rtt=251-1000
    Max MAE = 2.761647 at subregion: last_max_cwnd=251-1000, rtt=251-1000
    Max MAPE = 4.307149% at subregion: last_max_cwnd=4-16, rtt=63-251
    Min MSE = 0.000000 at subregion: last_max_cwnd=1-4, rtt=1-4
    Min MAE = 0.000000 at subregion: last_max_cwnd=1-4, rtt=1-4
    Min MAPE = 0.000000% at subregion: last_max_cwnd=1-4, rtt=1-4

  - equal range:
    Average Mean Squared Error (MSE) across all subregions: 0.733861
    Average Mean Absolute Error (MAE) across all subregions: 0.640471
    Average Mean Absolute Percentage Error (MAPE) across all subregions: 0.761974%
    Max MSE: 1.304875 at subregion last_max_cwnd=[1, 201), rtt=[600, 800))
    Max MAE: 0.952825 at subregion last_max_cwnd=[1, 201), rtt=[600, 800))
    Max MAPE: 4.046520% at subregion last_max_cwnd=[1, 201), rtt=[800, 1000))
    Min MSE: 0.366005 at subregion last_max_cwnd=[401, 600), rtt=[800, 1000))
    Min MAE: 0.365402 at subregion last_max_cwnd=[401, 600), rtt=[800, 1000))
    Min MAPE: 0.067367% at subregion last_max_cwnd=[800, 1000), rtt=[800, 1000))

- adaptive range:
  Average Mean Squared Error (MSE) across all subregions: 0.645030
  Average Mean Absolute Error (MAE) across all subregions: 0.573183
  Average Mean Absolute Percentage Error (MAPE) across all subregions: 0.842616%

  Max MSE: 1.519424 at subregion last_max_cwnd=(750.25, 875.125], rtt=(750.25, 875.125])
  Max MAE: 1.076544 at subregion last_max_cwnd=(750.25, 875.125], rtt=(750.25, 875.125])
  Max MAPE: 4.029431% at subregion last_max_cwnd=(1, 125.875], rtt=(750.25, 1000])

  Min MSE: 0.274336 at subregion last_max_cwnd=(125.875, 250.75], rtt=(750.25, 1000])
  Min MAE: 0.273440 at subregion last_max_cwnd=(125.875, 250.75], rtt=(750.25, 1000])
  Min MAPE: 0.045811% at subregion last_max_cwnd=(875.125, 1000], rtt=(750.25, 1000])

2. (range last-max-cwnd: 10000, rtt: 1000)
   **Number of interval: 3 \* 3 (9 total)**

- Exponential range:
  Average Mean Squared Error (MSE): 83.738588
  Average Mean Absolute Error (MAE): 3.076179
  Average Mean Absolute Percentage Error (MAPE): 2.641605%
  Max MSE = 722.004788 at subregion: last_max_cwnd=464-10000, rtt=100-1000
  Max MAE = 19.357687 at subregion: last_max_cwnd=464-10000, rtt=100-1000
  Max MAPE = 9.086456% at subregion: last_max_cwnd=1-22, rtt=100-1000
  Min MSE = 0.081189 at subregion: last_max_cwnd=1-22, rtt=1-10
  Min MAE = 0.249667 at subregion: last_max_cwnd=1-22, rtt=1-10
  Min MAPE = 0.081523% at subregion: last_max_cwnd=464-10000, rtt=1-10

- Equal range:
  Average Mean Squared Error (MSE) across all subregions: 26.387291
  Average Mean Absolute Error (MAE) across all subregions: 3.955978
  Average Mean Absolute Percentage Error (MAPE) across all subregions: 1.449723%
  Max MSE: 55.244038 at subregion last_max_cwnd=[1, 3334), rtt=[667, 1000))
  Max MAE: 5.832057 at subregion last_max_cwnd=[1, 3334), rtt=[667, 1000))
  Max MAPE: 5.590364% at subregion last_max_cwnd=[1, 3334), rtt=[667, 1000))
  Min MSE: 13.875014 at subregion last_max_cwnd=[3334, 6667), rtt=[667, 1000))
  Min MAE: 2.729580 at subregion last_max_cwnd=[3334, 6667), rtt=[667, 1000))
  Min MAPE: 0.059250% at subregion last_max_cwnd=[6667, 10000), rtt=[667, 1000))

- Adaptive range:
  Average Mean Squared Error (MSE) across all subregions: 24.105028
  Average Mean Absolute Error (MAE) across all subregions: 3.762468
  Average Mean Absolute Percentage Error (MAPE) across all subregions: 1.003363%
  Max MSE: 50.916705 at subregion last_max_cwnd=(8750.125, 10000], rtt=(1, 1000])
  Max MAE: 5.996503 at subregion last_max_cwnd=(8750.125, 10000], rtt=(1, 1000])
  Max MAPE: 7.778079% at subregion last_max_cwnd=(1, 625.9375], rtt=(1, 1000])
  Min MSE: 10.373387 at subregion last_max_cwnd=(625.9375, 1250.875], rtt=(1, 1000])
  Min MAE: 2.439179 at subregion last_max_cwnd=(625.9375, 1250.875], rtt=(1, 1000])
  Min MAPE: 0.065076% at subregion last_max_cwnd=(7500.25, 8750.125], rtt=(1, 1000])

**Number of interval: 4 \* 4 (16 total)**

- Exponential range:
  Average Mean Squared Error (MSE): 37.641669
  Average Mean Absolute Error (MAE): 1.996486
  Average Mean Absolute Percentage Error (MAPE): 2.590101%
  Max MSE = 537.238170 at subregion: last_max_cwnd=1000-10000, rtt=178-1000
  Max MAE = 17.152321 at subregion: last_max_cwnd=1000-10000, rtt=178-1000
  Max MAPE = 10.050606% at subregion: last_max_cwnd=1-10, rtt=178-1000
  Min MSE = 0.066667 at subregion: last_max_cwnd=1-10, rtt=6-32
  Min MAE = 0.222222 at subregion: last_max_cwnd=1-10, rtt=1-6
  Min MAPE = 0.056098% at subregion: last_max_cwnd=1000-10000, rtt=6-32

- Equal range:
  Average Mean Squared Error (MSE) across all subregions: 14.264945
  Average Mean Absolute Error (MAE) across all subregions: 2.996752
  Average Mean Absolute Percentage Error (MAPE) across all subregions: 0.958237%
  Max MSE: 31.719553 at subregion last_max_cwnd=[1, 2501), rtt=[750, 1000))
  Max MAE: 4.455822 at subregion last_max_cwnd=[1, 2501), rtt=[750, 1000))
  Max MAPE: 5.266543% at subregion last_max_cwnd=[1, 2501), rtt=[750, 1000))
  Min MSE: 5.472127 at subregion last_max_cwnd=[2501, 5000), rtt=[500, 750))
  Min MAE: 1.788699 at subregion last_max_cwnd=[2501, 5000), rtt=[500, 750))
  Min MAPE: 0.049955% at subregion last_max_cwnd=[7500, 10000), rtt=[500, 750))

- Adaptive range:
  Average Mean Squared Error (MSE) across all subregions: 12.600162
  Average Mean Absolute Error (MAE) across all subregions: 2.714364
  Average Mean Absolute Percentage Error (MAPE) across all subregions: 0.664285%
  Max MSE: 22.120133 at subregion last_max_cwnd=(8750.125, 9375.0625], rtt=(500.5, 1000])
  Max MAE: 3.870533 at subregion last_max_cwnd=(8750.125, 9375.0625], rtt=(500.5, 1000])
  Max MAPE: 4.737101% at subregion last_max_cwnd=(1, 625.9375], rtt=(500.5, 1000])
  Min MSE: 5.573961 at subregion last_max_cwnd=(3750.625, 4375.5625], rtt=(1, 1000])
  Min MAE: 1.708324 at subregion last_max_cwnd=(3750.625, 4375.5625], rtt=(1, 1000])
  Min MAPE: 0.042854% at subregion last_max_cwnd=(6875.3125, 7500.25], rtt=(1, 1000])

**Number of interval: 5 \* 5 (25 total)**

- Exponential range:
  Average Mean Squared Error (MSE): 18.371623
  Average Mean Absolute Error (MAE): 1.430974
  Average Mean Absolute Percentage Error (MAPE): 2.757475%
  Max MSE = 373.327349 at subregion: last_max_cwnd=1585-10000, rtt=251-1000
  Max MAE = 14.421127 at subregion: last_max_cwnd=1585-10000, rtt=251-1000
  Max MAPE = 13.296613% at subregion: last_max_cwnd=1-6, rtt=251-1000
  Min MSE = 0.060000 at subregion: last_max_cwnd=1-6, rtt=1-4
  Min MAE = 0.200000 at subregion: last_max_cwnd=1-6, rtt=1-4
  Min MAPE = 0.054402% at subregion: last_max_cwnd=1585-10000, rtt=4-16

- Equal range:
  Average Mean Squared Error (MSE) across all subregions: 11.410529
  Average Mean Absolute Error (MAE) across all subregions: 2.576068
  Average Mean Absolute Percentage Error (MAPE) across all subregions: 0.702326%
  Max MSE: 36.892976 at subregion last_max_cwnd=[8000, 10000), rtt=[401, 600))
  Max MAE: 5.139001 at subregion last_max_cwnd=[8000, 10000), rtt=[401, 600))
  Max MAPE: 4.971783% at subregion last_max_cwnd=[1, 2001), rtt=[800, 1000))
  Min MSE: 2.654386 at subregion last_max_cwnd=[4001, 6000), rtt=[800, 1000))
  Min MAE: 1.195934 at subregion last_max_cwnd=[4001, 6000), rtt=[800, 1000))
  Min MAPE: 0.032501% at subregion last_max_cwnd=[4001, 6000), rtt=[800, 1000))

- Adaptive range:
  Average Mean Squared Error (MSE) across all subregions: 9.477788
  Average Mean Absolute Error (MAE) across all subregions: 2.417210
  Average Mean Absolute Percentage Error (MAPE) across all subregions: 0.438983%
  Max MSE: 14.553215 at subregion last_max_cwnd=(7500.25, 8125.1875], rtt=(1, 1000])
  Max MAE: 3.217294 at subregion last_max_cwnd=(5000.5, 5625.4375], rtt=(1, 1000])
  Max MAPE: 4.737101% at subregion last_max_cwnd=(1, 625.9375], rtt=(500.5, 1000])
  Min MSE: 0.679000 at subregion last_max_cwnd=(8437.65625, 8750.125], rtt=(500.5, 1000])
  Min MAE: 0.607000 at subregion last_max_cwnd=(8437.65625, 8750.125], rtt=(500.5, 1000])
  Min MAPE: 0.009618% at subregion last_max_cwnd=(8437.65625, 8750.125], rtt=(500.5, 1000])

3.  (range last-max-cwnd: 100,000 rtt: 1000)
    **Number of interval: 3 \* 3 (9 total)**

- Exponential range:
  Average Mean Squared Error (MSE): 2115.959284
  Average Mean Absolute Error (MAE): 15.307615
  Average Mean Absolute Percentage Error (MAPE): 1.968419%
  Max MSE = 17778.858217 at subregion: last_max_cwnd=2154-100000, rtt=100-1000
  Max MAE = 95.682359 at subregion: last_max_cwnd=2154-100000, rtt=100-1000
  Max MAPE = 7.906121% at subregion: last_max_cwnd=1-46, rtt=100-1000
  Min MSE = 0.080349 at subregion: last_max_cwnd=1-46, rtt=1-10
  Min MAE = 0.245711 at subregion: last_max_cwnd=1-46, rtt=1-10
  Min MAPE = 0.062757% at subregion: last_max_cwnd=2154-100000, rtt=1-10

- Equal range:
  Average Mean Squared Error (MSE) across all subregions: 1189.614926
  Average Mean Absolute Error (MAE) across all subregions: 26.850132
  Average Mean Absolute Percentage Error (MAPE) across all subregions: 1.069216%
  Max MSE: 1863.761190 at subregion last_max_cwnd=[66667, 100000), rtt=[1, 334))
  Max MAE: 33.221646 at subregion last_max_cwnd=[66667, 100000), rtt=[1, 334))
  Max MAPE: 4.301419% at subregion last_max_cwnd=[1, 33334), rtt=[667, 1000))
  Min MSE: 601.047232 at subregion last_max_cwnd=[1, 33334), rtt=[1, 334))
  Min MAE: 18.388375 at subregion last_max_cwnd=[1, 33334), rtt=[1, 334))
  Min MAPE: 0.050120% at subregion last_max_cwnd=[66667, 100000), rtt=[667, 1000))

- Adaptive range:
  Average Mean Squared Error (MSE) across all subregions: 1098.685660
  Average Mean Absolute Error (MAE) across all subregions: 25.301782
  Average Mean Absolute Percentage Error (MAPE) across all subregions: 0.621378%
  Max MSE: 1700.479530 at subregion last_max_cwnd=(50000.5, 75000.25], rtt=(1, 1000])
  Max MAE: 33.138885 at subregion last_max_cwnd=(50000.5, 75000.25], rtt=(1, 1000])
  Max MAPE: 5.051648% at subregion last_max_cwnd=(1, 12500.875], rtt=(1, 1000])
  Min MSE: 23.086204 at subregion last_max_cwnd=(84375.15625, 87500.125], rtt=(1, 1000])
  Min MAE: 3.443889 at subregion last_max_cwnd=(84375.15625, 87500.125], rtt=(1, 1000])
  Min MAPE: 0.005644% at subregion last_max_cwnd=(84375.15625, 87500.125], rtt=(1, 1000])

**Number of interval: 4 \* 4 (16 total)**

- Exponential range:
  Average Mean Squared Error (MSE): 1055.986206
  Average Mean Absolute Error (MAE): 9.928676
  Average Mean Absolute Percentage Error (MAPE): 1.937084%
  Max MSE = 14584.785937 at subregion: last_max_cwnd=5623-100000, rtt=178-1000
  Max MAE = 86.162093 at subregion: last_max_cwnd=5623-100000, rtt=178-1000
  Max MAPE = 8.225371% at subregion: last_max_cwnd=1-18, rtt=178-1000
  Min MSE = 0.081773 at subregion: last_max_cwnd=1-18, rtt=6-32
  Min MAE = 0.248588 at subregion: last_max_cwnd=1-18, rtt=6-32
  Min MAPE = 0.053510% at subregion: last_max_cwnd=316-5623, rtt=1-6

- Equal range:
  Average Mean Squared Error (MSE) across all subregions: 898.288722
  Average Mean Absolute Error (MAE) across all subregions: 21.970849
  Average Mean Absolute Percentage Error (MAPE) across all subregions: 0.726777%
  Max MSE: 3136.121589 at subregion last_max_cwnd=[75000, 100000), rtt=[500, 750))
  Max MAE: 44.651504 at subregion last_max_cwnd=[75000, 100000), rtt=[500, 750))
  Max MAPE: 4.142129% at subregion last_max_cwnd=[1, 25001), rtt=[750, 1000))
  Min MSE: 229.759038 at subregion last_max_cwnd=[25001, 50000), rtt=[500, 750))
  Min MAE: 10.598747 at subregion last_max_cwnd=[25001, 50000), rtt=[500, 750))
  Min MAPE: 0.040530% at subregion last_max_cwnd=[25001, 50000), rtt=[500, 750))

- Adaptive range:
  Average Mean Squared Error (MSE) across all subregions: 715.998156
  Average Mean Absolute Error (MAE) across all subregions: 19.542974
  Average Mean Absolute Percentage Error (MAPE) across all subregions: 0.406313%
  Max MSE: 2368.854411 at subregion last_max_cwnd=(82031.4296875, 82422.05078125], rtt=(1, 500.5])
  Max MAE: 44.730503 at subregion last_max_cwnd=(82031.4296875, 82422.05078125], rtt=(1, 500.5])
  Max MAPE: 5.620396% at subregion last_max_cwnd=(1, 6250.9375], rtt=(1, 1000])
  Min MSE: 18.470192 at subregion last_max_cwnd=(81250.1875, 82031.4296875], rtt=(1, 1000])
  Min MAE: 3.212690 at subregion last_max_cwnd=(81250.1875, 82031.4296875], rtt=(1, 1000])
  Min MAPE: 0.005558% at subregion last_max_cwnd=(81250.1875, 82031.4296875], rtt=(1, 1000])

**Number of interval: 5 \* 5 (25 total)**

- Exponential range:
  Average Mean Squared Error (MSE): 572.955805
  Average Mean Absolute Error (MAE): 7.498418
  Average Mean Absolute Percentage Error (MAPE): 1.905275%
  Max MSE = 10958.448060 at subregion: last_max_cwnd=10000-100000, rtt=251-1000
  Max MAE = 79.662833 at subregion: last_max_cwnd=10000-100000, rtt=251-1000
  Max MAPE = 7.579583% at subregion: last_max_cwnd=1-10, rtt=251-1000
  Min MSE = 0.066667 at subregion: last_max_cwnd=1-10, rtt=1-4
  Min MAE = 0.222222 at subregion: last_max_cwnd=1-10, rtt=4-16
  Min MAPE = 0.051009% at subregion: last_max_cwnd=10000-100000, rtt=16-63

- Equal range:
  Average Mean Squared Error (MSE) across all subregions: 793.278270
  Average Mean Absolute Error (MAE) across all subregions: 20.905701
  Average Mean Absolute Percentage Error (MAPE) across all subregions: 0.558370%
  Max MSE: 2121.513303 at subregion last_max_cwnd=[80000, 100000), rtt=[800, 1000))
  Max MAE: 36.825190 at subregion last_max_cwnd=[80000, 100000), rtt=[800, 1000))
  Max MAPE: 4.079945% at subregion last_max_cwnd=[1, 20001), rtt=[800, 1000))
  Min MSE: 79.135289 at subregion last_max_cwnd=[20001, 40001), rtt=[800, 1000))
  Min MAE: 6.948806 at subregion last_max_cwnd=[20001, 40001), rtt=[800, 1000))
  Min MAPE: 0.034467% at subregion last_max_cwnd=[20001, 40001), rtt=[800, 1000))

- Adaptive range:
  Average Mean Squared Error (MSE) across all subregions: 801.305753
  Average Mean Absolute Error (MAE) across all subregions: 20.480452
  Average Mean Absolute Percentage Error (MAPE) across all subregions: 0.273754%
  Max MSE: 3062.877622 at subregion last_max_cwnd=(82031.4296875, 82129.0849609375], rtt=(313.1875, 375.625])
  Max MAE: 47.527972 at subregion last_max_cwnd=(82031.4296875, 82129.0849609375], rtt=(313.1875, 375.625])
  Max MAPE: 5.620396% at subregion last_max_cwnd=(1, 6250.9375], rtt=(1, 1000])
  Min MSE: 18.470192 at subregion last_max_cwnd=(81250.1875, 82031.4296875], rtt=(1, 1000])
  Min MAE: 3.212690 at subregion last_max_cwnd=(81250.1875, 82031.4296875], rtt=(1, 1000])
  Min MAPE: 0.005558% at subregion last_max_cwnd=(81250.1875, 82031.4296875], rtt=(1, 1000])

Accuracy:

\*\*Scalability
Original:
(last-max-cwnd, rtt) Time #of branches # full branch # partial branch
(klee-out-0) (10, 10): 144.57 14 9 5
(klee-out-1) (50, 50): 67.74 14 9 5
(klee-out-2) (100, 100): 32.45 14
(klee-out-3) (200, 200): 82.97 14
(klee-out-4) (500, 500): 72.36 14
(klee-out-5) (1000, 1000): 118.79 14
(klee-out-6) (10000, 1000): 293.89 14

With the 9 number of interval in adaptive method (1000-1000)
(last-max-cwnd, rtt) Time # of branch # full branch # partial branch
(klee-out-7) (10, 10): 0.45 10 1 7
(klee-out-8) (50, 50): 0.45 10 1 7  
(klee-out-9) (100, 100): 0.46 10 1 7
(klee-out-10) (200, 200): 0.47 10 1 7
(klee-out-11) (500, 500): 0.86 10 7 3
(klee-out-12) (1000, 1000): 1.16 10 10 0

With 16 interval in (1000-1000)
(last-max-cwnd, rtt) Time # of branch # full branch # partial branch
(klee-out-13) (10, 10): 0.24 16 0 6
(klee-out-14) (50, 50): 0.24 16 7  
(klee-out-15) (100, 100): 0.22 10 1 7
(klee-out-16) (200, 200): 0.23 10 1 7
(klee-out-17) (500, 500): 0.61 10 7 3
(klee-out-18) (1000, 1000): 1.01 16 15 1

with 25 invertal in (1000-1000)
klee-out-22 (1000, 1000): 1.62 25 24 1
(10,000 - 1000)

with 9 intervals in (10,000-1000)
(last-max-cwnd, rtt) Time # of branch # full branch # partial branch
(klee-out-19) (10, 10): 0.72 10 9 1

with 16 intervals (10,000-1000)
(last-max-cwnd, rtt) Time # of branch # full branch # partial branch
(klee-out-20) (10, 10): 1.13 16 15 1

with 25 intervals (10,000-1000)
(last-max-cwnd, rtt) Time # of branch # full branch # partial branch
(klee-out-21) (10, 10): 1.49 25 24 1

Average Mean Squared Error (MSE) across all subregions: 818.136260
Average Mean Absolute Error (MAE) across all subregions: 20.723380
Average Mean Absolute Percentage Error (MAPE) across all subregions: 0.450439%

Max MSE: 2742.438887 at subregion last_max_cwnd=[83334, 100000), rtt=[334, 500))
Max MAE: 44.245910 at subregion last_max_cwnd=[83334, 100000), rtt=[334, 500))
Max MAPE: 3.949513% at subregion last_max_cwnd=[1, 16668), rtt=[834, 1000))

Min MSE: 65.238782 at subregion last_max_cwnd=[16668, 33334), rtt=[667, 834))
Min MAE: 6.406314 at subregion last_max_cwnd=[16668, 33334), rtt=[168, 334))
Min MAPE: 0.037817% at subregion last_max_cwnd=[33334, 50000), rtt=[168, 334))

Average MSE across all subregions: 1070.653816
Average MAE across all subregions: 25.155958
Average MAPE across all subregions: 0.339945%

Subregion with Highest MSE:
Bounds: last_max_cwnd=(81250.0, 84375.0), rtt=(32.1875, 63.375)
Highest MSE: 3302.797600

Subregion with Highest MAE:
Bounds: last_max_cwnd=(81250.0, 84375.0), rtt=(32.1875, 63.375)
Highest MAE: 49.438238

Subregion with Highest MAPE:
Bounds: last_max_cwnd=(1, 25000.5), rtt=(750.0, 1000)
Highest MAPE: 4.142129%

Subregion with Least MSE:
Bounds: last_max_cwnd=(84375.0, 87500.0), rtt=(1, 32.1875)
Least MSE: 200.120847

Subregion with Least MAE:
Bounds: last_max_cwnd=(25000.5, 50000), rtt=(500, 750.0)
Least MAE: 10.598747

Subregion with Least MAPE:
Bounds: last_max_cwnd=(84375.0, 87500.0), rtt=(1, 32.1875)
Least MAPE: 0.023425%

Z3 solver scalability time:
\*\*Scalability
Original:
(last-max-cwnd, rtt) Time #of branches # full branch # partial branch
(klee-out-28) (10, 10): 2.32
(klee-out-29) (50, 50): 11.63
(klee-out-30) (100, 100): 21.16
(klee-out-31) (200, 200): 54.08
(klee-out-32) (500, 500): 145.08
(klee-out-33) (1000, 1000): 221.62
(klee-out-34) (10000, 1000): 369.87

With the 9 number of interval in adaptive method (1000-1000)
(last-max-cwnd, rtt) Time # of branch # full branch # partial branch
(klee-out-35) (10, 10): 0.18 9 0 9
(klee-out-36) (50, 50): 0.18 9 0 9
(klee-out-37) (100, 100): 0.18 9 0 9
(klee-out-38) (200, 200): 0.18 9 0 9
(klee-out-39) (500, 500): 0.23 9 3 6
(klee-out-40) (1000, 1000): 0.29 9 8 1

16 interval
(klee-out-41) (1000, 1000): 0.45 16 15 1

25 interval
(klee-out-42) (1000, 1000): 0.62 25 24 1

9 interval
(klee-out-43) (10000, 1000): 0.33 9 8 1

16 interval
(klee-out-44) (10000, 1000): 0.42 16 15 1

25 interval
(klee-out-45) (10000, 1000): 0.60 25 24 1

---

last_max_cwnd = 1000
rtt = 1000
adaptive method only
try the following 16 intervals (last_max_cwnd, rtt):

1.  2\*2:
    Average Mean Squared Error (MSE) across all subregions: 4.326182
    Average Mean Absolute Error (MAE) across all subregions: 1.626318
    Average Mean Absolute Percentage Error (MAPE) across all subregions: 2.664426%

    Scalability:
    KLEE: done: total instructions = 144
    KLEE: done: completed paths = 4
    KLEE: done: partially completed paths = 0
    KLEE: done: generated tests = 4
    KLEE Scalability Metrics:
    States: 4
    Time(s): 0.19
    Instrs: 144
    Mem(MiB): 20.45

2.  2\*4:
    Average Mean Squared Error (MSE) across all subregions: 1.856685
    Average Mean Absolute Error (MAE) across all subregions: 1.060319
    Average Mean Absolute Percentage Error (MAPE) across all subregions: 1.758618%

    scalability:
    KLEE: done: total instructions = 252
    KLEE: done: completed paths = 8
    KLEE: done: partially completed paths = 0
    KLEE: done: generated tests = 8
    KLEE Scalability Metrics:
    States: 8
    Time(s): 0.26
    Instrs: 252
    Mem(MiB): 20.58

3.  2\*8
    Average Mean Squared Error (MSE) across all subregions: 1.178810
    Average Mean Absolute Error (MAE) across all subregions: 0.803314
    Average Mean Absolute Percentage Error (MAPE) across all subregions: 1.450334%

    scalability:
    KLEE: done: total instructions = 468
    KLEE: done: completed paths = 16
    KLEE: done: partially completed paths = 0
    KLEE: done: generated tests = 16
    KLEE Scalability Metrics:
    States: 16
    Time(s): 0.47
    Instrs: 468
    Mem(MiB): 20.85

4.  2\*16
    Average Mean Squared Error (MSE) across all subregions: 1.079323
    Average Mean Absolute Error (MAE) across all subregions: 0.764579
    Average Mean Absolute Percentage Error (MAPE) across all subregions: 1.354831%

    scalability:
    KLEE: done: total instructions = 900
    KLEE: done: completed paths = 32
    KLEE: done: partially completed paths = 0
    KLEE: done: generated tests = 32
    KLEE Scalability Metrics:
    States: 32
    Time(s): 0.82
    Instrs: 900
    Mem(MiB): 21.46

5.  4\*2:
    Average Mean Squared Error (MSE) across all subregions: 1.630499
    Average Mean Absolute Error (MAE) across all subregions: 0.995900
    Average Mean Absolute Percentage Error (MAPE) across all subregions: 1.486386%

    scalability:
    KLEE: done: total instructions = 252
    KLEE: done: completed paths = 8
    KLEE: done: partially completed paths = 0
    KLEE: done: generated tests = 8
    KLEE Scalability Metrics:
    States: 8
    Time(s): 0.27
    Instrs: 252
    Mem(MiB): 20.59

6.  4\*4:
    Average Mean Squared Error (MSE) across all subregions: 0.751400
    Average Mean Absolute Error (MAE) across all subregions: 0.621049
    Average Mean Absolute Percentage Error (MAPE) across all subregions: 0.975970%

    scalability
    KLEE: done: total instructions = 468
    KLEE: done: completed paths = 16
    KLEE: done: partially completed paths = 0
    KLEE: done: generated tests = 16
    KLEE Scalability Metrics:
    States: 16
    Time(s): 0.45
    Instrs: 468
    Mem(MiB): 20.88

7.  4\*8:
    Average Mean Squared Error (MSE) across all subregions: 0.669000
    Average Mean Absolute Error (MAE) across all subregions: 0.590850
    Average Mean Absolute Percentage Error (MAPE) across all subregions: 0.788833%

    KLEE: done: total instructions = 900
    KLEE: done: completed paths = 32
    KLEE: done: partially completed paths = 0
    KLEE: done: generated tests = 32
    KLEE Scalability Metrics:
    States: 32
    Time(s): 0.82
    Instrs: 900
    Mem(MiB): 21.38

8.  4\*16:
    Average Mean Squared Error (MSE) across all subregions: 0.638678
    Average Mean Absolute Error (MAE) across all subregions: 0.574893
    Average Mean Absolute Percentage Error (MAPE) across all subregions: 0.752350%

    KLEE: done: total instructions = 1764
    KLEE: done: completed paths = 64
    KLEE: done: partially completed paths = 0
    KLEE: done: generated tests = 64
    KLEE Scalability Metrics:
    States: 64
    Time(s): 1.56
    Instrs: 1764
    Mem(MiB): 22.47

9.  8\*2:
    Average Mean Squared Error (MSE) across all subregions: 0.857370
    Average Mean Absolute Error (MAE) across all subregions: 0.694971
    Average Mean Absolute Percentage Error (MAPE) across all subregions: 0.882202%

    KLEE: done: total instructions = 468
    KLEE: done: completed paths = 16
    KLEE: done: partially completed paths = 0
    KLEE: done: generated tests = 16
    KLEE Scalability Metrics:
    States: 16
    Time(s): 0.46
    Instrs: 468
    Mem(MiB): 20.88

10. 8\*4:
    Average Mean Squared Error (MSE) across all subregions: 0.624116
    Average Mean Absolute Error (MAE) across all subregions: 0.561496
    Average Mean Absolute Percentage Error (MAPE) across all subregions: 0.609083%

KLEE: done: total instructions = 900
KLEE: done: completed paths = 32
KLEE: done: partially completed paths = 0
KLEE: done: generated tests = 32
KLEE Scalability Metrics:
States: 32
Time(s): 0.77
Instrs: 900
Mem(MiB): 21.41

11. 8\*8:
    Average Mean Squared Error (MSE) across all subregions: 0.564410
    Average Mean Absolute Error (MAE) across all subregions: 0.524930
    Average Mean Absolute Percentage Error (MAPE) across all subregions: 0.507937%

    KLEE: done: total instructions = 1764
    KLEE: done: completed paths = 64
    KLEE: done: partially completed paths = 0
    KLEE: done: generated tests = 64
    KLEE Scalability Metrics:
    States: 64
    Time(s): 1.59
    Instrs: 1764
    Mem(MiB): 22.40

12. 8\*16:
    Average Mean Squared Error (MSE) across all subregions: 0.568938
    Average Mean Absolute Error (MAE) across all subregions: 0.536440
    Average Mean Absolute Percentage Error (MAPE) across all subregions: 0.492735%

    KLEE: done: total instructions = 3492
    KLEE: done: completed paths = 128
    KLEE: done: partially completed paths = 0
    KLEE: done: generated tests = 128
    KLEE Scalability Metrics:
    States: 128
    Time(s): 3.54
    Instrs: 3492
    Mem(MiB): 25.36

13. 16\*2:
    Average Mean Squared Error (MSE) across all subregions: 0.618001
    Average Mean Absolute Error (MAE) across all subregions: 0.566312
    Average Mean Absolute Percentage Error (MAPE) across all subregions: 0.599563%

    KLEE: done: total instructions = 900
    KLEE: done: completed paths = 32
    KLEE: done: partially completed paths = 0
    KLEE: done: generated tests = 32
    KLEE Scalability Metrics:
    States: 32
    Time(s): 0.90
    Instrs: 900
    Mem(MiB): 21.41

14. 16\*4:
    Average Mean Squared Error (MSE) across all subregions: 0.604457
    Average Mean Absolute Error (MAE) across all subregions: 0.561902
    Average Mean Absolute Percentage Error (MAPE) across all subregions: 0.465424%

    KLEE: done: total instructions = 1764
    KLEE: done: completed paths = 64
    KLEE: done: partially completed paths = 0
    KLEE: done: generated tests = 64
    KLEE Scalability Metrics:
    States: 64
    Time(s): 1.63
    Instrs: 1764
    Mem(MiB): 22.47

15. 16\*8:
    Average Mean Squared Error (MSE) across all subregions: 0.570903
    Average Mean Absolute Error (MAE) across all subregions: 0.537245
    Average Mean Absolute Percentage Error (MAPE) across all subregions: 0.417246%

    KLEE: done: total instructions = 3492
    KLEE: done: completed paths = 128
    KLEE: done: partially completed paths = 0
    KLEE: done: generated tests = 128
    KLEE Scalability Metrics:
    States: 128
    Time(s): 3.54
    Instrs: 3492
    Mem(MiB): 25.45

16. 16\*16:
    Average Mean Squared Error (MSE) across all subregions: 0.580921
    Average Mean Absolute Error (MAE) across all subregions: 0.544672
    Average Mean Absolute Percentage Error (MAPE) across all subregions: 0.415736%

    KLEE: done: total instructions = 6948
    KLEE: done: completed paths = 256
    KLEE: done: partially completed paths = 0
    KLEE: done: generated tests = 256
    KLEE Scalability Metrics:
    States: 256
    Time(s): 8.94
    Instrs: 6948
    Mem(MiB): 33.16
