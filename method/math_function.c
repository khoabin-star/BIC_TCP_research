/* Optional: declare pow so pycparser knows its signature */
double pow(double, double);
double cbrt(double);

/* The math function your loader will pick up */
double tcp_function(double x1, double x2)
{
  return x1 - 0.4 * pow(cbrt(0.75 * x1) - x2 / 1000.0, 3);
}
