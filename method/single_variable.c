double pow(double, double);

/* The math function your loader will pick up */
double tcp_function(double x1)
{
  return pow(x1, 2) + 1;
}
