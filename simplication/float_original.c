#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

#define BICTCP_BETA_SCALE 1024 /* Scale factor beta calculation*/
#define BICTCP_HZ 10           /* BIC HZ 2^10 = 1024 */
#define HZ 1000                /* 1ms per HZ  */
#define BITS_PER_LONG 64

static int fls(int x)
{
  int r = 32;

  if (!x)
    return 0;
  if (!(x & 0xffff0000u))
  {
    x <<= 16;
    r -= 16;
  }
  if (!(x & 0xff000000u))
  {
    x <<= 8;
    r -= 8;
  }
  if (!(x & 0xf0000000u))
  {
    x <<= 4;
    r -= 4;
  }
  if (!(x & 0xc0000000u))
  {
    x <<= 2;
    r -= 2;
  }
  if (!(x & 0x80000000u))
  {
    x <<= 1;
    r -= 1;
  }
  return r;
}

static int __fls(unsigned long word)
{
  int num = BITS_PER_LONG - 1;

#if BITS_PER_LONG == 64
  if (!(word & (~0ul << 32)))
  {
    num -= 32;
    word <<= 32;
  }
#endif
  if (!(word & (~0ul << (BITS_PER_LONG - 16))))
  {
    num -= 16;
    word <<= 16;
  }
  if (!(word & (~0ul << (BITS_PER_LONG - 8))))
  {
    num -= 8;
    word <<= 8;
  }
  if (!(word & (~0ul << (BITS_PER_LONG - 4))))
  {
    num -= 4;
    word <<= 4;
  }
  if (!(word & (~0ul << (BITS_PER_LONG - 2))))
  {
    num -= 2;
    word <<= 2;
  }
  if (!(word & (~0ul << (BITS_PER_LONG - 1))))
    num -= 1;
  return num;
}

#if BITS_PER_LONG == 32
static int fls64(u64 x)
{
  u32 h = x >> 32;
  if (h)
    return fls(h) + 32;
  return fls(x);
}
#elif BITS_PER_LONG == 64
static int fls64(u64 x)
{
  if (x == 0)
    return 0;
  return __fls(x) + 1;
}
#else
#error BITS_PER_LONG not 32 or 64
#endif

int main()
{

  double beta = 717.0; /* = 717/1024 (BICTCP_BETA_SCALE) */
  double bic_scale = 41.0;

  double cube_rtt_scale;
  double beta_scale;
  double cube_factor;

  double last_max_cwnd; /* last maximum snd_cwnd */ // will change based on Klee assume
  double cwnd;                                      // will change based on change of last_max_cwnd
  double bic_K;
  double bic_origin_point; /* origin point of bic function */

  double rtt, delta, bic_target; // rtt: will change based on Klee assume
  double offs, t, croot, current;

  // initialize some constant parameters

  last_max_cwnd = 10;
  rtt = 10;
  double x = 10; // this is BICTCP_HZ that will change [1-10]

  cube_factor = pow(2, 40) / 410;
  bic_K = cbrt((pow(2, 40) / 410.0 - 1.00) * (last_max_cwnd - 717.0 * last_max_cwnd / 1024.0));
  t = rtt * pow(2, 10) / 1000.0;
  delta = 410.0 * pow((cbrt((pow(2, 40) / 410.0 - 1.00) * (last_max_cwnd - 717.0 * last_max_cwnd / 1024.0)) - (rtt * pow(2, 10) / 1000.0)), 3) / pow(2, 40);
  printf("Here is the cube-factor : %f\n", cube_factor);
  printf("Here is bic_k: %f\n", bic_K);
  printf("Here is t: %f\n", t);
  printf("Here is delta: %f\n", delta);
  // simplified bic_target
  double bic_target_1 = last_max_cwnd - (410.0 * pow((cbrt((pow(2, 10 + 3 * x) / 410.0 - 1.00) * (last_max_cwnd - 717.0 * last_max_cwnd / 1024.0)) - (rtt * pow(2, 10) / 1000.0)), 3) / pow(2, 10 + 3 * x));

  // double bic_target_2 = last_max_cwnd - (205.00 * pow((929 * cubic_root(last_max_cwnd) - (1.024 * rtt)), 3) / 549755813888.00);

  // check some properties
  printf("Here is the bit-target_1: %.2f\n", bic_target_1);
  // printf("Here is the bit-target_2: %.2f\n", bic_target_2);
  if (bic_target > 2 * cwnd)
  {
    printf("bic_target after one RTT is too aggressive\n");
  }

  return 0;
}