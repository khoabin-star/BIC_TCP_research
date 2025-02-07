#include <stdio.h>
#include <stdlib.h>

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

// https://elixir.bootlin.com/linux/latest/source/net/ipv4/tcp_cubic.c#L167
/* calculate the cubic root of x using a table lookup followed by one
 * Newton-Raphson iteration.
 * Avg err ~= 0.195%
 */
static u32 cubic_root(u64 a)
{
  u32 x, b, shift;
  /*
   * cbrt(x) MSB values for x MSB values in [0..63].
   * Precomputed then refined by hand - Willy Tarreau
   *
   * For x in [0..63],
   *   v = cbrt(x << 18) - 1
   *   cbrt(x) = (v[x] + 10) >> 6
   */
  static const u8 v[] = {
      /* 0x00 */ 0,
      54,
      54,
      54,
      118,
      118,
      118,
      118,
      /* 0x08 */ 123,
      129,
      134,
      138,
      143,
      147,
      151,
      156,
      /* 0x10 */ 157,
      161,
      164,
      168,
      170,
      173,
      176,
      179,
      /* 0x18 */ 181,
      185,
      187,
      190,
      192,
      194,
      197,
      199,
      /* 0x20 */ 200,
      202,
      204,
      206,
      209,
      211,
      213,
      215,
      /* 0x28 */ 217,
      219,
      221,
      222,
      224,
      225,
      227,
      229,
      /* 0x30 */ 231,
      232,
      234,
      236,
      237,
      239,
      240,
      242,
      /* 0x38 */ 244,
      245,
      246,
      248,
      250,
      251,
      252,
      254,
  };

  b = fls64(a);
  if (b < 7)
  {
    /* a in [0..63] */
    return ((u32)v[(u32)a] + 35) >> 6;
  }

  b = ((b * 84) >> 8) - 1;
  shift = (a >> (b * 3));

  x = ((u32)(((u32)v[shift] + 10) << b)) >> 6;

  /*
   * Newton-Raphson iteration
   *                         2
   * x    = ( 2 * x  +  a / x  ) / 3
   *  k+1          k         k
   */
  x = (2 * x + (u32)(a / ((u64)x * (u64)(x - 1))));
  x = ((x * 341) >> 10);
  return x;
}

int main()
{

  static int beta = 717; /* = 717/1024 (BICTCP_BETA_SCALE) */
  static int bic_scale = 41;

  static u32 cube_rtt_scale;
  static u32 beta_scale;
  static u64 cube_factor;

  u32 last_max_cwnd; /* last maximum snd_cwnd */ // will change based on Klee assume
  u32 cwnd;                                      // will change based on change of last_max_cwnd
  u32 bic_K;
  u32 bic_origin_point; /* origin point of bic function */

  u32 rtt, delta, bic_target; // rtt: will change based on Klee assume
  u64 offs, t, croot, current;

  if (BITS_PER_LONG != sizeof(long) * 8)
  {
    printf("BITS_PER_LONG is currently %d, please set it to the size of long that is %d\n", BITS_PER_LONG, (int)(sizeof(long) * 8));
    return 1;
  }

  // initialize some constant parameters

  // Part 1: Initilization

  beta_scale = 8 * (BICTCP_BETA_SCALE + beta) / 3 / (BICTCP_BETA_SCALE - beta);
  // beta_scale = 8 * (1024 + 717) / 3 / (1024 - 717) = 13928 / 921 = 15.122
  printf("Here is beta_scale: %d\n", beta_scale);

  cube_rtt_scale = (bic_scale * 10); /* 1024*c/rtt */
  // cube_rtt_scale = 410
  printf("Here is cube_rtt_scale: %d\n", cube_rtt_scale);

  /* 1/c * 2^2*bictcp_HZ * srtt */
  cube_factor = 1ull << (10 + 3 * BICTCP_HZ); /* 2^40 */
  // cube_factor = 2^40
  printf("Here is cube_factor: %llu\n", cube_factor);

  /* divide by bic_scale and by constant Srtt (100ms) */
  cube_factor = cube_factor / (bic_scale * 10);
  // cube_factor = 2^40 / 410
  printf("Here is cube_factor: %llu\n", cube_factor);

  last_max_cwnd = 95;
  rtt = 95;

  // Part 2: After a loss event
  cwnd = (last_max_cwnd * beta) / BICTCP_BETA_SCALE;

  printf("Here is cwnd: %d\n", cwnd);
  if (cwnd < 2)
    cwnd = 2;

  if (last_max_cwnd <= cwnd)
  {
    printf("SHOULD NOT GO INTO THIS IF 1\n");
    bic_K = 0;
    bic_origin_point = cwnd;
  }
  else
  {
    printf("Here is last_max_cwnd -cwnd: %d\n", last_max_cwnd - cwnd);
    croot = cube_factor * (last_max_cwnd - cwnd);
    printf("Here is croot: %llu\n", croot);
    bic_K = cubic_root(croot);
    printf("Here is bic_K: %d\n", bic_K);
    bic_origin_point = last_max_cwnd;
    printf("Here is bic_origin_point: %d\n", bic_origin_point);
  }
  // u64 test = 803996926LL * last_max_cwnd;
  // printf("Here is test: %llu\n", test);
  printf("Here is last_max_cwnd: %d\n", last_max_cwnd);
  // printf("Here is estimate of bic_K: %u\n", cubic_root(test));
  // Part 3: ACK Calculation
  current = 0;
  // time RTT
  current = current + rtt;

  t = current;
  t <<= BICTCP_HZ; /* change the unit from HZ to bictcp_HZ */
  t = t / HZ;
  printf("Here is t: %llu\n", t);
  if (t < bic_K) /* t - K */
  {
    offs = bic_K - t;
  }
  else
  {
    offs = t - bic_K;
    printf("SHOULD NOT GO INTO THIS IF 2\n");
  }
  printf("Here is offs: %llu\n", offs);
  delta = (cube_rtt_scale * offs * offs * offs) >> (10 + 3 * BICTCP_HZ);
  printf("Here is delta: %d\n", delta);
  if (t < bic_K) /* below origin*/
  {
    bic_target = bic_origin_point - delta;
  }
  else /* above origin*/
  {
    bic_target = bic_origin_point + delta;
    printf("SHOULD NOT GO INTO THIS IF 3\n");
  }
  // check some properties
  printf("Here is the bit-target: %d\n", bic_target);
  if (bic_target > 2 * cwnd)
  {
    printf("bic_target after one RTT is too aggressive\n");
  }

  return 0;
}