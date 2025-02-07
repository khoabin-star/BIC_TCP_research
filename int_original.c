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
  last_max_cwnd = 10;
  rtt = 10;   //[1-100]
  int x = 10; // this is BICTCP_HZ that will change [1-10]

  u32 final1 = (last_max_cwnd - (410 * (((u64)cubic_root((1ULL << (10 + 3 * x)) / 410 * (last_max_cwnd - (717 * last_max_cwnd / 1024))) - ((1 << 10) * rtt / 1000)) * ((u64)cubic_root((1ULL << (10 + 3 * x)) / 410 * (last_max_cwnd - (717 * last_max_cwnd / 1024))) - ((1 << 10) * rtt / 1000)) * ((u64)cubic_root((1ULL << (10 + 3 * x)) / 410 * (last_max_cwnd - (717 * last_max_cwnd / 1024))) - ((1 << 10) * rtt / 1000))) >> (10 + 3 * x)));

  printf("Here is the bic_target at int %d\n", final1);

  // printf("Here is the bit-target: %d\n", bic_target);
  if (bic_target > 2 * cwnd)
  {
    printf("bic_target after one RTT is too aggressive\n");
  }

  return 0;
}
