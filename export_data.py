# Original final1 function
def final1(last_max_cwnd, rtt, x=10):
    return (last_max_cwnd - (410 * ((cubic_root((1 << (10 + 3 * x)) // 410 * (last_max_cwnd - (717 * last_max_cwnd // 1024))) - ((1 << 10) * rtt // 1000)) ** 3) >> (10 + 3 * x)))

# Cubic root function (from earlier)
def cubic_root(a):
    v = [
        0, 54, 54, 54, 118, 118, 118, 118, 123, 129, 134, 138, 143, 147, 151, 156,
        157, 161, 164, 168, 170, 173, 176, 179, 181, 185, 187, 190, 192, 194, 197, 199,
        200, 202, 204, 206, 209, 211, 213, 215, 217, 219, 221, 222, 224, 225, 227, 229,
        231, 232, 234, 236, 237, 239, 240, 242, 244, 245, 246, 248, 250, 251, 252, 254
    ]
    b = fls64(a)
    if b < 7:
        return (v[a] + 35) >> 6
    b = ((b * 84) >> 8) - 1
    shift = (a >> (b * 3))
    x = ((v[shift] + 10) << b) >> 6
    x = (2 * x + (a // (x * (x - 1))))
    x = (x * 341) >> 10
    return x

# Function to find the last set bit in a 64-bit integer
def fls64(x):
    if x == 0:
        return 0
    return __fls(x) + 1

def __fls(word):
    num = BITS_PER_LONG - 1
    if BITS_PER_LONG == 64:
        if not (word & (~0 << 32)):
            num -= 32
            word <<= 32
    if not (word & (~0 << (BITS_PER_LONG - 16))):
        num -= 16
        word <<= 16
    if not (word & (~0 << (BITS_PER_LONG - 8))):
        num -= 8
        word <<= 8
    if not (word & (~0 << (BITS_PER_LONG - 4))):
        num -= 4
        word <<= 4
    if not (word & (~0 << (BITS_PER_LONG - 2))):
        num -= 2
        word <<= 2
    if not (word & (~0 << (BITS_PER_LONG - 1))):
        num -= 1
    return num

# Constants
BITS_PER_LONG = 64

# Generate data and write to file
def generate_data_file(filename="output.csv"):
    with open(filename, "w") as file:
        # Write header
        file.write("last_max_cwnd,rtt,result\n")
        
        # Iterate over last_max_cwnd and rtt
        for last_max_cwnd in range(1, 10001):
            for rtt in range(1, 1001):
                result = final1(last_max_cwnd, rtt)
                # Write data to file
                file.write(f"{last_max_cwnd},{rtt},{result}\n")

# Run the function to generate the data file
generate_data_file()