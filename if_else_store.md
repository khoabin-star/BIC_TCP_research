\*_ 9 interval in range (1000-1000)
if (last_max_cwnd >= 500 && last_max_cwnd <= 1000 && rtt >= 1 && rtt <= 500)
{
bic_target = (72 _ last*max_cwnd) / 100 + (7 * rtt) / 100 - 13;
}
else if (last*max_cwnd >= 500 && last_max_cwnd <= 1000 && rtt >= 500 && rtt <= 1000)
{
bic_target = (75 * last*max_cwnd) / 100 + (6 * rtt) / 100 - 34;
}
else if (last*max_cwnd >= 1 && last_max_cwnd <= 250 && rtt >= 500 && rtt <= 750)
{
bic_target = (78 * last*max_cwnd) / 100 + (1 * rtt) / 100 - 8;
}
else if (last*max_cwnd >= 1 && last_max_cwnd <= 250 && rtt >= 750 && rtt <= 1000)
{
bic_target = (81 * last*max_cwnd) / 100 + (1 * rtt) / 100 - 10;
}
else if (last*max_cwnd >= 250 && last_max_cwnd <= 500 && rtt >= 500 && rtt <= 750)
{
bic_target = (75 * last*max_cwnd) / 100 + (4 * rtt) / 100 - 17;
}
else if (last*max_cwnd >= 250 && last_max_cwnd <= 500 && rtt >= 750 && rtt <= 1000)
{
bic_target = (77 * last*max_cwnd) / 100 + (3 * rtt) / 100 - 22;
}
else if (last*max_cwnd >= 1 && last_max_cwnd <= 250 && rtt >= 1 && rtt <= 250)
{
bic_target = (72 * last*max_cwnd) / 100 + (2 * rtt) / 100 - 2;
}
else if (last*max_cwnd > 1 && last_max_cwnd < 250 && rtt > 250 && rtt <= 500)
{
bic_target = (75 * last*max_cwnd) / 100 + (2 * rtt) / 100 - 5;
}
else if (last*max_cwnd > 250 && last_max_cwnd < 500 && rtt >= 1 && rtt < 250)
{
bic_target = (71 * last*max_cwnd) / 100 + (5 * rtt) / 100 - 4;
}
else
{
bic*target = (73 * last_max_cwnd) / 100 + (4 \* rtt) / 100 - 11;
}

16
if (last*max_cwnd >= 1 && last_max_cwnd <= 250 && rtt >= 500 && rtt <= 750)
{
bic_target = (781 * last*max_cwnd + 18 * rtt + -8263) / 1000;
}
else if (last*max_cwnd >= 1 && last_max_cwnd <= 250 && rtt >= 750 && rtt <= 1000)
{
bic_target = (810 * last*max_cwnd + 16 * rtt + -10234) / 1000;
}
else if (last*max_cwnd >= 250 && last_max_cwnd <= 500 && rtt >= 500 && rtt <= 750)
{
bic_target = (755 * last*max_cwnd + 42 * rtt + -17634) / 1000;
}
else if (last*max_cwnd >= 250 && last_max_cwnd <= 500 && rtt >= 750 && rtt <= 1000)
{
bic_target = (775 * last*max_cwnd + 39 * rtt + -22651) / 1000;
}
else if (last*max_cwnd >= 1 && last_max_cwnd <= 250 && rtt >= 1 && rtt <= 250)
{
bic_target = (717 * last*max_cwnd + 22 * rtt + -1963) / 1000;
}
else if (last*max_cwnd >= 1 && last_max_cwnd <= 250 && rtt >= 250 && rtt <= 500)
{
bic_target = (750 * last*max_cwnd + 20 * rtt + -5557) / 1000;
}
else if (last*max_cwnd >= 250 && last_max_cwnd <= 500 && rtt >= 1 && rtt <= 250)
{
bic_target = (711 * last*max_cwnd + 49 * rtt + -4152) / 1000;
}
else if (last*max_cwnd >= 250 && last_max_cwnd < 500 && rtt >= 250 && rtt <= 500)
{
bic_target = (734 * last*max_cwnd + 46 * rtt + -11494) / 1000;
}
else if (last*max_cwnd > 500 && last_max_cwnd <= 750 && rtt >= 1 && rtt <= 250)
{
bic_target = (711 * last*max_cwnd + 70 * rtt + -6605) / 1000;
}
else if (last*max_cwnd > 500 && last_max_cwnd <= 750 && rtt >= 250 && rtt <= 500)
{
bic_target = (730 * last*max_cwnd + 66 * rtt + -17096) / 1000;
}
else if (last*max_cwnd > 750 && last_max_cwnd <= 1000 && rtt >= 1 && rtt <= 250)
{
bic_target = (709 * last*max_cwnd + 88 * rtt + -7427) / 1000;
}
else if (last*max_cwnd > 750 && last_max_cwnd <= 1000 && rtt >= 250 && rtt <= 500)
{
bic_target = (726 * last*max_cwnd + 83 * rtt + -20780) / 1000;
}
else if (last*max_cwnd > 500 && last_max_cwnd <= 750 && rtt >= 500 && rtt <= 750)
{
bic_target = (748 * last*max_cwnd + 61 * rtt + -26156) / 1000;
}
else if (last*max_cwnd > 500 && last_max_cwnd <= 750 && rtt >= 750 && rtt <= 1000)
{
bic_target = (765 * last*max_cwnd + 57 * rtt + -33859) / 1000;
}
else if (last*max_cwnd > 750 && last_max_cwnd <= 1000 && rtt >= 500 && rtt <= 750)
{
bic_target = (742 * last*max_cwnd + 78 * rtt + -32502) / 1000;
}
else
{
bic*target = (758 * last*max_cwnd + 73 * rtt + -42731) / 1000;
}

25 interval
if (last_max_cwnd >= 251 && last_max_cwnd <= 500 && rtt >= 501 && rtt <= 750)
{
// Subregion 1
result = (755 _ last_max_cwnd + 42 _ rtt - 17634) / 1000;
}
else if (last_max_cwnd >= 251 && last_max_cwnd <= 500 && rtt >= 751 && rtt <= 1000)
{
// Subregion 2
result = (775 _ last_max_cwnd + 39 _ rtt - 22651) / 1000;
}
else if (last_max_cwnd >= 1 && last_max_cwnd <= 250 && rtt >= 1 && rtt <= 250)
{
// Subregion 3
result = (717 _ last_max_cwnd + 22 _ rtt - 1963) / 1000;
}
else if (last_max_cwnd >= 251 && last_max_cwnd <= 500 && rtt >= 1 && rtt <= 250)
{
// Subregion 4
result = (711 _ last_max_cwnd + 49 _ rtt - 4152) / 1000;
}
else if (last_max_cwnd >= 251 && last_max_cwnd <= 500 && rtt >= 251 && rtt <= 500)
{
// Subregion 5
result = (734 _ last_max_cwnd + 46 _ rtt - 11494) / 1000;
}
else if (last_max_cwnd >= 501 && last_max_cwnd <= 750 && rtt >= 1 && rtt <= 250)
{
// Subregion 6
result = (711 _ last_max_cwnd + 70 _ rtt - 6605) / 1000;
}
else if (last_max_cwnd >= 501 && last_max_cwnd <= 750 && rtt >= 251 && rtt <= 500)
{
// Subregion 7
result = (730 _ last_max_cwnd + 66 _ rtt - 17096) / 1000;
}
else if (last_max_cwnd >= 751 && last_max_cwnd <= 1000 && rtt >= 1 && rtt <= 250)
{
// Subregion 8
result = (709 _ last_max_cwnd + 88 _ rtt - 7427) / 1000;
}
else if (last_max_cwnd >= 751 && last_max_cwnd <= 1000 && rtt >= 251 && rtt <= 500)
{
// Subregion 9
result = (726 _ last_max_cwnd + 83 _ rtt - 20780) / 1000;
}
else if (last_max_cwnd >= 501 && last_max_cwnd <= 750 && rtt >= 501 && rtt <= 750)
{
// Subregion 10
result = (748 _ last_max_cwnd + 61 _ rtt - 26156) / 1000;
}
else if (last_max_cwnd >= 501 && last_max_cwnd <= 750 && rtt >= 751 && rtt <= 1000)
{
// Subregion 11
result = (765 _ last_max_cwnd + 57 _ rtt - 33859) / 1000;
}
else if (last_max_cwnd >= 751 && last_max_cwnd <= 1000 && rtt >= 501 && rtt <= 750)
{
// Subregion 12
result = (742 _ last_max_cwnd + 78 _ rtt - 32502) / 1000;
}
else if (last_max_cwnd >= 751 && last_max_cwnd <= 1000 && rtt >= 751 && rtt <= 1000)
{
// Subregion 13
result = (758 _ last_max_cwnd + 73 _ rtt - 42731) / 1000;
}
else if (last_max_cwnd >= 1 && last_max_cwnd <= 125 && rtt >= 751 && rtt <= 875)
{
// Subregion 14
result = (825 _ last_max_cwnd + 9 _ rtt - 5241) / 1000;
}
else if (last_max_cwnd >= 1 && last_max_cwnd <= 125 && rtt >= 876 && rtt <= 1000)
{
// Subregion 15
result = (842 _ last_max_cwnd + 8 _ rtt - 5736) / 1000;
}
else if (last_max_cwnd >= 126 && last_max_cwnd <= 250 && rtt >= 751 && rtt <= 875)
{
// Subregion 16
result = (787 _ last_max_cwnd + 23 _ rtt - 12520) / 1000;
}
else if (last_max_cwnd >= 126 && last_max_cwnd <= 250 && rtt >= 876 && rtt <= 1000)
{
// Subregion 17
result = (799 _ last_max_cwnd + 22 _ rtt - 13596) / 1000;
}
else if (last_max_cwnd >= 1 && last_max_cwnd <= 125 && rtt >= 251 && rtt <= 375)
{
// Subregion 18
result = (752 _ last_max_cwnd + 13 _ rtt - 2864) / 1000;
}
else if (last_max_cwnd >= 1 && last_max_cwnd <= 125 && rtt >= 376 && rtt <= 500)
{
// Subregion 19
result = (772 _ last_max_cwnd + 12 _ rtt - 3636) / 1000;
}
else if (last_max_cwnd >= 126 && last_max_cwnd <= 250 && rtt >= 251 && rtt <= 375)
{
// Subregion 20
result = (735 _ last_max_cwnd + 29 _ rtt - 5991) / 1000;
}
else if (last_max_cwnd >= 126 && last_max_cwnd <= 250 && rtt >= 376 && rtt <= 500)
{
// Subregion 21
result = (749 _ last_max_cwnd + 27 _ rtt - 7985) / 1000;
}
else if (last_max_cwnd >= 1 && last_max_cwnd <= 125 && rtt >= 501 && rtt <= 625)
{
// Subregion 22
result = (791 _ last_max_cwnd + 11 _ rtt - 4388) / 1000;
}
else if (last_max_cwnd >= 1 && last_max_cwnd <= 125 && rtt >= 626 && rtt <= 750)
{
// Subregion 23
result = (808 _ last_max_cwnd + 10 _ rtt - 4734) / 1000;
}
else if (last_max_cwnd >= 126 && last_max_cwnd <= 250 && rtt >= 501 && rtt <= 625)
{
// Subregion 24
result = (762 _ last_max_cwnd + 26 _ rtt - 9718) / 1000;
}
else if (last_max_cwnd >= 126 && last_max_cwnd <= 250 && rtt >= 626 && rtt <= 750)
{
// Subregion 25
result = (775 _ last_max_cwnd + 24 _ rtt - 11214) / 1000;
}
else
{
// Default case or error handling
// You can set a default value or handle out-of-range inputs here
result = 0; // Example default value
}

10,000 - 1000
9 intervals:
if (last*max_cwnd >= 5000 && last_max_cwnd <= 10000 && rtt >= 1 && rtt <= 500) {
bic_target = (707 * last*max_cwnd + 359 * rtt - 43844) / 1000;
} else if (last*max_cwnd >= 5000 && last_max_cwnd <= 10000 && rtt > 500 && rtt <= 1000) {
bic_target = (724 * last*max_cwnd + 338 * rtt - 160466) / 1000;
} else if (last*max_cwnd >= 1 && last_max_cwnd <= 2500 && rtt > 500 && rtt <= 750) {
bic_target = (740 * last*max_cwnd + 81 * rtt - 36522) / 1000;
} else if (last*max_cwnd >= 1 && last_max_cwnd <= 2500 && rtt > 750 && rtt <= 1000) {
bic_target = (756 * last*max_cwnd + 77 * rtt - 48668) / 1000;
} else if (last*max_cwnd > 2500 && last_max_cwnd <= 5000 && rtt > 500 && rtt <= 750) {
bic_target = (726 * last*max_cwnd + 212 * rtt - 86721) / 1000;
} else if (last*max_cwnd > 2500 && last_max_cwnd <= 5000 && rtt > 750 && rtt <= 1000) {
bic_target = (736 * last*max_cwnd + 204 * rtt - 117773) / 1000;
} else if (last*max_cwnd >= 1 && last_max_cwnd <= 2500 && rtt >= 1 && rtt <= 250) {
bic_target = (708 * last*max_cwnd + 91 * rtt - 7584) / 1000;
} else if (last*max_cwnd >= 1 && last_max_cwnd <= 2500 && rtt > 250 && rtt <= 500) {
bic_target = (724 * last*max_cwnd + 86 * rtt - 22616) / 1000;
} else if (last*max_cwnd > 2500 && last_max_cwnd <= 5000 && rtt >= 1 && rtt <= 250) {
bic_target = (705 * last*max_cwnd + 228 * rtt - 17336) / 1000;
} else {
bic*target = (715 * last*max_cwnd + 220 * rtt - 52687) / 1000;
}

16 intervals:
if (last*max_cwnd >= 5000 && last_max_cwnd <= 7500 && rtt >= 1 && rtt <= 250) {
bic_target = (705 * last*max_cwnd + 328 * rtt - 30201) / 1000;
} else if (last*max_cwnd >= 5000 && last_max_cwnd <= 7500 && rtt > 250 && rtt <= 500) {
bic_target = (714 * last*max_cwnd + 318 * rtt - 81256) / 1000;
} else if (last*max_cwnd > 7500 && last_max_cwnd <= 10000 && rtt >= 1 && rtt <= 250) {
bic_target = (703 * last*max_cwnd + 413 * rtt - 24904) / 1000;
} else if (last*max_cwnd > 7500 && last_max_cwnd <= 10000 && rtt > 250 && rtt <= 500) {
bic_target = (711 * last*max_cwnd + 402 * rtt - 89935) / 1000;
} else if (last*max_cwnd >= 5000 && last_max_cwnd <= 7500 && rtt > 500 && rtt <= 750) {
bic_target = (723 * last*max_cwnd + 308 * rtt - 131051) / 1000;
} else if (last*max_cwnd >= 5000 && last_max_cwnd <= 7500 && rtt > 750 && rtt <= 1000) {
bic_target = (731 * last*max_cwnd + 299 * rtt - 177167) / 1000;
} else if (last*max_cwnd > 7500 && last_max_cwnd <= 10000 && rtt > 500 && rtt <= 750) {
bic_target = (719 * last*max_cwnd + 391 * rtt - 153797) / 1000;
} else if (last*max_cwnd > 7500 && last_max_cwnd <= 10000 && rtt > 750 && rtt <= 1000) {
bic_target = (727 * last*max_cwnd + 380 * rtt - 213481) / 1000;
} else if (last*max_cwnd >= 1 && last_max_cwnd <= 2500 && rtt > 500 && rtt <= 750) {
bic_target = (740 * last*max_cwnd + 81 * rtt - 36522) / 1000;
} else if (last*max_cwnd >= 1 && last_max_cwnd <= 2500 && rtt > 750 && rtt <= 1000) {
bic_target = (756 * last*max_cwnd + 77 * rtt - 48668) / 1000;
} else if (last*max_cwnd > 2500 && last_max_cwnd <= 5000 && rtt > 500 && rtt <= 750) {
bic_target = (726 * last*max_cwnd + 212 * rtt - 86721) / 1000;
} else if (last*max_cwnd > 2500 && last_max_cwnd <= 5000 && rtt > 750 && rtt <= 1000) {
bic_target = (736 * last*max_cwnd + 204 * rtt - 117773) / 1000;
} else if (last*max_cwnd >= 1 && last_max_cwnd <= 2500 && rtt >= 1 && rtt <= 250) {
bic_target = (708 * last*max_cwnd + 91 * rtt - 7584) / 1000;
} else if (last*max_cwnd >= 1 && last_max_cwnd <= 2500 && rtt > 250 && rtt <= 500) {
bic_target = (724 * last*max_cwnd + 86 * rtt - 22616) / 1000;
} else if (last*max_cwnd > 2500 && last_max_cwnd <= 5000 && rtt >= 1 && rtt <= 250) {
bic_target = (705 * last*max_cwnd + 228 * rtt - 17336) / 1000;
} else {
bic*target = (715 * last*max_cwnd + 220 * rtt - 52687) / 1000;
}

25 intervals:
if (last*max_cwnd > 2500 && last_max_cwnd <= 5000 && rtt > 500 && rtt <= 750) {
bic_target = (726 * last*max_cwnd + 212 * rtt - 86721) / 1000;
} else if (last*max_cwnd > 2500 && last_max_cwnd <= 5000 && rtt > 750 && rtt <= 1000) {
bic_target = (736 * last*max_cwnd + 204 * rtt - 117773) / 1000;
} else if (last*max_cwnd >= 1 && last_max_cwnd <= 2500 && rtt >= 1 && rtt <= 250) {
bic_target = (708 * last*max_cwnd + 91 * rtt - 7584) / 1000;
} else if (last*max_cwnd >= 1 && last_max_cwnd <= 2500 && rtt > 250 && rtt <= 500) {
bic_target = (724 * last*max_cwnd + 86 * rtt - 22616) / 1000;
} else if (last*max_cwnd > 2500 && last_max_cwnd <= 5000 && rtt >= 1 && rtt <= 250) {
bic_target = (705 * last*max_cwnd + 228 * rtt - 17336) / 1000;
} else if (last*max_cwnd > 2500 && last_max_cwnd <= 5000 && rtt > 250 && rtt <= 500) {
bic_target = (715 * last*max_cwnd + 220 * rtt - 52687) / 1000;
} else if (last*max_cwnd > 5000 && last_max_cwnd <= 7500 && rtt >= 1 && rtt <= 250) {
bic_target = (705 * last*max_cwnd + 328 * rtt - 30201) / 1000;
} else if (last*max_cwnd > 5000 && last_max_cwnd <= 7500 && rtt > 250 && rtt <= 500) {
bic_target = (714 * last*max_cwnd + 318 * rtt - 81256) / 1000;
} else if (last*max_cwnd > 7500 && last_max_cwnd <= 10000 && rtt >= 1 && rtt <= 250) {
bic_target = (703 * last*max_cwnd + 413 * rtt - 24904) / 1000;
} else if (last*max_cwnd > 7500 && last_max_cwnd <= 10000 && rtt > 250 && rtt <= 500) {
bic_target = (711 * last*max_cwnd + 402 * rtt - 89935) / 1000;
} else if (last*max_cwnd > 5000 && last_max_cwnd <= 7500 && rtt > 500 && rtt <= 750) {
bic_target = (723 * last*max_cwnd + 308 * rtt - 131051) / 1000;
} else if (last*max_cwnd > 7500 && last_max_cwnd <= 10000 && rtt > 500 && rtt <= 750) {
bic_target = (719 * last*max_cwnd + 391 * rtt - 153797) / 1000;
} else if (last*max_cwnd > 7500 && last_max_cwnd <= 10000 && rtt > 750 && rtt <= 1000) {
bic_target = (727 * last*max_cwnd + 380 * rtt - 213481) / 1000;
} else if (last*max_cwnd >= 1 && last_max_cwnd <= 1250 && rtt > 750 && rtt <= 875) {
bic_target = (764 * last*max_cwnd + 54 * rtt - 32628) / 1000;
} else if (last*max_cwnd >= 1 && last_max_cwnd <= 1250 && rtt > 875 && rtt <= 1000) {
bic_target = (773 * last*max_cwnd + 52 * rtt - 36306) / 1000;
} else if (last*max_cwnd > 1250 && last_max_cwnd <= 2500 && rtt > 750 && rtt <= 875) {
bic_target = (742 * last*max_cwnd + 125 * rtt - 67344) / 1000;
} else if (last*max_cwnd > 1250 && last_max_cwnd <= 2500 && rtt > 875 && rtt <= 1000) {
bic_target = (748 * last*max_cwnd + 122 * rtt - 75931) / 1000;
} else if (last*max_cwnd >= 1 && last_max_cwnd <= 1250 && rtt > 500 && rtt <= 625) {
bic_target = (745 * last*max_cwnd + 58 * rtt - 24101) / 1000;
} else if (last*max_cwnd >= 1 && last_max_cwnd <= 1250 && rtt > 625 && rtt <= 750) {
bic_target = (755 * last*max_cwnd + 56 * rtt - 28543) / 1000;
} else if (last*max_cwnd > 1250 && last_max_cwnd <= 2500 && rtt > 500 && rtt <= 625) {
bic_target = (730 * last*max_cwnd + 131 * rtt - 48749) / 1000;
} else if (last*max_cwnd > 1250 && last_max_cwnd <= 2500 && rtt > 625 && rtt <= 750) {
bic_target = (736 * last*max_cwnd + 128 * rtt - 58352) / 1000;
} else if (last*max_cwnd > 5000 && last_max_cwnd <= 6250 && rtt > 750 && rtt <= 875) {
bic_target = (732 * last*max_cwnd + 281 * rtt - 166789) / 1000;
} else if (last*max_cwnd > 5000 && last_max_cwnd <= 6250 && rtt > 875 && rtt <= 1000) {
bic_target = (737 * last*max_cwnd + 276 * rtt - 186688) / 1000;
} else if (last*max_cwnd > 6250 && last_max_cwnd <= 7500 && rtt > 750 && rtt <= 875) {
bic_target = (730 * last*max_cwnd + 324 * rtt - 186002) / 1000;
} else if (last*max_cwnd > 6250 && last_max_cwnd <= 7500 && rtt > 875 && rtt <= 1000) {
bic_target = (734 * last*max_cwnd + 319 * rtt - 209848) / 1000;
} else {
bic_target = -1; // Default case if no range matches
}
