from fractions import Fraction

def is_terminating_fraction(frac: Fraction) -> bool:
    """
    判断一个Fraction对象表示的分数在化简后是否为有限小数，
    条件是化简后的分母只含有质因数 2 和 5.
    """
    denom = frac.denominator
    for p in [2, 5]:
        while denom % p == 0:
            denom //= p
    return denom == 1

# print("通过完整判断（检查化简后分母因数）:")
cnt = 0
cnt1 = 0
# for i in range(5000):
#     # 由于 122.88 = 3072/25, 则 i/122.88 = (25*i)/3072
#     f = Fraction(25 * i, 3072)
#     if is_terminating_fraction(f):
#         # 输出分数和对应的小数值
#         print(f"{i}/122.88 = {f}  ->  {float(f)}")
#         cnt = cnt +1

print("\n-----------------------\n")
print("利用数学条件(i 可被3整除)简化判断:")
for i in range(1, 4096):
    # 0 特殊处理（0 无论如何都是有限小数）
    if i == 0 or i % 3 == 0:
        f = Fraction(25 * i, 3072)
        print(f"{i}/122.88 = {f}  \t  {float(f)} us,\t {1000/float(f)} kHz")
        cnt1 = cnt1 + 1
print(f"cnt1: {cnt1}, cnt: {cnt}")
