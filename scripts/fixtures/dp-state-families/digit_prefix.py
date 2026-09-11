from functools import cache


def count_distinct(bound):
    if type(bound) is not int or not 0 <= bound <= 10**18:
        raise ValueError("Use an integer bound from 0 through 10**18")
    digits = tuple(map(int, str(bound)))

    @cache
    def completions(position, tight, started, used):
        if position == len(digits):
            return int(started)  # The all-padding spelling is not positive.
        limit = digits[position] if tight else 9
        count = 0
        for digit in range(limit + 1):
            next_tight = tight and digit == digits[position]
            if not started and digit == 0:
                count += completions(position + 1, next_tight, False, used)
            elif not used & (1 << digit):
                count += completions(position + 1, next_tight, True, used | (1 << digit))
        return count

    return completions(0, True, False, 0)


def count_range(first, last):
    if type(first) is not int or type(last) is not int or not 0 <= first <= last <= 10**18:
        raise ValueError("Use an ordered nonnegative integer range through 10**18")
    before = count_distinct(first - 1) if first > 0 else 0
    return count_distinct(last) - before  # Zero stays excluded.


for bound in [0, 9, 99, 100, 102, 213, 999, 10**18]:
    print(bound, "=>", count_distinct(bound))
print("100 through 130:", count_range(100, 130))
print("0 through 0:", count_range(0, 0))
