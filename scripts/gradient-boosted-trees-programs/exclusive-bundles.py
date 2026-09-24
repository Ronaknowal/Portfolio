def pack(row):
    if len(row) != 3 or any(value not in [0, 1, 2] for value in row):
        raise ValueError("three bin IDs from {0,1,2} required")
    if sum(value != 0 for value in row) > 1:
        raise ValueError("conflict: exact bundling requires exclusivity")
    return sum(2 * index + value for index, value in enumerate(row) if value)


def unpack(code):
    if code not in range(7):
        raise ValueError("bundle code out of range")
    return [code - 2 * index if 2 * index < code <= 2 * index + 2 else 0 for index in range(3)]


rows = [[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 2]]
codes = [pack(row) for row in rows]
print("codes:", codes)
print("decoded:", [unpack(code) for code in codes])
try:
    pack([1, 1, 0])
except ValueError as error:
    print(error)
