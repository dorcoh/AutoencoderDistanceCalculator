def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


def pprint_dict(input):
    for key, value in input.items():
        print(key)
        print(value)
        print()