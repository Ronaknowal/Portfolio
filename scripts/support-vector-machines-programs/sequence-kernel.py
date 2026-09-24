from collections import Counter
import numpy as np


def counts(sequence, size=2):
    return Counter(sequence[index:index+size] for index in range(len(sequence)-size+1))


def kernel(left, right):
    a, b = counts(left), counts(right)
    return sum(value * b[word] for word, value in a.items())


if __name__ == '__main__':
    strings = ['ACACA', 'CACAC', 'AACCA']
    vocabulary = sorted(set().union(*(counts(word) for word in strings)))
    features = np.array([[counts(word)[piece] for piece in vocabulary] for word in strings])
    gram = np.array([[kernel(a,b) for b in strings] for a in strings])
    print('coordinates:', vocabulary)
    print('features:', features.tolist())
    print('Gram:', gram.tolist())
    print('dot products agree:', bool(np.array_equal(gram, features @ features.T)))
    print('Different positions can have the same two-mer counts:', bool(np.array_equal(features[0],features[1])))
