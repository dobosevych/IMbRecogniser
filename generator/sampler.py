import random

class Sampler:
    def __init__(self, size=10):
        self.size = size

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < self.size:
            self.index += 1
            return "".join(random.choice('AFDT') for i in range(65))
        else:
            raise StopIteration

    def __len__(self):
        return self.size

if __name__ == "__main__":
    sampler = Sampler(10)
    for sequence in sampler:
        print(sequence)

