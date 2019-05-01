class Generator:
    def __init__(self, truth, reader):
        self.truth = truth
        self.it = 0
        self.reader=reader

    def __iter__(self):
        return self

    def __next__(self):
        if self.it < len(self.truth):
            subject = self.truth[self.it]
            self.it += 1
            content = []
            for path in subject[2]:
                content.extend(self.reader(path))
            return subject[0], subject[1], content
        else:
            raise StopIteration

    def __len__(self):
        return len(self.truth)

    def reset(self):
        self.it = 0