from collections import defaultdict


class Association(object):
    def __init__(self, pairs):
        self.pairs = pairs
        self.right_children = defaultdict(set)
        self.left_children = defaultdict(set)
        for p in self.pairs:
            self.left_children[p[0]].add(p[1])
            self.right_children[p[1]].add(p[0])

    def left_set(self):
        return self.left_children.keys()

    def right_set(self):
        return self.right_children.keys()

    def __getitem__(self, i):
        return self.pairs[i]

    def __len__(self):
        return len(self.pairs)