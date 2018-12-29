

def load_all_folds():
    folds = []
    for i in range(1, 6):
        folds.append(load_fold(i))

    return folds


def load_fold(num=1):
    path_base = f'ml-100k/u{num}.base'
    path_test = f'ml-100k/u{num}.test'
    train = {}
    test = {}

    with open(path_base, 'r') as f:
        lines = f.readlines()

        # Line contains: user_id | movie_id | rating | timestamp.
        for line in lines:
            # Ignore timestamp.
            split = line.split('\t')[:-1]

            if split[0] in test:
                train[split[0]].append((split[1], split[2]))
            else:
                train[split[0]] = [(split[1], split[2])]

    with open(path_test, 'r') as f:
        lines = f.readlines()

        # Line contains: user_id | movie_id | rating | timestamp.
        for line in lines:
            # Ignore timestamp
            split = line.split('\t')[:-1]

            if split[0] in test:
                test[split[0]].append((split[1], split[2]))
            else:
                test[split[0]] = [(split[1], split[2])]

    # Check that the test and validation set is disjoint.
    for key, value in test.items():
        if key in train:
            values = train[key]
            for pair in values:
                # Remove pair from train set if overlapping.
                if pair in value:
                    train[key].remove(pair)

    return train, test
