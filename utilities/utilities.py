def write_vocabulary(filename, vocabulary):
    with open(filename, 'w') as f:
        for token in vocabulary:
            print(token, file=f)