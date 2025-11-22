

with open("A2-Data/1b_benchmark.train.tokens", "r") as train_file:
    text = ""
    for line in train_file:
        text += "<START> " + line.strip() + " <STOP>"
    
    tokens = text.split()
    unigrams = {}
    bigrams = {}
    trigrams = {}

    for token in tokens:
        if token in unigrams:
            unigrams[token] += 1
        else:
            unigrams[token] = 1

    for token in tokens[:-1]:
        bigram = token + " " + tokens[tokens.index(token) + 1]
        if bigram in bigrams:
            bigrams[bigram] += 1
        else:
            bigrams[bigram] = 1

    for i in range(len(tokens) - 2):
        trigram = tokens[i] + " " + tokens[i + 1] + " " + tokens[i + 2]
        if trigram in trigrams:
            trigrams[trigram] += 1
        else:
            trigrams[trigram] = 1
    
    final_tokens = unigrams.copy() + bigrams.copy() + trigrams.copy()
    final_tokens["UNK"] = 0

    