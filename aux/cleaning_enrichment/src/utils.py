def split_into_chunks(text, max_length):
    if len(text) > max_length:
        texts_splits = [text[i:i + max_length]
                        for i in range(0, len(text), max_length)]
    else:
        texts_splits = [text]
    return texts_splits

def flatten(xss):
    return [x for xs in xss for x in xs]