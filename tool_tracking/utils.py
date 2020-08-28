# TODO: might be needed later
ASCII_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def get_symbol(i):
    # https://github.com/dgasmith/opt_einsum/blob/master/opt_einsum/parser.py
    if i < len(ASCII_ALPHABET):
        return ASCII_ALPHABET[i]
    return chr(i + 140)
