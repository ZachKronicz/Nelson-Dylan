import torch
import numpy as np
import matplotlib.pyplot as plt

def read_names(file_path):
    lines = open(file_path, 'r').read().splitlines()
    names = [l.lower() for l in lines]
    return names

def build_charset(names):
    return sorted(set(''.join(names)))

def build_stoi_itos(charset):
    stoi = {s: i + 1 for i, s in enumerate(charset)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos

def build_bigram_count_matrix(names, stoi):
    N = torch.zeros((28, 28), dtype=torch.int32)
    for n in names:
        chars = ['.'] + list(n) + ['.']
        for ch1, ch2 in zip(chars, chars[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1
    return N

def compute_prob_matrix(N):
    P = (N + 1).float()
    Psums = P.sum(1, keepdim=True)
    P = P / Psums
    return P

def generate_names(P, itos, num_names):
    g = torch.Generator().manual_seed(2147483647)

    for i in range(num_names): 
        out = []
        ix = 0
        while True:
            p = P[ix]
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
        print(''.join(out))

def negative_log_likelihood(names, P, stoi):
    log_likelihood = 0.0
    count = 0
    for n in names:
        chars = ['.'] + list(n) + ['.']
        for ch1, ch2 in zip(chars, chars[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            prob = P[ix1, ix2]
            logprob = torch.log(prob)
            log_likelihood += logprob
            count += 1
    nll = -log_likelihood
    return nll, nll / count

def test_name(name, P, stoi, itos):
    chars = ['.'] + list(name) + ['.']
    log_likelihood = 0.0
    count = 0
    for ch1, ch2 in zip(chars, chars[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        count += 1
        print(f'{ch1}{ch2}: {prob*100:.2f} {logprob:.4f}')
    nll = -log_likelihood
    print(f'{nll=}')
    print(f'{nll/count=}')

def main():
    names = read_names('GoTNames.csv')
    charset = build_charset(names)
    stoi, itos = build_stoi_itos(charset)
    N = build_bigram_count_matrix(names, stoi)
    P = compute_prob_matrix(N)

    generate_names(P, itos, 10)

    nll, avg_nll = negative_log_likelihood(names, P, stoi)
    print(f'{nll=}')
    print(f'{avg_nll=}')


if __name__ == "__main__":
    main()