# Import PyTorch "An open source machine learning framework"
import torch
import numpy as np
import matplotlib.pyplot as plt



def main():

    # Read in the names csv and make it all lower case
    lines = open('GoTNames.csv', 'r').read().splitlines()
    names = [l.lower() for l in lines]

    # Build 2d array - This represents all characters on each the horizontal or vertical 
    # We will populte this array with the counts
    N = torch.zeros((28, 28), dtype=torch.int32)
    charset = sorted(set(''.join(names)))

    # String to int maping
    stoi = {s:i+1 for i,s in enumerate(charset)}
    stoi['.'] = 0

    # int to String maping
    itos = {i:s for s,i in stoi.items()}

    # Count all bigrams in all names - Populate N
    for n in names:
        # Add a start and end special chartacter to visualize the bigrams
        # . is like a 'pre-start' character and 'post-end' character
        chars = ['.']+list(n)+['.']
        for ch1, ch2 in zip(chars, chars[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1

    P = get_probabilites(N)
    
    # Pretty much the same method, ex2 is just more efficent since it doesnt recalculate probabilites
    generate_names_ex(N, 10, itos)
    generate_names_ex2(P, 10, itos)
    # We get the same results which shows we did it correctly

    nnll_ex(P,names,stoi)

    # Can check chance of single name like :
    nnll_ex(P,["nelson"],stoi)

    # However if we pick a name like: [zkroni] 
    # We will get inf because P(k|z) = 0
    nnll_ex(P,["zkroni"],stoi)

    # To fix this we will add smoothing. We do this by adding 1 to every value in our matrix so no probability = 0.
    # Since one is relativly small relativly it shouldnt impact the modle but will fix this issue
    N = N+1
    P = get_probabilites(N)
    nnll_ex(P,["zkroni"],stoi)








##### CREATE NEURAL NETWORK ######


# Create the training set of bigrams (x,y)
# xs = input (first letter). ys =  prediction (secton letter)
xs, ys = [] ,[]

for n in names[:1]:
    # Add a start and end special chartacter to visualize the bigrams
    # . is like a 'pre-start' character and 'post-end' character
    chars = ['.']+list(n)+['.']
    for ch1, ch2 in zip(chars, chars[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        print(ch1,ch2)
        xs.append(ix1)
        ys.append(ix2)
        
xs = torch.tensor(xs)
ys = torch.tensor(ys)

# Doesnt make sense to just stick an int into a NN so we use one hot encoding.
# Each Vector becomes an input to the NN
import torch.nn.functional as F
# Need to cast to float so we can pass into NN
xenc = F.one_hot(xs, num_classes=27).float()
print(xenc)



























# Prints the Normalized Negative Log Likelihood (nnll) which is a measure of how good our model is
def nnll_ex(P, names, stoi):

    # How can we see how good our modle is?
    # Lets look at some bigrams from names in the training set. If our probability array (our model) has high 
    #   prababilites for these bigrams, that is a good sign that it will more often predict that that bigram 
    #   will come next correctly, at least for these bigrams in the training set.
    for n in names[:3]:
        chars = ['.']+list(n)+['.']
        for ch1, ch2 in zip(chars, chars[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            prob = P[ix1, ix2]
            print(f'{ch1}{ch2}: {prob*100:.2f}')
    


    # To get an estimation of how good all these probabilites are people use the 'likelihood' which is really just the product
    #   of all these probabilites. However since all these are between 0-1 that will be a very small number so instead we use 
    #   the log likelihood
    log_likelihood = 0.0
    count = 0
    for n in names:
        chars = ['.']+list(n)+['.']
        for ch1, ch2 in zip(chars, chars[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            prob = P[ix1, ix2]
            logprob = torch.log(prob)
            # Because log(a*b*c) = log(a) + log(b) + log(c):
            log_likelihood += logprob
            count += 1
    print(f'{log_likelihood=}')
    # In a loss function though low is good and here low is bad so instead we use negative log likelihood (nll)
    nll = -log_likelihood
    print(f'{nll=}')
    # Last step is to normalize so it doesnt get too large, so we take an average
    nnll = nll/count
    print(f'{nnll=}')

# Well commented function that explains how we generate the names
def generate_names_ex(N, num_names, itos):

    # Generator : This makes it so rand is reproduceable in consecutive runs or accross computers
    g = torch.Generator().manual_seed(2147483647)

    # N[0, :] same as N[0]  ==>  Zero'th Row (first row), and all cols. 
    p = N[0].float()
    # Create a probability vector. This is the P(what letter comes next given this is the first letter) 
    #   ie P('a' | '.'), P('b' | '.') ...
    p = p / p.sum()
    p

    # PLOT HERE TEST THIS
    plot(N, itos)

    # Sample from diustriubution p
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    # Bassically what that ^ is doing is sampling from the distribution we created (represented by the chart).
    #   By looking in the first row N[0] we are looking at the probability of each letter being the first letter
    #   of a name given that there was no previous character. 
    #   Here that result has given us "r" which makes sense as it is the fourth most likley letter to start a name.
    print("First sampled letter = " + str(itos[ix]))

    # Now get the next letter, except instead of sampeling the probebility for the first letter (N[0]), we need to
    #   sample the probability distribuition of the previous letter ix (here that is r)
    p1 = N[ix].float()
    p1 = p1 / p1.sum()
    p1

    # Sample from diustriubution p1
    ix = torch.multinomial(p1, num_samples=1, replacement=True, generator=g).item()
    print("Second sampled letter = " + str(itos[ix]))

    print("Do it in a loop to get a name:")
    out = []
    ix = 0 # First row is starting character
    while True:
        p = N[ix].float()
        p = p / p.sum()
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

    # Generator : This makes it so rand is reproduceable in consecutive runs or accross computers
    g = torch.Generator().manual_seed(2147483647)

    print("Do the loop in a loop to get multiple names:")
    # Do this 10 times:
    for i in range(num_names): 

        out = []
        ix = 0 # First row is starting character
        while True:
            p = N[ix].float()
            p = p / p.sum()
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
        print(''.join(out))

# More efficient function that generate names, See the get_probabilites function to see why it is more efficent 
def generate_names_ex2(P, num_names, itos):

    # Generator : This makes it so rand is reproduceable in consecutive runs or accross computers
    g = torch.Generator().manual_seed(2147483647)

    # If we did that right we should now be able to use P in place of p = p / p.sum() in 
    #   the generate_names_ex() example and get the same results
    for i in range(num_names): 

        out = []
        ix = 0 # First row is starting character
        while True:

            p = P[ix]
            #p = N[ix].float()
            #p = p / p.sum()
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
        print(''.join(out))

# Generate the probability distribution for each of the 28 charcters.
def get_probabilites(N):
    # The generate_names_ex() function is inefficent since we keep re computing the distributions. 
    # There are only 28 distributions we need to ever know so lets just calculate them and save them off:
    P = N.float()

    # If we do P.sum we would be summing over the entire 2D array. We just want to sum of each row:
    #   Here the first param is 0,1 for which way we want to sum, vertically or horizontally. 
    #   (1 is horizontal so across the row)
    #   Keep dim just retains the demtion of the array (2D, even tho now we only need 1D)
    Psums = P.sum(1, keepdim=True)

    # To determine if you can do binary operations on two arrays you need to see if they are broadcastable:
    #   https://pytorch.org/docs/stable/notes/broadcasting.html
    # Here we have:
    # P:     28, 28
    # Psums: 28, 1
    # So its Broadcastable!
    P = P / Psums
    return P

# Should Plot but doesnt IDK why, it works in the notebook.
def plot(N, itos):
    plt.figure(figsize=(16,16))
    plt.imshow(N, cmap='Blues')

    # NOT SURE IF THE 0 and 1 below are correct, might be flipped (Shouldnt matter here cause its a 
    # square matrix but in the future...)
    for i in range(N.size(0)):
        for j in range(N.size(1)):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
    plt.axis('off')
    plt

# Start with a bigram language model 
def bigram_model_visual_ex(names):
    
    #Create Dict b to store bigram counts
    b = {}
    for n in names[:3]:
        # Add a start and end special chartacter to visualize the bigrams
        # <S> is like a 'pre-start' character and <E> is like a 'post-end' character
        chars = ['<S>']+list(n)+['<E>']
        for ch1, ch2 in zip(chars, chars[1:]):
            bigram = (ch1, ch2)
            b[bigram] = b.get(bigram, 0) + 1
            print(ch1, ch2)

# Very Simple tensor example
def short_tensor_ex():
    #Short Tensor Example:
    a = torch.zeros((3, 5), dtype=torch.int32)
    print(a)

    a[1,3] += 1
    print(a)

# Quick Example on how to sample a distribution:
def sample_distribution_ex():

    # Generator Example. This makes it so rand is reproduceable in consecutive runs or accross computers
    g = torch.Generator().manual_seed(2147483647)
    p_temp = torch.rand(3, generator=g)
    p_tmep = p_temp / p_temp.sum() # Normalize / Create Probability Distribution
    print(p_temp)
    # Sample from diustriubution p_temp
    print(torch.multinomial(p_temp, num_samples=100, replacement=True, generator=g))


if __name__ == "__main__":
    main()