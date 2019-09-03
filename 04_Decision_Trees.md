# Decision Trees

## Coding
### Calculating Entropy
    def Entropy(Events):
       from math import log
       e = 0
       for i in Events:
           e -= i/sum(Events)*log(i/sum(Events),2)
       return e 
    # Example
    ## For a bucket of four red balls and ten blue balls
    print(Entropy([4,10]))
## Warnings
1. Large depth very often causes overfitting, since a tree that is too deep, can memorize the data.
2. Small minimum samples to/per split may result in a complicated, highly branched tree, which can mean the model has memorized the data, or in other words, overfit.
