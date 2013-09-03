"""
  Implemenetation of a Naive-Bayes k-class Gibbs sampler, following
  Resnik and Hardisty 2010, "Gibbs Sampling for the Uninitiated"

  this sampler differs from the one they describe in that it allows for multiple classes,
  and in that it also marginalizes out the word-distributions theta

  author: benjamin.boerschinger@googlemail.com
  date: 02/09/13
"""

import random,math,sys
from numpy.random import multinomial,dirichlet
from numpy import exp, log, logaddexp


"""
  Variables
"""
documents = []    #set of documents, each document is just a counting dictionary
labels = []       #set of labels, 0 or 1, same index as documents
thetas = []       #the class-wise word-distributions, log-probability vectors
Ccounts = []      #two lists which give class counts
gammat = 1.0      #symmetric prior gammaTheta, 1.0 for uniform
gammaTheta = []   #prior for thetas
gammap = 1.0      #symmetric prior on gammaPi, 1.0 for uniform
gammaPi = []      #prior for class-labels
NLABELS = 2       #number of classes

wordsToIds = {}   #maps strings to ints
idsToWords = {}   #maps ints to strings
nextId = 0

"""
  Helper functions
"""
def wToId(w):
    """maps a word to an identifier"""
    global wordsToIds,idsToWords,nextId
    try:
        return wordsToIds[w]
    except KeyError:
        wordsToIds[w]=nextId
        idsToWords[nextId]=w
        nextId+=1
        return wordsToIds[w]

def incr(hm,k):
    """increments hm[k] by one, setting it to one if k is not present yet"""
    try:
        hm[k]+=1
    except KeyError:
        hm[k]=1

def vecAddDict(l,d):
    """adds a list and a sparse vector elementwise without mutating inputs"""
    res = l[:]
    for (i,c) in d.iteritems():
        res[i]+=c
    return res

def vecAddSc(l,s):
    """adds a scalar to each element in l"""
    return [x+s for x in l]

def vecAddVec(l1,l2):
    """adds two lists elementwise"""
    return [x+y for (x,y) in zip(l1,l2)]


def K():
    """number of distinct labels"""
    global NLABELS
    return NLABELS

def N():
    """number of documents"""
    return len(documents)

def V():
    """number of word types in vocabulary"""
    return len(wordsToIds.keys())

"""Sampler Code"""

def init(f): 
    """read documents from file and initialize sampler

    init initializes the sampler for the corpus in file f
    
    @param f: name of the corpus-file which has to have one document per line
    @type f: String
    """
    global documents,labels,thetas,Ccounts,pi,gammat,gammaTheta,gammaPi
    reset()
    for l in open(f):
        l=l.strip().lower().split()
        newDoc = {}
        for w in l:
            incr(newDoc,wToId(w))
        documents.append(newDoc)
    print K()
    labels = [random.randint(0,K()-1) for x in range(N())]
    gammaTheta = [gammat for x in range(V())]
    gammaPi = [gammap for x in range(K())]
    initClassCounts()
    resampleThetas()

def reset():
    """resets all the global variables"""
    global documents,labels,thetas,Ccounts,wordsToIds,idsToWords,pi,nextId
    documents = []
    labels = []
    Ccounts = []
    wordsToIds = {}
    idsToWords = {}
    nextId = 0

def initClassCounts(): 
    """given current label-assignments, redo class-counts"""
    global Ccounts,documents,labels
    Cs = [[0 for x in range(V())] for x in range(K())]
    for (d,l) in zip(documents,labels):
        Cs[l]=vecAddDict(Cs[l],d)
    Ccounts=Cs

def removeCounts(d,l):
    """remove counts for document from global counts for label"""
    global Ccounts
    for (word,counts) in d.iteritems():
        Ccounts[l][word]-=counts

def addCounts(d,l):
    """add counts for document to global counts for label"""
    global Ccounts
    for (word,counts) in d.iteritems():
        Ccounts[l][word]+=counts

def labelCounts(omit=-1):
    """count-vector for different labels, potentially omitting a single document"""
    res =  [labels.count(x) for x in range(K())]
    if omit==-1:
        return res
    else:
        res[labels[omit]]-=1
        return res

def resampleThetas():
    """resample the word-distributions"""
    global gammaTheta,Ccounts,thetas
    thetas=[]
    for i in range(K()):
        thetas.append(log(dirichlet(vecAddVec(gammaTheta,Ccounts[i]),1))[0])  #fromDirichlet(vecAddVec(gammaTheta,Ccounts[0]))

def dirichletProb(vec,prior):
    """log-probability of log-transformed probability vector under Dirichlet(prior)"""
    res = math.lgamma(sum(prior))-sum([math.lgamma(ai) for ai in prior])
    for (lp,ak) in zip(vec,prior):
        res += lp*(ak-1)
    return res

def sampleFrom(logs):
    """sample from a log-transformed distribution
    
    @param logs: array of log-probabilities
    @return: index i of sampled element with probability logs[i]
    """
    cust = 0
    flip = random.random()
    for i in range(len(logs)):
        cust += exp(logs[i])
        if flip<=cust:
            return i

def normalize(logs):
    """normalize a log-probability vector"""
    norm = logsum(logs)
    return [x-norm for x in logs]

def logsum(logs):
    """sum a list of log-probabilities"""
    tPi = max(logs)
    return tPi+math.log(sum([math.exp(i-tPi) for i in logs]))

def logProb():
    """calculate log-probability of sampler state"""
    global thetas,pi,gammaPi,Ccounts,gammat,gammaTheta,documents,labels
    #label probabilities
    res = math.lgamma(sum(gammaPi)) - math.lgamma(sum([x+y for (x,y) in zip(labelCounts(-1),gammaPi)])) + sum([math.lgamma(labelCounts(-1)[i]+gammaPi[i]) - math.lgamma(gammaPi[i]) for i in range(K())])
    #marginalized document probabilities
    for l in range(K()):
        res = res + math.lgamma(gammat*V()) - math.lgamma(sum(vecAddSc(Ccounts[l],gammat)))
        for i in range(V()):
            res = res + math.lgamma(Ccounts[l][i]+gammat)-math.lgamma(gammat)
    return res

def probLabel(x,i):
    """predictive probability of label x, excluding counts of document i"""
    global gammaPi
    counts = labelCounts(i)
    return (counts[x]+gammaPi[x])/(float(sum(counts))+sum(gammaPi))

def probD(d,l):
    """probability of document under label l, under marginalized theta"""
    global gammat,Ccounts
    sumGammaTheta = sum([x+gammat for x in Ccounts[l]])
    NA = sumGammaTheta+sum(d.values())
    res = math.lgamma(sumGammaTheta)-math.lgamma(NA)
    for (wId,wCount) in d.iteritems():
        res = res + (math.lgamma(wCount+gammat+Ccounts[l][wId])-math.lgamma(gammat+Ccounts[l][wId])) 
    return res

def topWords(l,n):
    """retrieve the top n words from distribution for label l"""
    global idsToWords,gammat,Ccounts
    norm = sum(Ccounts[l])+V()*gammat
    return ", ".join(["(%.3f,%s)"%((x[1]+gammat)/norm,x[0]) for x in [(idsToWords[x[0]],x[1]) for x in sorted(enumerate(Ccounts[l]),lambda x,y:-cmp(x[1],y[1]))[:n]] ])

def sample(iters=1000):
    global documents,labels,Ccounts,thetas,wordsToIds
    print "# iteration logProb pi"
    indices = range(N())
    for it in range(iters):
        random.shuffle(indices) #randomize order each iteration
        print "%i %f %.3f %s"%(it,logProb(),probLabel(0,-1)," ".join([str(x) for x in labels]))
        for i in indices:
            d = documents[i]
            choices = []
            removeCounts(d,labels[i])
            for li in range(K()):
                choices.append(probD(d,li)+math.log(probLabel(li,i)))
            choices = normalize(choices)
            labels[i] = sampleFrom(choices)
            addCounts(d,labels[i])

def inspectTopics(n):
    """print out the top-n words for all word-distributions"""
    for i in range(K()):
       print "#Label %s top words at end: %s"%(i,topWords(i,5))

if __name__=="__main__":
    iters = 20
    if len(sys.argv)>2:
        iters = int(sys.argv[2])
        if len(sys.argv)>3:
            NLABELS=int(sys.argv[3])
    init(sys.argv[1])
    inspectTopics(5)
    sample(iters)
    inspectTopics(5)
