import random,sys

words = ["sport","baseball","basketball","money","foul","finance","gold","customs"]
topic1 = [0.3,0.25,0.25,0.0,0.2,0.0,0.0,0.0]
topic2 = [0.0,0.0,0.0,0.1,0.1,0.1,0.3,0.4]

def sample(t):
    res = 0
    r = random.random()
    for i in range(len(t)):
        res+=t[i]
        if r<=res:
            return i

def generate(t,n):
    global words
    res = []
    for i in range(n):
        res.append(words[sample(t)])
    return " ".join(res)

if __name__=="__main__":
    t1 = int(sys.argv[1])
    t2 = int(sys.argv[2])
    for l in range(t1):
        print generate(topic1,10)
    for l in range(t2):
        print generate(topic2,10)
