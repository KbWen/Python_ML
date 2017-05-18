# 1.避免多重分類或分類不均
# 2.會導致那個神經元被拉在兩個極端點
# 3.bias:準度，好得bias會讓預測結果高，但會造成overfitting
#   Variance:穩定性，好的variance會讓每次的預測結果相近，會造成underfitting
# 4.手上只有一棵樹，怎樣修剪都是一樣的分類
# 5.用N個暫存器對N個特徵值做編碼，每個暫存器接獨立，且只有一個有效位元。
# 6.a.再訓練時加入更多的資料。
#   b.L1,L2 Regularization：就像是個退化模型，因為我們用w去限制他過多的變動
#   c.dropout：每次訓練時隨機忽略部分神經元



from operator import itemgetter

filename = open('raw_sentences.txt','r')
sentence =[]
for line in filename.readlines():
    sentence.append(line)
for i,j in enumerate(sentence):
    sentence[i] = j.strip('\n').split(' ')
print(sentence)

def ngram_probs(S):
    dicts = {}
    for j in S:
        for i in j:
            dicts[i] = dicts.get(i,0)+1
    chsorted=sorted(dicts.items(), key=itemgetter(1), reverse=True)
    dicts2 = {}
    for (i,j) in S:
        dicts2[(i,j)]=dicts2.get((i,j),0)+1
    dicts3 = {}
    for (i,j,k) in S:
        dicts3[(i,j,k)]=dicrs3.get((i,j,k),0)+1
    return (dicts2,dicts3)
def prob3(bigram,cnt2,cnt3):
    dicts2 = {}
    for j in S:
        for i in j:
            dicts2[(i,j[i+1])]=dicts2.get((i,j[i+1]),0)+1
    dicts3 = {}
    for j in S:
        for i in j:
        dicts3[(i,j[i+1],j[i+2])]=dicrs3.get((i,j[i+1],j[i+2]),0)+1
    bigram = sorted(dicts2.items(), key=itemgetter(1), reverse=True)
    trigram = sorted(dicts3.items(), key=itemgetter(1), reverse=True)
    return (dicts2,dicts3)

def predict_max(starting):
    predic = prob3(bigram,cnt2,cnt3)
    for
