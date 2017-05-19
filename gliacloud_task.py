
    for j in S:
        for i in j:
            dicts[i] = dicts.get(i,0)+1
    chsorted=sorted(dicts.items(), key=itemgetter(1), reverse=True)
    dicts2 = {}
    for listn in S:
        for i in range(len(listn)-1):
             dicts2[(listn[i],listn[i+1])]= dicts2.get((listn[i],listn[i+1]),0)+1
    dicts3 = {}
    for listn in S:
        for i in range(len(listn)-2):
             dicts3[(listn[i],listn[i+1],listn[i+2])]=dicts3.get((listn[i],listn[i+1],listn[i+2]),0)+1

    bigramsorted=sorted(dicts2.items(), key=itemgetter(1), reverse=True)
    trigramsorted=sorted(dicts3.items(), key=itemgetter(1), reverse=True)
    return (bigramsorted, trigramsorted)
print(ngram_probs(sentence))

# probability
# 看了好久，還是不懂得怎麼在輸入就先給bigram and trigram 機率
def prob3(bigram):
    proba2_list,proba3_list = ngram_probs(sentence)
    dicts = dict()
    cnt2_sum = 0
    cnt3_sum = 0
    cnt2_num = 0
    cnt3_num = 0
    for i in range(len(proba2_list)):
        cnt2_sum +=  proba2_list[i][1]
        if proba2_list[i][0] == bigram:
            cnt2_num += proba2_list[i][1]
    cnt2 = cnt2_num/cnt2_sum
    for i in range(len(proba3_list)):
        cnt3_sum +=  proba3_list[i][1]
        if bigram[0] == proba3_list[i][0][0] and bigram[1] == proba3_list[i][0][1]:
            cnt3_num = proba3_list[i][1]
            dicts[proba3_list[i][0][2]] = dicts.get(i,0)+ (cnt3_num/cnt3_sum)/cnt2
    return(dicts)
print(prob3(('we','are')))
# 有邊界沒考慮到
def predict_max(starting):
    dicts_prob = prob3(starting)
    gramsorted=sorted(dicts_prob.items(), key=itemgetter(1), reverse=True)
    return gramsorted[0][0]

print(predict_max(starting=('we','are')))
