import matplotlib.pyplot as plt
import numpy as np
# import data
train_data = open('PLA_train.txt','r')
pla_data = train_data.read()
pla_data = pla_data.split('\n')
nums1 = 0
# figure
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(-0.5,0.5)
ax.set_ylim(-0.5,0.5)
# data type
for i in pla_data:
    pla_data[nums1] = i.split('\t')
    pla_data[nums1] = [float(x) for x in pla_data[nums1]]
    pla_data[nums1] = [(pla_data[nums1][0], pla_data[nums1][1]),
     pla_data[nums1][2]]
    if pla_data[nums1][1] == 1:
        ax.plot(pla_data[nums1][0][0],pla_data[nums1][0][1],'bo')
    elif pla_data[nums1][1] == -1:
        ax.plot(pla_data[nums1][0][0],pla_data[nums1][0][1],'rx')
    nums1 += 1
pla_data = np.asarray(pla_data)
print(pla_data)
train_data.close()
# naive LS PLA
def LS_pla(datas):
    w = np.zeros(2)
    error = 1
    while error:
        error = 0
        for x,s in datas:
            x = np.array(x)
            y = w.T.dot(x)
            try:
                ax.lines.remove(lines[0])
            except:
                pass
            xline = np.linspace(-0.5,0.5)
            lines = ax.plot(xline, -xline*w[0]/w[1])
            plt.pause(0.1)
            if np.sign(y) != np.sign(s):
                w += s * x
                error = 1
        if not error:
            break
    return w

W = LS_pla(pla_data)
print(W)
plt.ion()
plt.show()
