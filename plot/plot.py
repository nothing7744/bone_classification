from matplotlib import pyplot as plt
import pandas as pd
# from tensorboardX import tensorboard_smoothing
fig, ax1 = plt.subplots(1, 1)    # a figure with a 2x1 grid of Axes
len_mean = pd.read_csv("./csvfile/run-acc_acc-tag-acc2.csv")
len_mean2=pd.read_csv("./csvfile/run-acc_val_acc-tag-acc2.csv")
# ax1.plot(len_mean['Step'], tensorboard_smoothing(len_mean['Value'], smooth=0.6), color="#3399FF")
plt.xlim(0,100)
plt.ylim(20,100)
ax1.plot(len_mean['Step'], 100*len_mean['Value'],  color="#0000FF")
ax1.plot(len_mean2['Step'], 100*len_mean2['Value'],  color="#FF0000")
#ax1.set_xticks(np.arange(0, 24, step=2))
ax1.legend(["train acc","val acc"],loc='lower right')
ax1.set_xlabel("epoch")
ax1.set_ylabel("accuracy(%)")
ax1.set_title("Resnet18")
fig.savefig(fname='./figure/resnet18'+'.png', format='png')
# plt.savefig('./figure/ep_len_mean'+'.png')
plt.show()
