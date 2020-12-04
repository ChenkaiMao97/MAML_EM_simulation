from matplotlib import pyplot as plt

f = open("result_logs/Meta_val_200_epochs_1_fre.txt", 'r')
lines = f.readlines()

epochs = []
pre_losses = []
after_losses = []
average_num = 2
average_list_pre = []
average_list_after = []
count = 0
for line in lines:
	count += 1
	loss_strings = line.split(',')
	average_list_pre.append(float(loss_strings[0][-7:]))
	average_list_after.append(float(loss_strings[1][-7:]))
	if(count % average_num ==0):
		pre_losses.append(sum(average_list_pre)/average_num)
		after_losses.append(sum(average_list_after)/average_num)
		epochs.append(25*count/200)
		average_list_pre = []
		average_list_after = []

plt.figure()
plt.plot(epochs, pre_losses)
plt.plot(epochs, after_losses)
plt.xlabel("epochs")
plt.ylabel("losses")
plt.ylim([0,0.8])
plt.xlim([0,160])
plt.legend(["loss_tr","loss_ts"])
plt.grid()
plt.title("Meta Validation loss")
plt.show()
#plt.save("Meta Validation loss")
