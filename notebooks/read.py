import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt


def merge(l):
	l = np.array(l)
	return np.mean(l, axis=0)


training_loss, testing_loss, accuracy = [], [], []

for n in ['org','5','10','20']:
	training, testing, acc = [], [], []
	for i in range(5):
		f_name = n+'-alexnet-cifar-'+str(i)+'.pkl'
		train_losses, test_losses, accuracys = pkl.load(open(f_name, 'rb'))
		training.append(train_losses)
		testing.append(test_losses)
		acc.append(accuracys)
	training_loss.append(merge(training))
	testing_loss.append(merge(testing))
	accuracy.append(merge(acc))

for i in range(4):
	print(testing_loss[i][-1])

print()
for i in range(4):
	print(1-accuracy[i][-1])

# plt.title('Testing loss for diffrent sparsity')
# plt.plot(np.arange(len(testing_loss[0])), testing_loss[0], label="original")
# plt.plot(np.arange(len(testing_loss[1])), testing_loss[1], label="80%")
# plt.plot(np.arange(len(testing_loss[2])), testing_loss[2], label="90%")
# plt.plot(np.arange(len(testing_loss[3])), testing_loss[3], label="95%")
# plt.ylabel("Loss")
# plt.xlabel('Epochs')
# plt.legend()

# plt.figure()
# plt.title('AlexNet-cifar-10 Accuracies for diffrent sparsity')
# plt.plot(np.arange(len(accuracy[0])), accuracy[0], label="original")
# plt.plot(np.arange(len(accuracy[1])), accuracy[1], label="80%")
# plt.plot(np.arange(len(accuracy[2])), accuracy[2], label="90%")
# plt.plot(np.arange(len(accuracy[3])), accuracy[3], label="95%")
# plt.ylabel("accuracy")
# plt.xlabel('Epochs')
# plt.legend()
# plt.show()
# train_mean = merge(training)
# test_mean = merge(testing)
# acc_mean = merge(acc)

# def print_losses_and_acc(training_losses_, test_losses_, acc_):
#     plt.title("L'erreur moyenne d'un batch")
#     plt.plot(np.arange(len(training_losses_)), training_losses_, label="Train")
#     plt.plot(np.arange(len(test_losses_)), test_losses_, label="Test")
#     plt.ylabel("L'erreur")
#     plt.xlabel('Epochs')
#     plt.legend()
#     plt.figure()
#     plt.plot(np.arange(len(acc_)), acc_)
#     plt.show()

# print_losses_and_acc(train_mean, test_mean, acc_mean)