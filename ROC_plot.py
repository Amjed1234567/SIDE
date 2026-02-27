import matplotlib.pyplot as plt

dim_1 = [0.8711, 0.8716, 0.8675, 0.8706, 0.8742]
dim_2 = [0.9158, 0.9181, 0.9187, 0.9171, 0.9169]
dim_3 = [0.9308, 0.9302, 0.9326, 0.9332, 0.9319]
dim_4 = [0.9412, 0.9405, 0.9408, 0.9423, 0.9395]
dim_5 = [0.9442, 0.9472, 0.9453, 0.9476, 0.9464]
dim_6 = [0.9506, 0.9508, 0.9508, 0.9500, 0.9501]
dim_7 = [0.9560, 0.9537, 0.9548, 0.9531, 0.9545]
dim_8 = [0.9581, 0.9585, 0.9571, 0.9572, 0.9568]
dim_9 = [0.9603, 0.9617, 0.9594, 0.9608, 0.9618]
dim_10 = [0.9608, 0.9636, 0.9619, 0.9629, 0.9635]

data = [dim_1, dim_2, dim_3, dim_4, dim_5, dim_6, dim_7, dim_8, dim_9, dim_10,]

positions = list(range(1, 11))
labels = [str(i) for i in range(1, 11)]

plt.boxplot(data, positions=positions, labels=labels)

plt.title('ROC-AUC Boxplots')
plt.xlabel('Dimensions')
plt.ylabel('Value')

plt.savefig('boxplot_ROC.png')
plt.close()