import matplotlib.pyplot as plt

dim_1 = [0.0551, 0.0530, 0.0503, 0.0542, 0.0592]
dim_2 = [0.1215, 0.1171, 0.1235, 0.1155, 0.1182]
dim_3 = [0.1382, 0.1390, 0.1400, 0.1473, 0.1373]
dim_4 = [0.1649, 0.1758, 0.1603, 0.1646, 0.1580]
dim_5 = [0.1721, 0.1670, 0.1643, 0.1738, 0.1757]
dim_6 = [0.1826, 0.1897, 0.1868, 0.1899, 0.1844]
dim_7 = [0.2021, 0.1888, 0.1945, 0.1861, 0.1940]
dim_8 = [0.2003, 0.2053, 0.1990, 0.2065, 0.2058]
dim_9 = [0.2224, 0.2121, 0.2021, 0.2139, 0.2228]
dim_10 = [0.2192, 0.2229, 0.2111, 0.2277, 0.2182]

data = [dim_1, dim_2, dim_3, dim_4, dim_5, dim_6, dim_7, dim_8, dim_9, dim_10,]

positions = list(range(1, 11))
labels = [str(i) for i in range(1, 11)]

plt.boxplot(data, positions=positions, labels=labels)

plt.title('PR-AUC Boxplots')
plt.xlabel('Dimensions')
plt.ylabel('Value')

plt.savefig('boxplot_PR.png')
plt.close()