import pylab as pl

x1 = [1, 2, 3, 4, 5, 6]# Make x, y arrays for each graph
x = [0.001, 0.01, 0.1, 1, 10, 100]# Make x, y arrays for each graph
y1 = [0.8487, 0.8562, 0.8578, 0.8581, 0.8255, 0.7031]
x2 = x1
y2 = [0.9116, 0.9148, 0.9101, 0.9081, 0.8992, 0.7140]

import matplotlib.pyplot as plt

ax = plt.subplot(111, xlabel='lambda', ylabel='F1 score')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
 item.set_fontsize(15)

plot1, = ax.plot(x1, y1, 'rs-', label='Disease')# use pylab to plot x and y : Give your plots names
plot2, = ax.plot(x2, y2, 'b^-', label='Procedure')

# pl.title('MTL performance with different lambda of constraints')# give plot a title
pl.xticks(x1, x, rotation=0)

pl.legend(handles=[plot1, plot2],  numpoints=1, fontsize=15)# make legend
pl.show()# show the plot on the screen
