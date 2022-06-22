
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot(timestamps):
    fig, ax = plt.subplots()
    lCollections = list() 
    colors = ['silver', 'orange', 'cyan', 'gray', 'red', 'magenta', 'brown', 'pink', 'peru', 'purple']
    hatches = ['/', '\\', '+', 'x', '.', '*', 'o', 'O',]
    for index,line in enumerate(timestamps):
        time_list = [matplotlib.patches.Rectangle(xy=(x[0],index),width=x[1]-x[0],height=0.5) for x in line]
        line_coll=matplotlib.collections.PatchCollection(time_list, color=colors[index], edgecolor='black', hatch=hatches[0])
        lCollections.append(line_coll)
        ax.add_collection(line_coll)

    annot = ax.annotate("",xy=(0,0),xytext=(1,8),xycoords="data",textcoords="offset points",
            bbox=dict(boxstyle="round",fc="w"),
            arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    # hover function for annotations in figure
    def hover(event):
        if event.inaxes == ax:
            bSome = False
            for index in range(len(timestamps)):
                cont,ind = lCollections[index].contains(event)
                if cont:
                    i = ind["ind"][0]
                    annot.set_text("%f %f"%(timestamps[index][i][0],timestamps[index][i][1]))
                    annot.xy = (np.mean(timestamps[index][i]),index+0.5)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    bSome = True
            if not bSome:
                annot.set_visible(False)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event",hover)
    ax.autoscale()

    plt.show()
    plt.savefig('timeline.jpg')

if __name__ == '__main__':
    # l=[np.reshape(np.sort(np.random.randn(20)),[10,2]) for x in range(4)]
    # Plot(l)
    data = [[[1, 9]] for _ in range(7)]
    plot(data)