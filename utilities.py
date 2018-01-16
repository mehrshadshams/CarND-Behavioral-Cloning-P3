import matplotlib.pyplot as plt


def plot_history(history, show=True):
    epochs = len(history)
    plt.plot([history[i][0] for i in range(epochs)], label='train')
    plt.plot([history[i][1] for i in range(epochs)], label='valid')
    plt.title('Accuracy')
    plt.legend(['train', 'test'])
    plt.savefig('history.png')

    if show:
        plt.show()
