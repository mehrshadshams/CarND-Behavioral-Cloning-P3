import matplotlib.pyplot as plt


def plot_history(history, show=True):
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='valid')
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.savefig('history.png')

    if show:
        plt.show()
