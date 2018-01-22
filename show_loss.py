import pickle
import utilities


def main():
    with open('history.pkl', 'rb') as f:
        history = pickle.load(f)

    utilities.plot_history(history)


if __name__ == "__main__":
    main()