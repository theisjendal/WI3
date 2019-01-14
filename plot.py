import pickle
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = pickle.load(open('results.pkl', 'rb'))

    x = [int(v) for v in data]
    test = [float(v['test']) for k, v in data.items()]
    train = [float(v['train']) for k, v in data.items()]

    plt.plot(x, test)
    plt.plot(x, train)

    plt.legend(['test', 'train'], loc='upper left')

    plt.show()
