from k_means import KMeans


def main():
    k = 3
    data = [1.5, 9.5, 5.4, 1.6, 5.5, 9.3, 1.7, 9.1, 1.3, 2.0, 5.0, 7.0, 7.7, 8.0]
    data = zip(data, data)
    KMeans(k, data).cluster()


if __name__ == '__main__':
    main()
