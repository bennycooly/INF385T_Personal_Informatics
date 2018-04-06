
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans


def modulo(index, window_size):
    return index % window_size


def load_data():
    # Read in the steps dataset
    df = pd.read_csv("steps.csv")

    window_size = 10

    # remove timestamps from values
    df_values = df[["accelerometerAccelerationX(G)", "accelerometerAccelerationY(G)", "accelerometerAccelerationZ(G)",
        "gyroRotationX(rad/s)", "gyroRotationY(rad/s)", "gyroRotationZ(rad/s)"]]
    
    # now scale the data
    scaler = MinMaxScaler()
    # df_values = pd.DataFrame(scaler.fit_transform(df_values.values), columns=df_values.columns, index=df_values.index)

    # Now calculate the max, mean, and standard deviation for each window of time    
    df_features = None

    df_max = df_values.rolling(window_size).max()
    df_max = df_max.add_suffix("_max")
    df_features = df_max
    # print(df_features)

    df_mean = df_values.rolling(window_size).mean()
    df_mean = df_mean.add_suffix("_mean")
    # print(df_mean)
    df_features = df_features.join(df_mean)

    df_std = df_values.rolling(window_size).std()
    df_std = df_std.add_suffix("_std")
    df_features = df_features.join(df_std)
    # print(df_std)

    # Add a new column that represents the time series index of each entry in the window
    df_features["time_index"] = df_features.index
    df_features["time_index"] = df_features["time_index"].apply(modulo, window_size=window_size)

    df_features = df_features.dropna()
    print(df_features)

    plt.figure()
    plt.subplot(121)
    plt.scatter(df_features["time_index"], df_features["accelerometerAccelerationX(G)_max"], label="X", marker=".", alpha=0.5)
    plt.scatter(df_features["time_index"], df_features["accelerometerAccelerationY(G)_max"], label="Y", marker=".", alpha=0.5)
    plt.scatter(df_features["time_index"], df_features["accelerometerAccelerationZ(G)_max"], label="Z", marker=".", alpha=0.5)
    plt.legend()

    plt.subplot(122)
    plt.scatter(df_features["time_index"], df_features["gyroRotationX(rad/s)_max"], label="X", marker=".", alpha=0.5)
    plt.scatter(df_features["time_index"], df_features["gyroRotationY(rad/s)_max"], label="Y", marker=".", alpha=0.5)
    plt.scatter(df_features["time_index"], df_features["gyroRotationZ(rad/s)_max"], label="Z", marker=".", alpha=0.5)
    plt.legend()
    plt.show()

    X = df_features

    km = KMeans(n_clusters=4)
    km.fit(X)

    clusters = pd.DataFrame(km.cluster_centers_, columns=df_features.columns)
    print(clusters)
    print(km.labels_)

    unique, counts = np.unique(km.labels_, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    print(cluster_counts)

    # Calculate number of steps based on the visualizations of the training data
    print("Number of steps: " + str(get_num_steps(clusters, cluster_counts)))

def get_num_steps(clusters, counts):
    clusters["sort_index"] = clusters.index
    # Choose to sort by most disparate (gyroZ)
    clusters = clusters.sort_values(by=["gyroRotationZ(rad/s)_max"])

    # Take the local maximums of each cluster and return the counts
    return counts[clusters.iloc[1]["sort_index"]] + counts[clusters.iloc[3]["sort_index"]]


def main():
    load_data()


if __name__ == "__main__":
    main()
