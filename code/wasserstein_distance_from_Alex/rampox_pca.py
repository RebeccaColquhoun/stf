import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ot as ot
import time

def w2(distribution_1: np.ndarray, distribution_2: np.ndarray) -> float:
    """
    Compute the 2nd Wasserstein distance between two energy distributions.
    """
    # Assert that the sum of each distribution is 1
    assert np.allclose(distribution_1.sum(), 1)
    assert np.allclose(distribution_2.sum(), 1)
    # Assert length of the two distributions is the same and equal to the length of the energies
    assert len(distribution_1) == len(ENERGIES)
    assert len(distribution_2) == len(ENERGIES)
    return np.sqrt(
        ot.wasserstein_1d(
            u_values=ENERGIES,
            v_values=ENERGIES,
            u_weights=distribution_1,
            v_weights=distribution_2,
            p=2,
        )
    )


def w2_distance_matrix(data: np.ndarray) -> np.ndarray:
    """
    Compute the 2nd Wasserstein distance matrix between all pairs of energy distributions.
    """
    n = data.shape[0]
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = w2(data[i, :], data[j, :])
    # Assert the distance matrix is symmetric
    assert np.allclose(distance_matrix, distance_matrix.T)
    return distance_matrix


def centering_matrix(n: int) -> np.ndarray:
    """
    Return the centering matrix H of size n x n.
    """
    return np.eye(n) - np.ones((n, n)) / n


def gram_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the Gram matrix G from the distance matrix.
    """
    n = distance_matrix.shape[0]
    H = centering_matrix(n)
    # Define A the matrix of negative squared distances (i.e., a_ij = -0.5 * d_ij^2)
    A = -0.5 * distance_matrix**2
    G = H @ A @ H
    return G

for n in [12]:
    FILENAME = "/home/earthquakes1/homes/Rebecca/phd/stf/code/wasserstein_distance_from_Alex/norm_processed_data.csv"

    # Record the start time
    start_time = time.time()

    # Load in file
    df = pd.read_csv(FILENAME)
    # Extract the values of the energies
    ENERGIES = np.array(df.columns[2:].astype(float))
    # Extract the values which are from the 3rd column to the last column
    data = df.iloc[0:n, 2:]
    print(data)
    data = data.div(data.sum(axis=1), axis=0)
    print(data)
    print(data.sum(axis=1))
    # Assert that the sum of each row is 1
    assert np.allclose(data.sum(axis=1), 1)
    data = data.values

    distance_matrix = w2_distance_matrix(data)

    gram_matrix = gram_matrix(distance_matrix)
    # Perform eigendecomposition of the Gram matrix
    eigenvalues, eigenvectors = np.linalg.eigh(gram_matrix)

    # Sort the eigenvectors by decreasing eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # save the eigenvectors
    np.savetxt("eigenvectors.csv", eigenvectors, delimiter=",")

    # check if any eigenvalues are negative. The last few eigenvalues may be negative due to numerical errors, and the final
    # eigenvalue in this calculation is always 0 (which can round to a small negative number)
    if np.any(eigenvalues < 0):
        # Print a warning in red
        print("\033[91m" + "Some eigenvalues are negative! Setting them to 0...")
        # Print the eigenvalues that are negative and set them to 0
        print("-ve eigenvalues:", eigenvalues[eigenvalues < 0],"\033[0m")
        eigenvalues[eigenvalues < 0] = 0

    # Calculate coordinates of points by multiplying eigenvectors by the square root of the eigenvalues
    coordinates = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    # Calculate the cumulative variance explained by the eigenvectors
    cumulative_variance_explained = np.cumsum(eigenvalues) / eigenvalues.sum()

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Save the elapsed time and the value of n to a file
    with open("/home/earthquakes1/homes/Rebecca/phd/stf/code/wasserstein_distance_from_Alex/execution_time.txt", "a") as f:
        f.write(f"n: {n}, Execution time: {elapsed_time} seconds\n")


    ### Plotting ###

    # Load in which coordinate to plot on x axis from user input
    x_axis_coord = 1 - 1 #int(input("Enter the coordinate to plot on the x axis: ")) - 1
    # Load in which coordinate to plot on y axis from user input
    y_axis_coord = 2 - 1# int(input("Enter the coordinate to plot on the y axis: ")) - 1

    # Extract the proportion of variance explained by each axis
    x_axis_variance_explained = eigenvalues[x_axis_coord] / eigenvalues.sum()
    y_axis_variance_explained = eigenvalues[y_axis_coord] / eigenvalues.sum()

    plt.figure(figsize=(8, 10))
    plt.subplot(1, 1, 1)
    x = coordinates[:, x_axis_coord]  # First column of the coordinates
    y = coordinates[:, y_axis_coord]  #1 Second column of the coordinates
    categories = df["magnitude"].values
    labels = df["scardec_name"].values

    # Extract the sample IDs of the two samples with the largest and smallest x coordinates and their distribution
    largest_x_coordinate = labels[np.argmax(x)]
    largest_x_coordinate_distribution = data[np.argmax(x), :]
    smallest_x_coordinate = labels[np.argmin(x)]
    smallest_x_coordinate_distribution = data[np.argmin(x), :]

    # Extract the sample IDs of the two samples with the largest and smallest y coordinates and their distribution
    largest_y_coordinate = labels[np.argmax(y)]
    largest_y_coordinate_distribution = data[np.argmax(y), :]
    smallest_y_coordinate = labels[np.argmin(y)]
    smallest_y_coordinate_distribution = data[np.argmin(y), :]

    unique_categories = np.unique(categories)
    # Create a dictionary that maps each category to a color
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_categories)))
    color_dict = dict(zip(unique_categories, colors))

    for i in range(len(x)):
        plt.scatter(x[i], y[i], color=color_dict[categories[i]])
    # Create a custom legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_dict[cat],
            markersize=10,
        )
        for cat in unique_categories
    ]
    plt.legend(handles, unique_categories)
    plt.xlabel(
        "Coordinate "
        + str(x_axis_coord + 1)
        + " (prop. var.: "
        + str(round(x_axis_variance_explained, 3))
        + ")"
    )
    plt.ylabel(
        "Coordinate "
        + str(y_axis_coord + 1)
        + " (prop. var.: "
        + str(round(y_axis_variance_explained, 3))
        + ")"
    )
    # Make the axes equal
    plt.title("Categories")
    plt.axis("equal")

    # Plot the same data but with labels
    # plt.subplot(2, 1, 2)
    # for i in range(len(x)):
    #     plt.scatter(x[i], y[i], c="grey")
    #     plt.text(x[i], y[i], labels[i], fontsize=8)
    # plt.axis("equal")
    # plt.xlabel(
    #     "Coordinate "
    #     + str(x_axis_coord + 1)
    #     + " (prop. var.: "
    #     + str(round(x_axis_variance_explained, 2))
    #     + ")"
    # )
    # plt.ylabel(
    #     "Coordinate "
    #     + str(y_axis_coord + 1)
    #     + " (prop. var.: "
    #     + str(round(y_axis_variance_explained, 2))
    #     + ")"
    # )
    # plt.title("Sample IDs")
    # plt.tight_layout()
    # save figure at a sensible size
    plt.savefig(
        f"/home/earthquakes1/homes/Rebecca/phd/stf/figures/wasserstein/{n}_coord_plot_" + str(x_axis_coord + 1) + "_" + str(y_axis_coord + 1) + ".png",
        dpi=300,
    )
    # plt.show()
    plt.close()


    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    # Plot the distributions of the two samples with the largest and smallest first coordinates on the same plot
    plt.plot(
        ENERGIES, largest_x_coordinate_distribution, label=largest_x_coordinate, c="#1f78b4"
    )
    plt.plot(
        ENERGIES,
        smallest_x_coordinate_distribution,
        label=smallest_x_coordinate,
        c="#a6cee3",
    )
    plt.xlabel("Energy")
    plt.ylabel("Distribution")
    # Make y axis start at 0
    plt.ylim(0, None)
    plt.legend()
    plt.title("Largest/smallest on coordinate " + str(x_axis_coord + 1))

    plt.subplot(2, 1, 2)
    # Plot the distributions of the two samples with the largest and smallest second coordinates on the same plot
    plt.plot(
        ENERGIES, largest_y_coordinate_distribution, label=largest_y_coordinate, c="#33a02c"
    )
    plt.plot(
        ENERGIES,
        smallest_y_coordinate_distribution,
        label=smallest_y_coordinate,
        c="#b2df8a",
    )
    plt.xlabel("Energy")
    plt.ylabel("Distribution")
    # Make y axis start at 0
    plt.ylim(0, None)
    plt.legend()
    plt.title("Largest/smallest on coordinate " + str(y_axis_coord + 1))
    plt.tight_layout()
    # Save the figure as a png file including the coordinates used
    plt.savefig(f"/home/earthquakes1/homes/Rebecca/phd/stf/figures/wasserstein/{n}_interpretation_" + str(x_axis_coord + 1) + "_" + str(y_axis_coord + 1) + ".png", dpi=300)
    # plt.show()
    # plt.show()
    plt.close()