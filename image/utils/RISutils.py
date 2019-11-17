import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import tensorflow as tf

def dataset_to_array(dataset, steps):
    xs = []
    ys = []
    for x_batch, y_batch in dataset.take(steps):
        xs.append(x_batch.numpy())
        ys.append(y_batch.numpy())

    return np.asarray(xs).reshape((-1, *(299,299,3))), np.asarray(ys).flatten()

def show_distances(
    xs, ys, target_index, similar_indices, distances, cols: int = 4, debug: bool = False
):
    if cols >= len(distances) + 1:
        cols = len(distances) + 1
        rows = 1
    else:
        rows = math.ceil((len(distances) + 1) / cols)

    figsize = (3 * cols, 4 * rows) if debug else (3 * cols, 3 * rows)
    _, ax = plt.subplots(rows, cols, figsize=figsize)

    i = 0
    for x, y, distance in zip(
        xs[[target_index] + similar_indices],
        ys[[target_index] + similar_indices],
        [0] + distances,
    ):
        idx = (i // cols, i % cols) if rows > 1 else i % cols
        ax[idx].axis("off")
        ax[idx].imshow(x)
        title = f"Label: {y}\nShape: {x.shape}\n" if debug else f"{y}\n{distance:.2f}"
        ax[idx].set_title(title)
        i += 1

def show_minmax_emb_images(df, embedding_column, xs):
    # Get min/max image indices
    min_emb_indices = df.sort_values(embedding_column, ascending=True)[:5].index
    max_emb_indices = df.sort_values(embedding_column, ascending=False)[:5].index
    # Plot
    _, ax = plt.subplots(1, 5, figsize=(15,10))
    _, ax2 = plt.subplots(1, 5, figsize=(15,10))
    for i in range(5):
        ax[i].imshow(xs[min_emb_indices[i]])
        ax[i].set_title(f"{embedding_column} value: {round(df.loc[min_emb_indices[i], embedding_column], 0)}")
        ax2[4-i].imshow(xs[max_emb_indices[i]])
        ax2[4-i].set_title(f"{embedding_column} value: {round(df.loc[max_emb_indices[i], embedding_column], 0)}")
