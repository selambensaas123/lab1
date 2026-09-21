"""Заготовки задач на NumPy."""

import numpy as np

from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices = np.array(data.matrices)
    vectors = np.array(data.vectors)

    result = np.zeros_like(vectors[0])

    for i in range(len(matrices)):
        product = np.matmul(matrices[i], vectors[i])
        result = result + product

    return result


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix = np.array(data.matrix)
    threshold = data.threshold

    result = np.zeros(matrix.shape, dtype=int)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] > threshold:
                result[i][j] = 1

    return result


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.array(data.matrix)

    result = []

    for row in matrix:
        unique_values = np.unique(row)
        result.append(unique_values.tolist())

    return result


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.array(data.matrix)

    result = []

    for column_number in range(matrix.shape[1]):
        column = matrix[:, column_number]
        unique_values = np.unique(column)
        result.append(unique_values.tolist())

    return result


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows = data.rows
    columns = data.columns
    mean = data.mean
    std = data.std
    seed = data.seed

    rng = np.random.default_rng(seed)

    matrix = rng.normal(
        loc=mean,
        scale=std,
        size=(rows, columns)
    )

    row_means = np.mean(matrix, axis=1)
    column_means = np.mean(matrix, axis=0)

    row_variances = np.var(matrix, axis=1)
    column_variances = np.var(matrix, axis=0)

    return MatrixStatistics(
        matrix=matrix,
        row_means=row_means,
        column_means=column_means,
        row_variances=row_variances,
        column_variances=column_variances
    )


def chess(data: ChessInput) -> np.ndarray:
    rows = data.rows
    columns = data.columns
    first = data.first
    second = data.second

    matrix = np.zeros((rows, columns))

    for i in range(rows):
        for j in range(columns):
            if (i + j) % 2 == 0:
                matrix[i][j] = first
            else:
                matrix[i][j] = second

    return matrix


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width = data.width
    height = data.height

    image_height = data.image_height
    image_width = data.image_width

    shape_color = data.shape_color
    background_color = data.background_color

    image = np.zeros(
        (image_height, image_width, 3),
        dtype=int
    )

    image[:] = background_color

    start_x = (image_width - width) // 2
    start_y = (image_height - height) // 2

    end_x = start_x + width
    end_y = start_y + height

    image[start_y:end_y, start_x:end_x] = shape_color

    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x = data.semi_axis_x
    semi_axis_y = data.semi_axis_y

    image_height = data.image_height
    image_width = data.image_width

    shape_color = data.shape_color
    background_color = data.background_color

    image = np.zeros(
        (image_height, image_width, 3),
        dtype=int
    )

    image[:] = background_color

    center_x = (image_width - 1) / 2
    center_y = (image_height - 1) / 2

    for y in range(image_height):
        for x in range(image_width):

            x_part = ((x - center_x) ** 2) / (semi_axis_x ** 2)
            y_part = ((y - center_y) ** 2) / (semi_axis_y ** 2)

            if x_part + y_part <= 1:
                image[y][x] = shape_color

    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values = np.array(data.values)
    window = data.window

    mean = np.mean(values)
    variance = np.var(values)
    std = np.std(values)

    local_maxima = []
    local_minima = []

    for i in range(1, len(values) - 1):

        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            local_maxima.append(i)

        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            local_minima.append(i)

    moving_average = []

    for i in range(len(values) - window + 1):
        current_window = values[i:i + window]
        average = np.mean(current_window)
        moving_average.append(average)

    return TimeSeriesStatistics(
        mean=float(mean),
        variance=float(variance),
        std=float(std),
        local_maxima_indices=np.array(local_maxima),
        local_minima_indices=np.array(local_minima),
        moving_average=np.array(moving_average)
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels = np.array(data.labels)
    class_count = data.class_count

    if class_count is None:
        class_count = int(np.max(labels)) + 1

    result = np.zeros(
        (len(labels), class_count),
        dtype=int
    )

    for i in range(len(labels)):
        class_number = labels[i]
        result[i][class_number] = 1

    return result