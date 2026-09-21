import numpy as np

from python_basics import (
    count_vowels,
    has_unique_characters,
    count_one_bits,
    multiplicative_persistence,
    mse,
    prime_factorization,
    pyramid,
    is_balanced_number,
)

from numpy_tasks import (
    sum_prod,
    binarize,
    unique_rows,
    unique_columns,
    matrix_statistics,
    chess,
    draw_rectangle,
    draw_ellipse,
    analyze_time_series,
    one_hot,
)

from grader_contracts.python_basics import (
    TextInput,
    PositiveIntegerInput,
    VectorPairInput,
)

from grader_contracts.numpy_tasks import (
    MatrixVectorBatchInput,
    BinarizeInput,
    MatrixInput,
    RandomMatrixInput,
    ChessInput,
    RectangleInput,
    EllipseInput,
    TimeSeriesInput,
    OneHotInput,
)


# -------------------------
# Python testleri
# -------------------------

def test_count_vowels():
    assert count_vowels(TextInput("Hello")) == 2
    assert count_vowels(TextInput("AEIOU")) == 5
    assert count_vowels(TextInput("")) == 0


def test_unique_characters():
    assert has_unique_characters(TextInput("abc")) == True
    assert has_unique_characters(TextInput("hello")) == False
    assert has_unique_characters(TextInput("")) == True


def test_count_one_bits():
    assert count_one_bits(PositiveIntegerInput(1)) == 1
    assert count_one_bits(PositiveIntegerInput(7)) == 3
    assert count_one_bits(PositiveIntegerInput(8)) == 1


def test_multiplicative_persistence():
    assert multiplicative_persistence(PositiveIntegerInput(39)) == 3
    assert multiplicative_persistence(PositiveIntegerInput(4)) == 0
    assert multiplicative_persistence(PositiveIntegerInput(999)) == 4


def test_mse():
    data = VectorPairInput(
        predicted=[1, 2, 3],
        expected=[1, 2, 3]
    )

    assert mse(data) == 0

    data = VectorPairInput(
        predicted=[0, 0],
        expected=[1, 3]
    )

    assert mse(data) == 5


def test_prime_factorization():
    assert prime_factorization(
        PositiveIntegerInput(86240)
    ) == "(2**5)(5)(7**2)(11)"

    assert prime_factorization(
        PositiveIntegerInput(2)
    ) == "(2)"


def test_pyramid():
    assert pyramid(PositiveIntegerInput(1)) == 1
    assert pyramid(PositiveIntegerInput(5)) == 2
    assert pyramid(PositiveIntegerInput(14)) == 3
    assert pyramid(PositiveIntegerInput(4)) == "It is impossible"


def test_balanced_number():
    assert is_balanced_number(
        PositiveIntegerInput(1234006)
    ) == True

    assert is_balanced_number(
        PositiveIntegerInput(123456)
    ) == False


# -------------------------
# NumPy testleri
# -------------------------

def test_sum_prod():
    matrices = np.array([
        [[1, 0], [0, 1]],
        [[2, 0], [0, 2]]
    ])

    vectors = np.array([
        [[1], [2]],
        [[3], [4]]
    ])

    data = MatrixVectorBatchInput(
        matrices=matrices,
        vectors=vectors
    )

    result = sum_prod(data)

    expected = np.array([
        [7],
        [10]
    ])

    assert np.array_equal(result, expected)


def test_binarize():
    matrix = np.array([
        [0.2, 0.7],
        [0.5, 0.9]
    ])

    data = BinarizeInput(
        matrix=matrix,
        threshold=0.5
    )

    expected = np.array([
        [0, 1],
        [0, 1]
    ])

    assert np.array_equal(
        binarize(data),
        expected
    )


def test_unique_rows():
    matrix = np.array([
        [1, 1, 2],
        [3, 3, 4]
    ])

    data = MatrixInput(matrix)

    assert unique_rows(data) == [
        [1, 2],
        [3, 4]
    ]


def test_unique_columns():
    matrix = np.array([
        [1, 2],
        [1, 3],
        [4, 3]
    ])

    data = MatrixInput(matrix)

    assert unique_columns(data) == [
        [1, 4],
        [2, 3]
    ]


def test_matrix_statistics():
    data = RandomMatrixInput(
        rows=3,
        columns=4,
        mean=0,
        std=1,
        seed=42
    )

    result = matrix_statistics(data)

    assert result.matrix.shape == (3, 4)
    assert len(result.row_means) == 3
    assert len(result.column_means) == 4
    assert len(result.row_variances) == 3
    assert len(result.column_variances) == 4


def test_chess():
    data = ChessInput(
        rows=3,
        columns=4,
        first=0,
        second=1
    )

    expected = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])

    assert np.array_equal(
        chess(data),
        expected
    )


def test_draw_rectangle():
    data = RectangleInput(
        width=4,
        height=2,
        image_height=6,
        image_width=8,
        shape_color=(255, 0, 0),
        background_color=(0, 0, 0)
    )

    result = draw_rectangle(data)

    assert result.shape == (6, 8, 3)

    assert np.array_equal(
        result[3][4],
        np.array([255, 0, 0])
    )

    assert np.array_equal(
        result[0][0],
        np.array([0, 0, 0])
    )


def test_draw_ellipse():
    data = EllipseInput(
        semi_axis_x=3,
        semi_axis_y=2,
        image_height=7,
        image_width=9,
        shape_color=(255, 255, 255),
        background_color=(0, 0, 0)
    )

    result = draw_ellipse(data)

    assert result.shape == (7, 9, 3)

    assert np.array_equal(
        result[3][4],
        np.array([255, 255, 255])
    )

    assert np.array_equal(
        result[0][0],
        np.array([0, 0, 0])
    )


def test_analyze_time_series():
    data = TimeSeriesInput(
        values=[1, 3, 1, 4, 2],
        window=3
    )

    result = analyze_time_series(data)

    assert result.mean == np.mean(
        [1, 3, 1, 4, 2]
    )

    assert result.variance == np.var(
        [1, 3, 1, 4, 2]
    )

    assert result.std == np.std(
        [1, 3, 1, 4, 2]
    )

    assert np.array_equal(
        result.local_maxima_indices,
        np.array([1, 3])
    )

    assert np.array_equal(
        result.local_minima_indices,
        np.array([2])
    )

    expected_average = np.array([
        (1 + 3 + 1) / 3,
        (3 + 1 + 4) / 3,
        (1 + 4 + 2) / 3
    ])

    assert np.allclose(
        result.moving_average,
        expected_average
    )


def test_one_hot():
    data = OneHotInput(
        labels=[0, 2, 3, 0],
        class_count=4
    )

    expected = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])

    assert np.array_equal(
        one_hot(data),
        expected
    )