"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value

    return sum(1 for c in text.lower() if c in "aeiou")


def has_unique_characters(data: TextInput) -> bool:
    text = data.value

    return len(set(text)) == len(text)


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value

    return bin(number).count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value

    steps = 0

    while number >= 10:
        product = 1

        for digit in str(number):
            product = product * int(digit)

        number = product
        steps = steps + 1

    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected

    total = 0

    for i in range(len(predicted)):
        difference = predicted[i] - expected[i]
        square = difference * difference
        total = total + square

    result = total / len(predicted)

    return result


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value

    result = ""
    divisor = 2

    while divisor * divisor <= number:
        power = 0

        while number % divisor == 0:
            number = number // divisor
            power = power + 1

        if power == 1:
            result = result + "(" + str(divisor) + ")"

        if power > 1:
            result = result + "(" + str(divisor) + "**" + str(power) + ")"

        divisor = divisor + 1

    if number > 1:
        result = result + "(" + str(number) + ")"

    return result


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value

    level = 0
    total = 0

    while total < cube_count:
        level = level + 1
        total = total + level * level

    if total == cube_count:
        return level
    else:
        return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value

    digits = str(number)
    length = len(digits)

    left_sum = 0
    right_sum = 0

    if length % 2 == 1:
        middle = length // 2

        for i in range(middle):
            left_sum = left_sum + int(digits[i])

        for i in range(middle + 1, length):
            right_sum = right_sum + int(digits[i])

    else:
        middle = length // 2

        for i in range(middle - 1):
            left_sum = left_sum + int(digits[i])

        for i in range(middle + 1, length):
            right_sum = right_sum + int(digits[i])

    if left_sum == right_sum:
        return True
    else:
        return False