from labml import monit
from numpy.random import random, randint

from labml_app.db import analyses


def update_equal_gap_equal_sizes(prev: int = 0, size=10, gap: int = 5, max_step=100):
    data: analyses.SeriesModel = analyses.series.Series().to_data()
    updates = 0
    while prev <= max_step:
        last = prev + size

        step = [*range(prev, last, gap)]
        value = random(len(step)).tolist()

        prev = step[-1] + gap

        s = analyses.series.Series().load(data)
        s.update(step, value)

        data = s.to_data()

        updates += 1

    print(data['step'].tolist())
    print(updates)


def update_zero_gap(prev: int = 0, size=10, gap: int = 5, max_step=100):
    data: analyses.SeriesModel = analyses.series.Series().to_data()
    while prev <= max_step:
        step = []
        value = []
        for i in range(0, size // gap):
            prev += gap
            step.append(prev)
            value.append(1)
            step.append(prev)
            value.append(10)

        s = analyses.series.Series().load(data)
        s.update(step, value)

        data = s.to_data()

    print(data['step'].tolist())


def update_equal_gap_equal_sizes_diff_gap_between(prev: int = 0, size=5, gap: int = 1, max_step=100):
    data: analyses.SeriesModel = analyses.series.Series().to_data()
    while prev <= max_step:
        last = prev + size

        step = [*range(prev, last, gap)]
        value = random(len(step)).tolist()

        prev = last

        s = analyses.series.Series().load(data)
        s.update(step, value)

        data = s.to_data()

    print(data['step'].tolist())


def update_equal_gap_diff_sizes(prev: int = 0, gap: int = 1, max_step=100):
    data: analyses.SeriesModel = analyses.series.Series().to_data()
    while prev <= max_step:
        size = randint(1, 9)
        last = prev + size

        step = [*range(prev, last, gap)]
        value = random(len(step)).tolist()

        prev = step[-1] + gap

        s = analyses.series.Series().load(data)
        s.update(step, value)

        data = s.to_data()

    print(data['step'].tolist())


def update_equal_gap_diff_sizes_diff_gap_between(prev: int = 0, gap: int = 1, max_step=100):
    data: analyses.SeriesModel = analyses.series.Series().to_data()
    while prev <= max_step:
        size = randint(1, 9)
        last = prev + size

        step = [*range(prev, last, gap)]
        value = random(len(step)).tolist()

        prev = last

        s = analyses.series.Series().load(data)
        s.update(step, value)

        data = s.to_data()

    print(data['step'].tolist())


def update_diff_gap_diff_sizes(prev: int = 0, max_step=100):
    data: analyses.SeriesModel = analyses.series.Series().to_data()
    while prev <= max_step:
        size = randint(1, 6)
        last = prev + size

        step = []
        for s in range(size):
            last = last + randint(1, 10)
            step.append(last)
        value = random(len(step)).tolist()

        prev = last

        s = analyses.series.Series().load(data)
        s.update(step, value)

        data = s.to_data()

    print(data['step'].tolist())


def update_equal_and_diff_gaps_diff_sizes(prev: int = 0, size=5, gap: int = 1, max_step=100):
    data: analyses.SeriesModel = analyses.series.Series().to_data()
    while prev <= max_step / 2:
        last = prev + size

        step = [*range(prev, last, gap)]
        value = random(len(step)).tolist()

        prev = step[-1] + gap

        s = analyses.series.Series().load(data)
        s.update(step, value)

        data = s.to_data()

    while prev <= max_step:
        size = randint(1, 6)
        last = prev + size

        step = []
        for s in range(size):
            last = last + randint(1, 10)
            step.append(last)
        value = random(len(step)).tolist()

        prev = last

        s = analyses.series.Series().load(data)
        s.update(step, value)

        data = s.to_data()

    print(data['step'].tolist())


if __name__ == "__main__":
    with monit.section("Equal gap"):
        update_equal_gap_equal_sizes(size=10000, max_step=1_000_000, gap=1)

    # update_zero_gap()
    # update_equal_gap_equal_sizes(gap=2, max_step=int(1e+4))
    # update_equal_gap_equal_sizes_diff_gap_between(gap=2, max_step=int(1e+4))
    #
    # update_equal_gap_diff_sizes(max_step=int(1e+4))
    # update_equal_gap_diff_sizes(gap=2, max_step=int(1e+4))
    # update_equal_gap_diff_sizes_diff_gap_between(gap=2, max_step=int(1e+4))
    #
    # update_diff_gap_diff_sizes(max_step=int(1e+4))
    # #
    # update_equal_and_diff_gaps_diff_sizes(max_step=int(1e+4))
