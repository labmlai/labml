class Schedule:
    def __call__(self, x):
        raise NotImplementedError()


class Flat(Schedule):
    def __init__(self, value):
        self.__value = value

    def __call__(self, x):
        return self.__value


class Dynamic(Schedule):
    def __init__(self, value):
        self.__value = value

    def __call__(self, x):
        return self.__value

    def update(self, value):
        self.__value = value


class Piecewise(Schedule):
    """
    ## Piecewise schedule
    """

    def __init__(self, endpoints, outside_value=None):
        """
        ### Initialize

        `endpoints` is list of pairs `(x, y)`.
         The values between endpoints are linearly interpolated.
        `y` values outside the range covered by `x` are
        `outside_value`.
        """

        # `(x, y)` pairs should be sorted
        indexes = [e[0] for e in endpoints]
        assert indexes == sorted(indexes)

        self._outside_value = outside_value
        self._endpoints = endpoints

    def __call__(self, x):
        """
        ### Find `y` for given `x`
        """

        # iterate through each segment
        for (x1, y1), (x2, y2) in zip(self._endpoints[:-1], self._endpoints[1:]):
            # interpolate if `x` is within the segment
            if x1 <= x < x2:
                dx = float(x - x1) / (x2 - x1)
                return y1 + dx * (y2 - y1)

        # return outside value otherwise
        return self._outside_value
