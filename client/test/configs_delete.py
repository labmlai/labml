from labml import experiment
from labml.configs import BaseConfigs, option


class Configs(BaseConfigs):
    discard: str
    final: str
    other: int


@option(Configs.discard)
def _discard(c: Configs):
    return 'Varune '


@option(Configs.final)
def _final(c: Configs):
    res = c.discard + 'Final'
    # Cannot be updated after accessing
    c.other = 2
    res +=  str(c.other)
    c.other = 3
    c.discard = None


def main():
    experiment.evaluate()
    conf = Configs()
    experiment.configs(conf)

    with experiment.start():
        print(conf.final)
        print(conf.discard)


if __name__ == '__main__':
    main()
