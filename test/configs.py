import json

import yaml

from labml.configs import option
from labml.internal.configs.base import Configs


class ParentConfigs(Configs):
    p1: str = 'p1_default'
    p2: str = 'p2_default'
    calc1: str


@option(ParentConfigs.calc1)
def sample_model(c: ParentConfigs):
    return c.p1 + c.p2 + ' calc'


class ModuleWithPrimary(Configs):
    mp_calc: str
    p1_m1: str

    def __init__(self):
        super().__init__(_primary='mp_calc')


@option(ModuleWithPrimary.mp_calc)
def sample_mp_calc(c: ModuleWithPrimary):
    return c.p1_m1 + ' main'


class Module(Configs):
    m_calc: str
    p1_m1: str


@option(Module.m_calc)
def sample_m_calc(c: Module):
    return c.p1_m1 + ' main no primary'


class MyConfigs(ParentConfigs):
    p2: str = 'p2_default'
    no_default: str
    with_primary: ModuleWithPrimary
    without_primary: Module


@option(MyConfigs.with_primary)
def module_with_primary(c: MyConfigs):
    conf = ModuleWithPrimary()
    conf.p1_m1 = c.p1 + c.no_default + ' m1'
    return conf


@option(MyConfigs.without_primary)
def module_without_primary(c: MyConfigs):
    conf = Module()
    conf.p1_m1 = c.p1 + ' m1'
    return conf


# TEST: This should fail
# @option(MyConfigs.undefined)
# def undefined_config(c: MyConfigs):
#     return c.p1


def test():
    configs = MyConfigs()
    configs.p2 = 'p2_set'

    try:
        configs.p3 = 'not defined'
    except AttributeError:
        pass
    else:
        assert False

    assert configs.p1 == 'p1_default'
    assert configs.p2 == 'p2_set'
    assert configs.calc1 == 'p1_defaultp2_set calc'

    try:
        print(configs.with_primary)
    except AttributeError:
        pass
    else:
        assert False

    configs.no_default = 'set'

    assert configs.with_primary == 'p1_defaultset m1 main'
    assert configs.without_primary.m_calc == 'p1_default m1 main no primary'

    print(yaml.dump(configs.to_json()))


if __name__ == '__main__':
    test()
