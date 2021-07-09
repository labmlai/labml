from fastai.vision.all import untar_data, URLs, ImageDataLoaders, get_image_files, Resize, error_rate, resnet34, \
    cnn_learner

from labml import lab, experiment
from labml.utils.fastai import LabMLFastAICallback

path = untar_data(URLs.PETS, dest=lab.get_data_path(), fname=lab.get_data_path() / URLs.path(URLs.PETS).name) / 'images'


def is_cat(x): return x[0].isupper()


dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path),
    valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))
# Train the model ⚡
learn = cnn_learner(dls, resnet34, metrics=error_rate, cbs=LabMLFastAICallback())

with experiment.record(name='pets', exp_conf=learn.labml_configs()):
    learn.fine_tune(5)
