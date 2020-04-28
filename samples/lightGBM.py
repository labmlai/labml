from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score

import lightgbm

from lab import tracker, experiment
from lab.configs import BaseConfigs


class Configs(BaseConfigs):
    c = 0.5

    train_data: any
    test_data: any

    model: any

    random_state = 42

    def run(self):
        x_train, y_train = self.train_data
        self.model.fit(x_train, y_train)

        x_test, y_test = self.test_data
        y_pred = self.model.predict(x_test)

        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        tracker.add(f1=f1)
        tracker.add(precision=precision)
        tracker.add(recall=recall)

        tracker.save()


@Configs.calc([Configs.train_data, Configs.test_data])
def data_loaders(c: Configs):
    cancer = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        cancer.data,
        cancer.target,
        stratify=cancer.target,
        random_state=c.random_state)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    train_data = lightgbm.Dataset(x_train_scaled, label=y_train)
    test_data = lightgbm.Dataset(x_test_scaled, label=y_test)

    return train_data, test_data


@Configs.calc(Configs.model)
def model(c: Configs):
    m = LogisticRegression(C=c.c)

    return m


def main():
    conf = Configs()
    experiment.create(writers={'sqlite'})
    experiment.calculate_configs(conf)

    experiment.add_pytorch_models(dict(model=conf.model))
    experiment.start()
    conf.run()


if __name__ == '__main__':
    main()
