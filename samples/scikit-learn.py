from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score

from lab import tracker, experiment
from lab.configs import BaseConfigs


class Configs(BaseConfigs):
    c = 0.2

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

    return (x_train_scaled, y_train), (x_test_scaled, y_test)


@Configs.calc(Configs.model)
def model(c: Configs):
    m = LogisticRegression(C=c.c)

    return m


def main():
    conf = Configs()
    experiment.create(writers={'sqlite'})
    experiment.calculate_configs(conf)

    experiment.add_sklearn_models(dict(model=conf.model))
    experiment.start()
    conf.run()

    experiment.save_checkpoint()


if __name__ == '__main__':
    main()
