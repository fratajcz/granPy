from src.experiment import ExperimentArray
from src.utils import opts
import pandas as pd


class TrainingPipeline:
    def __init__(self, opts):
        self.experimentarray = ExperimentArray(opts)

    def run(self):
        performance_reports = self.experimentarray.run()

        metrics = list(performance_reports[0].keys())

        df = pd.DataFrame()

        for metric in metrics:
            values = []
            for run in performance_reports:
                values.append(run[metric])

            df[metric] = values

        df_t = df.transpose()
        df_t["mean"] = df_t.mean(axis=1)

        return df_t


if __name__ == "__main__":
    pipeline = TrainingPipeline(opts())
    df = pipeline.run()
    df_t = df.transpose()
    df_t["mean"] = df_t.mean(axis=1)
    print(df_t)