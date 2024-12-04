import dataclasses
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import tyro


@dataclasses.dataclass
class Args:
    import_csv: str
    """File to import the CSV"""


def main(args: Args):
    df = pd.read_csv(args.import_csv)
    avg_df = df.groupby(["SettingId", "Algorithm"], as_index=False).agg({"Makespan": "mean", "EnergyJ": "mean"})
    fig, ax = plt.subplots(ncols=2, figsize=(12, 8))
    sns.boxplot(data=df, x="Algorithm", y="Makespan", ax=ax[0], gap=0, hue="Algorithm")
    sns.boxplot(data=df, x="Algorithm", y="EnergyJ", ax=ax[1], gap=0, hue="Algorithm")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(tyro.cli(Args))
