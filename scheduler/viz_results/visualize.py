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
    avg_makespan_df = df.groupby(["NumWorkflows", "Algorithm"], as_index=False).agg({"Makespan": "mean"})
    avg_energy_df = df.groupby(["NumWorkflows", "Algorithm"], as_index=False).agg({"EnergyJ": "mean"})

    fig, ax = plt.subplots(ncols=2, figsize=(12, 8))
    sns.lineplot(data=avg_makespan_df, x="NumWorkflows", y="Makespan", hue="Algorithm", style="Algorithm", ax=ax[0])
    sns.lineplot(data=avg_energy_df, x="NumWorkflows", y="EnergyJ", hue="Algorithm", style="Algorithm", ax=ax[1])

    ax[0].set_title("Average Makespan Trend with Number of Workflows")
    ax[0].set_ylabel("Average Makespan (s)")
    ax[0].set_xlabel("Number of Workflows")
    ax[0].tick_params(axis="x", rotation=45)
    ax[0].legend(title="Algorithm")

    ax[1].set_title("Average Energy Consumption Trend with Number of Workflows")
    ax[1].set_ylabel("Average Energy Consumption (J)")
    ax[1].set_xlabel("Number of Workflows")
    ax[1].tick_params(axis="x", rotation=45)
    ax[1].legend(title="Algorithm")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(tyro.cli(Args))
