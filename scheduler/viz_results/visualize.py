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
    avg_df = df.groupby(["NumTasks", "Algorithm"], as_index=False).agg({"Makespan": "mean"})

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(
        data=avg_df,
        x="NumTasks",
        y="Makespan",
        hue="Algorithm",
        style="Algorithm",
        markers=True,
        ax=ax,
        linewidth=2,
        palette="Set2",
        legend=True,
    )

    # Customize the plot
    ax.set_title("Average Makespan Trends Across Dataset Sizes")
    ax.set_ylabel("Average Makespan (s)")
    ax.set_xlabel("Dataset Size")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Algorithm")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(tyro.cli(Args))
