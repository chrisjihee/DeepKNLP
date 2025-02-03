import logging

import pandas as pd
import typer
from typing_extensions import Annotated

from chrisbase.data import AppTyper, JobTimer, NewProjectEnv
from chrisbase.data import FileOption, FileStreamer
from chrisbase.io import files
from chrisbase.io import new_path

# Global settings
main = AppTyper()
logger: logging.Logger = logging.getLogger("DeepKNLP")


@main.command("summarize")
def summarize(
        # for CustomDataArguments
        input_dir: Annotated[str, typer.Argument()] = ...,  # "output/GNER-Baseline/FLAN-T5-Large-Baseline",
        output_file: Annotated[str, typer.Option("--output_file")] = "",
        logging_level: Annotated[int, typer.Option("--logging_level")] = logging.INFO,
):
    env = NewProjectEnv(logging_level=logging_level)
    if not output_file:
        output_file = new_path(input_dir, post="eval").with_suffix(".csv")
    with (
        JobTimer(f"python {env.current_file} {' '.join(env.command_args)}", rt=1, rb=1, rc='=', verbose=logging_level <= logging.INFO),
        FileStreamer(FileOption.from_path(path=output_file, mode="w")) as output_file,
        FileStreamer(FileOption.from_path(path=input_dir)) as input_dir,
    ):
        no_interest_columns = ['eval_average', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second']
        dfs = []
        for input_file in files(input_dir.path / "train-metrics-*.csv"):
            df = pd.read_csv(input_file)
            eval_columns = [col for col in df.columns if col.startswith("eval_") and col not in no_interest_columns]
            df = df.dropna(subset=eval_columns, how="all")[["epoch"] + eval_columns]
            df["epoch"] = df["epoch"].round(1)
            dfs.append(df)
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df = merged_df.groupby("epoch").first().reset_index()
        eval_columns = [col for col in merged_df.columns if col.startswith("eval_")]
        merged_df["eval_average"] = merged_df[eval_columns].mean(axis=1)
        merged_df = merged_df[["epoch", "eval_average"] + eval_columns]
        print(merged_df)
        merged_df.to_csv(output_file.path, index=False)


if __name__ == "__main__":
    main()
