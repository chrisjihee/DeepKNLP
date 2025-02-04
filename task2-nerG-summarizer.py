import logging

import pandas as pd
import typer
from typing_extensions import Annotated

from chrisbase.data import AppTyper, JobTimer, NewProjectEnv
from chrisbase.io import files, dirs

# Global settings
main = AppTyper()
logger: logging.Logger = logging.getLogger("DeepKNLP")


@main.command("summarize")
def summarize(
        input_dirs: Annotated[str, typer.Argument()] = ...,  # "output/GNER-Baseline/*", "output/GNER-MR_EQ/*"
        csv_filename: Annotated[str, typer.Option("--csv_filename")] = "train-metrics-*.csv",
        logging_level: Annotated[int, typer.Option("--logging_level")] = logging.INFO,
):
    env = NewProjectEnv(logging_level=logging_level)
    with (
        JobTimer(f"python {env.current_file} {' '.join(env.command_args)}", rt=1, rb=1, rc='=', verbose=logging_level <= logging.INFO),
    ):
        no_interest_columns = ['eval_average', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second']
        for input_dir in dirs(input_dirs):
            output_file = input_dir.with_suffix(".csv")
            logger.info("[input_dir] %s", input_dir)
            dfs = []
            for input_file in files(input_dir / csv_filename):
                df = pd.read_csv(input_file)
                eval_columns = [col for col in df.columns if col.startswith("eval_") and col not in no_interest_columns]
                df = df.dropna(subset=eval_columns, how="all")[["epoch"] + eval_columns]
                if df.shape[0] == 0:
                    continue
                df["epoch"] = df["epoch"].round(1)
                dfs.append(df)
            if dfs:
                merged_df = pd.concat(dfs, ignore_index=True)
                merged_df = merged_df.groupby("epoch").first().reset_index()
                eval_columns = [col for col in merged_df.columns if col.startswith("eval_")]
                merged_df["eval_average"] = merged_df[eval_columns].mean(axis=1)
                merged_df = merged_df[["epoch", "eval_average"] + eval_columns]
                merged_df.to_csv(output_file, index=False)
                logger.info(f"            >> {len(dfs)} files are merged into {output_file}")
            else:
                logger.info(f"            >> No files or no evaluation output in input folder")


if __name__ == "__main__":
    main()
