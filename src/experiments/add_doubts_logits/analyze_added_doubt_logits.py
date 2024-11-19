from heapq import merge
import pandas as pd
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple
import plotly.express as px
import json
from pathlib import Path
import pandas as pd
from src.consts import PATHS
import matplotlib.pyplot as pltcha
import seaborn as sns
from src.types import DATASETS, DatasetArgs
from src.datasets.download_dataset import load_custom_dataset
from functools import lru_cache


def load_experiment_results(
    prompt_title: str,
    dataset_args: DatasetArgs,
    output_path: Path = PATHS.OUTPUT_DIR / "add_doubt_logits_diff",
) -> Dict[str, Any]:
    """Load experiment results from JSON files."""
    results = {}
    suffix = f"_{dataset_args.dataset_name}.json"

    for json_file in (output_path / prompt_title).glob("*.json"):
        if not json_file.name.endswith(suffix) or json_file.name.startswith("test"):
            continue
        model_name = json_file.name[: -len(suffix)]
        with open(json_file) as f:
            results[model_name] = json.load(f)

    return results


def convert_results_to_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """Convert nested results structure to a pandas DataFrame."""
    rows = []
    for model_name, model_results in results.items():
        for condition, entries in model_results.items():
            correct_first, correct_response = map(
                lambda x: x == "True", condition.split("_")
            )

            for idx, entry in enumerate(entries):
                row = {
                    "index": idx,
                    "model": model_name,
                    "correct_first": correct_first,
                    "correct_response": correct_response,
                }
                for prefix in ["first", "last"]:
                    row[f"{prefix}_diff"] = entry[f"{prefix}_generated_stats"]["diff"]
                    row[f"{prefix}_top_5_tokens"] = entry[f"{prefix}_generated_stats"][
                        "top_5_tokens"
                    ]
                    row[f"{prefix}_top_5_probs"] = entry[f"{prefix}_generated_stats"][
                        "top_5_probs"
                    ]
                    for correctness in ["correct", "wrong"]:
                        row[f"{prefix}_{correctness}_prob"] = entry[
                            f"{prefix}_generated_stats"
                        ][correctness]

                rows.append(row)

    return pd.DataFrame(rows)


def fix_mistral_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the data for the mistral_8x7B model.
    For this model, there was a bug in the calculation of 'first_correct_prob', 'first_wrong_prob', 'last_correct_prob', 'last_wrong_prob'
    which also affected the calculation of 'first_diff', 'last_diff'.
    We will fix this by recalculating these values by checking if a or b (based on correct_first) is in the (first/last)_top_5_tokens
    and assign the corresponding probability to 'first_correct_prob', 'first_wrong_prob', 'last_correct_prob', 'last_wrong_prob'.
    """

    def fix_record(row):
        if row["model"] != "mistral_8x7B":
            return row

        correct_first = row["correct_first"]
        correct_answer = "a" if correct_first else "b"
        wrong_answer = "b" if correct_first else "a"

        for prefix in ["first", "last"]:
            try:
                correct_prob = row[f"{prefix}_top_5_probs"][
                    row[f"{prefix}_top_5_tokens"].index(correct_answer)
                ]
            except ValueError:
                correct_prob = 0.0

            try:
                wrong_prob = row[f"{prefix}_top_5_probs"][
                    row[f"{prefix}_top_5_tokens"].index(wrong_answer)
                ]
            except ValueError:
                wrong_prob = 0.0

            row[f"{prefix}_correct_prob"] = correct_prob
            row[f"{prefix}_wrong_prob"] = wrong_prob
            row[f"{prefix}_diff"] = correct_prob - wrong_prob

        return row

    return df.apply(fix_record, axis=1)


def calculate_average_differences(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate average differences per model and condition."""
    return (
        df.groupby(["model", "correct_first", "correct_response"])[
            ["first_diff", "last_diff"]
        ]
        .mean()
        .round(3)
    )


def calculate_average_differences_with_correctness(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate average differences per model and condition, including correctness."""

    # Calculate correctness for each row in the original dataframe
    df["was_correct"] = df["first_diff"] > 0

    # Calculate group sizes
    size_per_model = df.groupby("model").size()
    group_sizes = (
        (df.groupby(["model", "was_correct"]).size() / size_per_model) * 100
    ).reset_index(name="group_size")

    df = pd.merge(
        df,
        group_sizes,
        on=["model", "was_correct"],
    )

    return (
        df.groupby(["model", "correct_first", "correct_response", "was_correct"])
        .agg({"first_diff": "mean", "last_diff": "mean", "group_size": "first"})
        .round(3)
    )


def get_test_mappings() -> Dict[str, str]:
    """Get mapping of test conditions to their descriptions."""
    return {
        "first_diffTrueTrue": "(1st Diff, Correct 1st)",  # The difference in the 1st answer, where the correct answer is presented 1st
        "first_diffFalseTrue": "(1st Diff, Correct 2nd)",  # The difference in the 1st answer, where the correct answer is presented 2nd
        "last_diffTrueTrue": "(2nd Diff, Correct 1st, Correct 1st)",  # The difference in the 2nd answer, where the correct answer is presented 1st, and the first response was a correct
        "last_diffTrueFalse": "(2nd Diff, Correct 1st, Mistake 1st)",  # The difference in the 2nd answer, where the correct answer is presented 1st, and the first response was a mistake
        "last_diffFalseTrue": "(2nd Diff, Correct 2nd, Correct 1st)",  # The difference in the 2nd answer, where the correct answer is presented 2nd, and the first response was a correct
        "last_diffFalseFalse": "(2nd Diff, Correct 2nd, Mistake 1st)",  # The difference in the 2nd answer, where the correct answer is presented 2nd, and the first response was a mistake
    }


def prepare_plotting_data(avg_diffs: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for plotting by melting and organizing conditions."""
    avg_diffs_reset = avg_diffs.reset_index()

    melted_avg_diffs = (
        avg_diffs_reset.melt(
            id_vars=["model", "correct_first", "correct_response"],
            value_vars=["first_diff", "last_diff"],
            var_name="diff_type",
            value_name="diff_value",
        )
        .loc[
            lambda x: (x["diff_type"] == "last_diff") | (x["correct_response"] == True)
        ]
        .assign(
            test=lambda x: x["diff_type"]
            + x["correct_first"].astype(str)
            + x["correct_response"].astype(str)
        )
        .replace({"test": get_test_mappings()})
        .assign(
            **{
                "test": lambda df: pd.Categorical(
                    df["test"], categories=get_test_mappings().values(), ordered=True
                )
            }
        )
        .sort_values(["model", "test"])
        .reset_index(drop=True)
    )

    return melted_avg_diffs


def plot_grouped_differences(df: pd.DataFrame):
    """Create a grouped bar plot of the differences."""
    return px.bar(
        df,
        x="model",
        y="diff_value",
        color="test",
        barmode="group",
    )


def merge_with_dataset_questions(
    df: pd.DataFrame, dataset_args: DatasetArgs
) -> pd.DataFrame:
    # Load the original dataset
    dataset_args = DatasetArgs(name=DATASETS.COUNTER_FACT, splits="train1")
    ds = load_custom_dataset(dataset_args)
    ds_df = pd.DataFrame(ds)

    # Expand the original DataFrame with question/answers repeated for each condition
    expanded_rows = []
    for _, row in ds_df.iterrows():
        for correct_first in [True, False]:
            for correct_response in [True, False]:
                expanded_rows.append(
                    {
                        "index": row.name,
                        "question": row["prompt"],
                        "first_answer": (
                            row["target_true"] if correct_first else row["target_false"]
                        ),
                        "second_answer": (
                            row["target_false"] if correct_first else row["target_true"]
                        ),
                        "correct_first": correct_first,
                        "correct_response": correct_response,
                    }
                )

    ds_expanded_df = pd.DataFrame(expanded_rows)
    assert (
        df.shape[0] == ds_expanded_df.shape[0] * df["model"].nunique()
    ), "Mismatch in number of rows"

    # Merge the logits data with the dataset
    df_with_qa = pd.merge(
        df,
        ds_expanded_df,
        on=["index", "correct_first", "correct_response"],
        # how='left'
    )

    # # Add some useful derived columns
    # df_with_qa['chosen_answer'] = df_with_qa.apply(
    #     lambda x: x['first_answer'] if x['correct_response'] == x['correct_first'] else x['second_answer'],
    #     axis=1
    # )

    # df_with_qa['correct_answer'] = df_with_qa.apply(
    #     lambda x: x['first_answer'] if x['correct_first'] else x['second_answer'],
    #     axis=1
    # )

    assert df_with_qa.shape[0] == df.shape[0], "Merging not successful"

    return df_with_qa


def plot_scatter_with_questions(df_with_qa: pd.DataFrame):
    """Create an interactive scatter plot with question details on hover."""
    fig = px.scatter(
        df_with_qa,
        x="first_diff",
        y="last_diff",
        color="model",
        hover_data=["question", "first_answer", "second_answer", "index"],
        title="First vs Last Logit Differences with Question Details",
        facet_row="correct_first",
        facet_col="correct_response",
        labels={
            "first_diff": "",
            "last_diff": "",
            "index": "Question Index",
            "question": "Question",
        },
        opacity=0.01,
    )

    # Add a diagonal line y=x
    fig.add_scatter(
        x=[-1, 1],
        y=[-1, 1],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="No Change Line",
    )

    fig.update_layout(
        xaxis_title="Initial Response Difference",
        yaxis_title="Post-Doubt Response Difference",
    )

    return fig


def plot_confidence_changes(df_with_qa: pd.DataFrame):
    """Plot how confidence changes before and after doubt by condition."""
    fig = px.box(
        df_with_qa,
        x="model",
        y=["first_diff", "last_diff"],
        facet_row="correct_first",
        facet_col="correct_response",
        title="Confidence Changes by Condition",
        labels={
            "value": "Logit Difference",
            "variable": "Response Stage",
        },
        hover_data=["index", "question", "first_answer", "second_answer"],
    )

    # Update the opacity of outliers
    fig.update_traces(marker=dict(opacity=0.01))

    fig.update_layout(
        xaxis_title="Model",
        yaxis_title="Logit Difference",
    )

    return fig


def plot_question_specific_responses(df_with_qa: pd.DataFrame, question_index: int):
    """Create a detailed plot for a specific question."""
    question_df = df_with_qa[df_with_qa["index"] == question_index]

    fig = px.scatter(
        question_df,
        x="first_diff",
        y="last_diff",
        color="model",
        facet_row="correct_first",
        facet_col="correct_response",
        title=f'Question Analysis: {question_df.iloc[0]["question"][:100]}...',
        labels={"first_diff": "", "last_diff": ""},
        hover_data=["first_answer", "second_answer"],
    )

    # Update layout to show labels only once
    fig.update_layout(
        xaxis_title="Initial Response Difference",
        yaxis_title="Post-Doubt Response Difference",
    )
    return fig


def prepare_plotting_data_with_correctness(avg_diffs: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for plotting by melting and organizing conditions, including first response correctness."""
    # Prepare the average differences data
    avg_diffs_reset = avg_diffs.reset_index()

    melted_avg_diffs = (
        avg_diffs_reset.melt(
            id_vars=[
                "model",
                "correct_first",
                "correct_response",
                "was_correct",
                "group_size",
            ],
            value_vars=["first_diff", "last_diff"],
            var_name="diff_type",
            value_name="diff_value",
        )
        .loc[
            lambda x: (x["diff_type"] == "last_diff") | (x["correct_response"] == True)
        ]
        .assign(
            test=lambda x: x["diff_type"]
            + x["correct_first"].astype(str)
            + x["correct_response"].astype(str)
        )
        .replace({"test": get_test_mappings()})
        .assign(
            **{
                "test": lambda df: pd.Categorical(
                    df["test"], categories=get_test_mappings().values(), ordered=True
                )
            }
        )
        .sort_values(["was_correct"], ascending=False)
        .sort_values(["model", "test"])
    )

    return melted_avg_diffs


def plot_grouped_differences_with_correctness(df: pd.DataFrame):
    """Create a grouped bar plot of the differences split by initial response correctness."""
    fig = px.bar(
        df,
        x="model",
        y="diff_value",
        color="test",
        barmode="group",
        facet_row="was_correct",
        title="Model Performance by Initial Correctness",
        labels={
            "diff_value": "Logit Difference",
            "was_correct": "Initially Correct",
        },
        hover_data=["group_size"],
    )

    # Update facet row labels
    fig.for_each_annotation(
        lambda a: a.update(
            text=f"Initially {'Correct' if a.text.endswith('True') else 'Incorrect'}"
        )
    )

    return fig

def calculate_response_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the percentages of response changes for each model."""
    
    # Calculate correctness for first and last responses
    df['first_correct'] = df['first_diff'] > 0
    df['last_correct'] = df['last_diff'] > 0
       
    # Create categories
    def get_change_category(row):
        if row['first_correct'] and not row['last_correct']:
            return 'V→X'
        elif not row['first_correct'] and row['last_correct']:
            return 'X→V'
        elif row['first_correct'] and row['last_correct']:
            return 'V→V'
        else:
            return 'X→X'
    
    df['change_category'] = df.apply(get_change_category, axis=1)
    
    # Calculate percentages per model
    result = (df.groupby(['model', 'change_category'])
             .size()
             .unstack()
             .fillna(0))
    
    # Convert to percentages
    total_per_model = result.sum(axis=1)
    result_percentages = (result.div(total_per_model, axis=0) * 100).round(2)
    
    result_percentages['Before Doubt'] = result_percentages['V→V'] + result_percentages['V→X']
    result_percentages['After Doubt'] = result_percentages['V→V'] + result_percentages['X→V']
    
    # Add total counts as a separate column
    result_percentages['Total Samples'] = total_per_model
    
    print(result_percentages.to_markdown())

def calculate_response_changes_natural(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate the percentages of response changes for each model, only for natural responses.
    i.e. where the correct response matches the first response.
    """
    # Filter for natural responses (where correct_response matches correct_first)
    df = df[df['correct_response'] == (df['first_diff'] > 0)]
    
    # Split by correct_first and calculate separately
    print("\nResults when correct answer was presented first:")
    print("==============================================")
    calculate_response_changes(df[df['correct_first']].copy())
    
    print("\nResults when correct answer was presented second:")
    print("==============================================")
    calculate_response_changes(df[~df['correct_first']].copy())
    
    print("\nCombined results:")
    print("==============================================")
    calculate_response_changes(df.copy())

@lru_cache(maxsize=32)
def load_and_process_results(
    prompt_title: str,
    dataset_args: DatasetArgs,
    output_path: Path = PATHS.OUTPUT_DIR / "add_doubt_logits_diff",
) -> pd.DataFrame:
    """
    Load and process experiment results, including fixing Mistral data.
    Combines load_experiment_results, convert_results_to_dataframe, and fix_mistral_data.
    Results are cached based on prompt_title and dataset_args.
    """
    output_path_str = str(output_path)  # Convert Path to string for caching
    results = load_experiment_results(prompt_title, dataset_args, Path(output_path_str))
    df = convert_results_to_dataframe(results)
    return fix_mistral_data(df)


def add_derived_columns(logit_diffs: pd.DataFrame) -> pd.DataFrame:
    return (
        logit_diffs
        .pipe(lambda df : df[df['model'] != 'mistral_8x7B'])
    )
    

def main(prompt_title: str, dataset_args: DatasetArgs):
    df_fixed = load_and_process_results(prompt_title, dataset_args)
    
    
    calculate_response_changes_natural(df_fixed)
    
    return df_fixed
