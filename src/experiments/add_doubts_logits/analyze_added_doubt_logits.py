from math import log
import pandas as pd
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple
import plotly.express as px
import json
from pathlib import Path
import pandas as pd
from regex import T
from src.consts import PATHS
from src.types import DATASETS, DatasetArgs
from src.datasets.download_dataset import load_custom_dataset
from functools import lru_cache

MODELS_ORDER = [
    "llama3_2_1B",
    "llama3_2_3B",
    "phi_3.5-mini",
    "llama3_1_8B",
    # "mistral_8x7B", # TODO: bring back
    "mistral_Nemo",
]


class JSON_C:
    DIFF = "diff"
    GENERATED_STATS = "_generated_stats"
    TOP_5_TOKENS = "top_5_tokens"
    TOP_5_PROBS = "top_5_probs"


class PART_C:
    FIRST = "first"
    LAST = "last"
    CORRECT = "correct"
    WRONG = "wrong"
    DIFF = "_diff"
    TOP_5_TOKENS = "_top_5_tokens"
    TOP_5_PROBS = "_top_5_probs"
    PROB = "_prob"
    PREFIXES = [FIRST, LAST]
    CORRECTNESS = [CORRECT, WRONG]


# columns
class C:
    # original columns
    # from anlysis
    INDEX = "index"
    MODEL = "model"
    CORRECT_FIRST = "correct_first"
    CORRECT_RESPONSE = "correct_response"
    FIRST_DIFF = PART_C.FIRST + PART_C.DIFF
    LAST_DIFF = PART_C.LAST + PART_C.DIFF
    FIRST_TOP_5_TOKENS = PART_C.FIRST + PART_C.TOP_5_TOKENS
    FIRST_TOP_5_PROBS = PART_C.FIRST + PART_C.TOP_5_PROBS
    FIRST_CORRECT_PROB = PART_C.FIRST + "_" + PART_C.CORRECT + PART_C.PROB
    FIRST_WRONG_PROB = PART_C.FIRST + "_" + PART_C.WRONG + PART_C.PROB

    # From dataset
    QUESTION = "question"
    FIRST_ANSWER = "first_answer"
    SECOND_ANSWER = "second_answer"

    # added columns
    WAS_CORRECT = "was_correct"
    NATURAL_RESPONSE = "natural_response"
    GROUP_SIZE = "group_size"
    TEST = "test"
    CHANGE_CATEGORY = "change_category"

    # plotting and analysis columns
    DIFF_TYPE = "diff_type"
    DIFF_VALUE = "diff_value"
    LAST_CORRECT = "last_correct"
    TOTAL_SAMPLES = "Total Samples"
    BEFORE_DOUBT = "Before Doubt"
    AFTER_DOUBT = "After Doubt"
    CHANGE_V_TO_X = "V→X"
    CHANGE_X_TO_V = "X→V"
    CHANGE_V_TO_V = "V→V"
    CHANGE_X_TO_X = "X→X"


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
                    C.INDEX: idx,
                    C.MODEL: model_name,
                    C.CORRECT_FIRST: correct_first,
                    C.CORRECT_RESPONSE: correct_response,
                }
                for prefix in PART_C.PREFIXES:
                    row[f"{prefix}{PART_C.DIFF}"] = entry[
                        f"{prefix}{JSON_C.GENERATED_STATS}"
                    ][JSON_C.DIFF]
                    row[f"{prefix}{PART_C.TOP_5_TOKENS}"] = entry[
                        f"{prefix}{JSON_C.GENERATED_STATS}"
                    ][JSON_C.TOP_5_TOKENS]
                    row[f"{prefix}{PART_C.TOP_5_PROBS}"] = entry[
                        f"{prefix}{JSON_C.GENERATED_STATS}"
                    ][JSON_C.TOP_5_PROBS]
                    for correctness in PART_C.CORRECTNESS:
                        row[f"{prefix}_{correctness}{PART_C.PROB}"] = entry[
                            f"{prefix}{JSON_C.GENERATED_STATS}"
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
        if row[C.MODEL] != "mistral_8x7B":
            return row

        correct_first = row[C.CORRECT_FIRST]
        correct_answer = "a" if correct_first else "b"
        wrong_answer = "b" if correct_first else "a"

        for prefix in PART_C.PREFIXES:
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
        df.groupby([C.MODEL, C.CORRECT_FIRST, C.CORRECT_RESPONSE])[
            [C.FIRST_DIFF, C.LAST_DIFF]
        ]
        .mean()
        .round(3)
    )


def calculate_average_differences_with_correctness(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate average differences per model and condition, including correctness."""

    # Calculate group sizes
    size_per_model = df.groupby(C.MODEL).size()
    group_sizes = (
        (df.groupby([C.MODEL, C.WAS_CORRECT]).size() / size_per_model) * 100
    ).reset_index(name=C.GROUP_SIZE)

    df = pd.merge(
        df,
        group_sizes,
        on=[C.MODEL, C.WAS_CORRECT],
    )

    return (
        df.groupby([C.MODEL, C.CORRECT_FIRST, C.CORRECT_RESPONSE, C.WAS_CORRECT])
        .agg({C.FIRST_DIFF: "mean", C.LAST_DIFF: "mean", C.GROUP_SIZE: "first"})
        .round(3)
    )


def get_test_mappings() -> Dict[str, str]:
    """Get mapping of test conditions to their descriptions."""
    return {
        "first_diffTrueTrue": "(1st Diff, Correct 1st)",
        # The difference in the 1st answer, where the correct answer is presented 1st
        "first_diffFalseTrue": "(1st Diff, Correct 2nd)",
        # The difference in the 1st answer, where the correct answer is presented 2nd
        "last_diffTrueTrue": "(2nd Diff, Correct 1st, Correct 1st)",
        # The difference in the 2nd answer, where the correct answer is presented 1st, and the first response was a correct
        "last_diffTrueFalse": "(2nd Diff, Correct 1st, Mistake 1st)",
        # The difference in the 2nd answer, where the correct answer is presented 1st, and the first response was a mistake
        "last_diffFalseTrue": "(2nd Diff, Correct 2nd, Correct 1st)",
        # The difference in the 2nd answer, where the correct answer is presented 2nd, and the first response was a correct
        "last_diffFalseFalse": "(2nd Diff, Correct 2nd, Mistake 1st)",
        # The difference in the 2nd answer, where the correct answer is presented 2nd, and the first response was a mistake
    }


def prepare_plotting_data(avg_diffs: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for plotting by melting and organizing conditions."""
    avg_diffs_reset = avg_diffs.reset_index()

    melted_avg_diffs = (
        avg_diffs_reset.melt(
            id_vars=[C.MODEL, C.CORRECT_FIRST, C.CORRECT_RESPONSE],
            value_vars=[C.FIRST_DIFF, C.LAST_DIFF],
            var_name=C.DIFF_TYPE,
            value_name=C.DIFF_VALUE,
        )
        .loc[
            lambda x: (x[C.DIFF_TYPE] == C.LAST_DIFF) | (x[C.CORRECT_RESPONSE] == True)
        ]
        .assign(
            test=lambda x: x[C.DIFF_TYPE]
            + x[C.CORRECT_FIRST].astype(str)
            + x[C.CORRECT_RESPONSE].astype(str)
        )
        .replace({"test": get_test_mappings()})
        .assign(
            **{
                C.TEST: lambda df: pd.Categorical(
                    df[C.TEST], categories=get_test_mappings().values(), ordered=True
                )
            }
        )
        .sort_values([C.MODEL, C.TEST])
        .reset_index(drop=True)
    )

    return melted_avg_diffs


def plot_grouped_differences(df: pd.DataFrame):
    """Create a grouped bar plot of the differences."""
    return px.bar(
        df,
        x=C.MODEL,
        y=C.DIFF_VALUE,
        color=C.TEST,
        barmode="group",
    )


def merge_with_dataset_questions(
    df: pd.DataFrame, dataset_args: DatasetArgs
) -> pd.DataFrame:
    # Load the original dataset
    ds = load_custom_dataset(dataset_args)
    ds_df = pd.DataFrame(ds)

    # Expand the original DataFrame with question/answers repeated for each condition
    expanded_rows = []
    for _, row in ds_df.iterrows():
        for correct_first in [True, False]:
            for correct_response in [True, False]:
                expanded_rows.append(
                    {
                        C.INDEX: row.name,
                        C.QUESTION: row["prompt"],
                        C.FIRST_ANSWER: (
                            row["target_true"] if correct_first else row["target_false"]
                        ),
                        C.SECOND_ANSWER: (
                            row["target_false"] if correct_first else row["target_true"]
                        ),
                        C.CORRECT_FIRST: correct_first,
                        C.CORRECT_RESPONSE: correct_response,
                    }
                )

    ds_expanded_df = pd.DataFrame(expanded_rows)
    assert (
        df.shape[0] == ds_expanded_df.shape[0] * df[C.MODEL].nunique()
    ), "Mismatch in number of rows"

    # Merge the logits data with the dataset
    df_with_qa = pd.merge(
        df,
        ds_expanded_df,
        on=[C.INDEX, C.CORRECT_FIRST, C.CORRECT_RESPONSE],
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
        x=C.FIRST_DIFF,
        y=C.LAST_DIFF,
        color=C.MODEL,
        hover_data=[C.QUESTION, C.FIRST_ANSWER, C.SECOND_ANSWER, C.INDEX],
        title="First vs Last Logit Differences with Question Details",
        facet_row=C.CORRECT_FIRST,
        facet_col=C.CORRECT_RESPONSE,
        labels={
            C.FIRST_DIFF: "",
            C.LAST_DIFF: "",
            C.INDEX: "Question Index",
            C.QUESTION: "Question",
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
        x=C.MODEL,
        y=[C.FIRST_DIFF, C.LAST_DIFF],
        facet_row=C.CORRECT_FIRST,
        facet_col=C.CORRECT_RESPONSE,
        title="Confidence Changes by Condition",
        labels={
            "value": "Logit Difference",
            "variable": "Response Stage",
        },
        hover_data=[C.INDEX, C.QUESTION, C.FIRST_ANSWER, C.SECOND_ANSWER],
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
    question_df = df_with_qa[df_with_qa[C.INDEX] == question_index]

    fig = px.scatter(
        question_df,
        x=C.FIRST_DIFF,
        y=C.LAST_DIFF,
        color=C.MODEL,
        facet_row=C.CORRECT_FIRST,
        facet_col=C.CORRECT_RESPONSE,
        title=f"Question Analysis: {question_df.iloc[0][C.QUESTION][:100]}...",
        labels={C.FIRST_DIFF: "", C.LAST_DIFF: ""},
        hover_data=[C.FIRST_ANSWER, C.SECOND_ANSWER],
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
                C.MODEL,
                C.CORRECT_FIRST,
                C.CORRECT_RESPONSE,
                C.WAS_CORRECT,
                C.GROUP_SIZE,
            ],
            value_vars=[C.FIRST_DIFF, C.LAST_DIFF],
            var_name=C.DIFF_TYPE,
            value_name=C.DIFF_VALUE,
        )
        .loc[
            lambda x: (x[C.DIFF_TYPE] == C.LAST_DIFF) | (x[C.CORRECT_RESPONSE] == True)
        ]
        .assign(
            **{
                C.TEST: lambda x: x[C.DIFF_TYPE]
                + x[C.CORRECT_FIRST].astype(str)
                + x[C.CORRECT_RESPONSE].astype(str)
            }
        )
        .replace({C.TEST: get_test_mappings()})
        .assign(
            **{
                C.TEST: lambda df: pd.Categorical(
                    df[C.TEST], categories=get_test_mappings().values(), ordered=True
                )
            }
        )
        .sort_values([C.WAS_CORRECT], ascending=False)
        .sort_values([C.MODEL, C.TEST])
    )

    return melted_avg_diffs


def plot_grouped_differences_with_correctness(df: pd.DataFrame):
    """Create a grouped bar plot of the differences split by initial response correctness."""
    fig = px.bar(
        df,
        x=C.MODEL,
        y=C.DIFF_VALUE,
        color=C.TEST,
        barmode="group",
        facet_row=C.WAS_CORRECT,
        title="Model Performance by Initial Correctness",
        labels={
            C.DIFF_VALUE: "Logit Difference",
            C.WAS_CORRECT: "Initially Correct",
        },
        hover_data=[C.GROUP_SIZE],
    )

    # Update facet row labels
    fig.for_each_annotation(
        lambda a: a.update(
            text=f"Initially {'Correct' if a.text.endswith('True') else 'Incorrect'}"
        )
    )

    return fig


def calculate_response_changes(df: pd.DataFrame):
    """Calculate the percentages of response changes for each model."""
    # Calculate percentages per model
    result = df.groupby([C.MODEL, C.CHANGE_CATEGORY]).size().unstack().fillna(0)

    # Convert to percentages
    total_per_model = result.sum(axis=1)
    result_percentages = (result.div(total_per_model, axis=0) * 100).round(2)

    result_percentages[C.BEFORE_DOUBT] = (
        result_percentages[C.CHANGE_V_TO_V] + result_percentages[C.CHANGE_V_TO_X]
    )
    result_percentages[C.AFTER_DOUBT] = (
        result_percentages[C.CHANGE_V_TO_V] + result_percentages[C.CHANGE_X_TO_V]
    )

    # Add total counts as a separate column
    result_percentages[C.TOTAL_SAMPLES] = total_per_model

    print(result_percentages.to_markdown())


def calculate_response_changes_natural(
    df: pd.DataFrame,
):
    """
    Calculate the percentages of response changes for each model, only for natural responses.
    i.e. where the correct response matches the first response.
    """
    # Split by correct_first and calculate separately
    print("\nResults when correct answer was presented first:")
    print("==============================================")
    calculate_response_changes(df[df[C.CORRECT_FIRST] & df[C.NATURAL_RESPONSE]])

    print("\nResults when correct answer was presented second:")
    print("==============================================")
    calculate_response_changes(df[~df[C.CORRECT_FIRST] & df[C.NATURAL_RESPONSE]])

    print("\nCombined results:")
    print("==============================================")
    calculate_response_changes(df[df[C.NATURAL_RESPONSE]])


def add_derived_columns(logit_diffs: pd.DataFrame) -> pd.DataFrame:
    # Create categories
    def get_change_category(row):
        if row[C.WAS_CORRECT] and not row[C.LAST_CORRECT]:
            return C.CHANGE_V_TO_X
        elif not row[C.WAS_CORRECT] and row[C.LAST_CORRECT]:
            return C.CHANGE_X_TO_V
        elif row[C.WAS_CORRECT] and row[C.LAST_CORRECT]:
            return C.CHANGE_V_TO_V
        else:
            return C.CHANGE_X_TO_X

    return logit_diffs.assign(
        **{
            C.WAS_CORRECT: lambda df: df[C.FIRST_DIFF] > 0,
            C.LAST_CORRECT: lambda df: df[C.LAST_DIFF] > 0,
            C.CHANGE_CATEGORY: lambda df: df.apply(get_change_category, axis=1),
            C.MODEL: lambda df: pd.Categorical(
                df[C.MODEL], categories=MODELS_ORDER, ordered=True
            ),
            C.NATURAL_RESPONSE: lambda df: df[C.WAS_CORRECT] == df[C.CORRECT_RESPONSE],
        }
    )


# @lru_cache(maxsize=32)
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
    return (
        convert_results_to_dataframe(results)
        .pipe(fix_mistral_data)
        .pipe(merge_with_dataset_questions, dataset_args)
        .pipe(add_derived_columns)
    )


def main(prompt_title: str, dataset_args: DatasetArgs):
    df_fixed = load_and_process_results(prompt_title, dataset_args).pipe(
        lambda df: df[df[C.MODEL] != "mistral_8x7B"]
    )

    calculate_response_changes_natural(df_fixed)

    return df_fixed
