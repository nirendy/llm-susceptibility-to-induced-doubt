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


def load_experiment_results(
    dataset_args: DatasetArgs,
    output_path: Path = PATHS.OUTPUT_DIR / "add_doubt_logits_diff",
) -> Dict[str, Any]:
    """Load experiment results from JSON files."""
    results = {}
    suffix = f"_{dataset_args.dataset_name}.json"

    for json_file in output_path.glob("*.json"):
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
                    row[f"{prefix}_top_5_tokens"] = entry[f"{prefix}_generated_stats"]["top_5_tokens"]
                    row[f"{prefix}_top_5_probs"] = entry[f"{prefix}_generated_stats"]["top_5_probs"]
                    for correctness in ['correct', 'wrong']:
                        row[f"{prefix}_{correctness}_prob"] = entry[f"{prefix}_generated_stats"][correctness]
                    
                    
                
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


def get_test_mappings() -> Dict[str, str]:
    """Get mapping of test conditions to their descriptions."""
    return {
        'first_diffTrueTrue':   '(1st Diff, Correct 1st)',              # The difference in the 1st answer, where the correct answer is presented 1st
        'first_diffFalseTrue':  '(1st Diff, Correct 2nd)',              # The difference in the 1st answer, where the correct answer is presented 2nd
        'last_diffTrueTrue':    '(2nd Diff, Correct 1st, Correct 1st)', # The difference in the 2nd answer, where the correct answer is presented 1st, and the first response was a correct
        'last_diffTrueFalse':   '(2nd Diff, Correct 1st, Mistake 1st)', # The difference in the 2nd answer, where the correct answer is presented 1st, and the first response was a mistake
        'last_diffFalseTrue':   '(2nd Diff, Correct 2nd, Correct 1st)', # The difference in the 2nd answer, where the correct answer is presented 2nd, and the first response was a correct
        'last_diffFalseFalse':  '(2nd Diff, Correct 2nd, Mistake 1st)', # The difference in the 2nd answer, where the correct answer is presented 2nd, and the first response was a mistake
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
        .assign(**{
            'test': lambda df: pd.Categorical(df['test'], categories=get_test_mappings().values(), ordered=True)
        })
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


def merge_with_dataset_questions(df: pd.DataFrame, dataset_args: DatasetArgs) -> pd.DataFrame:
    # Load the original dataset
    dataset_args = DatasetArgs(name=DATASETS.COUNTER_FACT, splits="train1")
    ds = load_custom_dataset(dataset_args)
    ds_df = pd.DataFrame(ds)

    # Expand the original DataFrame with question/answers repeated for each condition
    expanded_rows = []
    for _, row in ds_df.iterrows():
        for correct_first in [True, False]:
            for correct_response in [True, False]:
                expanded_rows.append({
                    'index': row.name,
                    'question': row['prompt'],
                    'first_answer': row['target_true'] if correct_first else row['target_false'],
                    'second_answer': row['target_false'] if correct_first else row['target_true'],
                    'correct_first': correct_first,
                    'correct_response': correct_response
                })

    ds_expanded_df = pd.DataFrame(expanded_rows)
    assert df.shape[0] == ds_expanded_df.shape[0] * df['model'].nunique(), "Mismatch in number of rows"

    # Merge the logits data with the dataset
    df_with_qa = pd.merge(
        df,
        ds_expanded_df,
        on=['index', 'correct_first', 'correct_response'],
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
        x='first_diff',
        y='last_diff',
        color='model',
        hover_data=['question', 'first_answer', 'second_answer', 'index'],
        title='First vs Last Logit Differences with Question Details',
        facet_row='correct_first',
        facet_col='correct_response',
        labels={
            'first_diff': '',
            'last_diff': '',
            'index': 'Question Index',
            'question': 'Question',
        },
        opacity=0.01
    )
    
    # Add a diagonal line y=x
    fig.add_scatter(
        x=[-1, 1],
        y=[-1, 1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='No Change Line'
    )
    
    fig.update_layout(
        xaxis_title='Initial Response Difference',
        yaxis_title='Post-Doubt Response Difference',
    )
    
    return fig

def plot_confidence_changes(df_with_qa: pd.DataFrame):
    """Plot how confidence changes before and after doubt by condition."""
    fig = px.box(
        df_with_qa,
        x='model',
        y=['first_diff', 'last_diff'],
        facet_row='correct_first',
        facet_col='correct_response',
        title='Confidence Changes by Condition',
        labels={
            'value': 'Logit Difference',
            'variable': 'Response Stage',
        },
        hover_data=['index', 'question', 'first_answer', 'second_answer']
    )
    
    # Update the opacity of outliers
    fig.update_traces(marker=dict(opacity=0.01))
    
    fig.update_layout(
        xaxis_title='Model',
        yaxis_title='Logit Difference',
    )
    
    return fig

def plot_question_specific_responses(df_with_qa: pd.DataFrame, question_index: int):
    """Create a detailed plot for a specific question."""
    question_df = df_with_qa[df_with_qa['index'] == question_index]
    
    fig = px.scatter(
        question_df,
        x='first_diff',
        y='last_diff',
        color='model',
        facet_row='correct_first',
        facet_col='correct_response',
        title=f'Question Analysis: {question_df.iloc[0]["question"][:100]}...',
        labels={
            'first_diff': '',
            'last_diff': ''
        },
        hover_data=['first_answer', 'second_answer']
    )

    # Update layout to show labels only once
    fig.update_layout(
        xaxis_title='Initial Response Difference',
        yaxis_title='Post-Doubt Response Difference',
    )
    return fig

def main(dataset_args: DatasetArgs):
    results = load_experiment_results(dataset_args)
    df = convert_results_to_dataframe(results)
    df_fixed = fix_mistral_data(df)
    avg_diffs = calculate_average_differences(df_fixed)
    plotting_data = prepare_plotting_data(avg_diffs)

    grouped_diffs_plot = plot_grouped_differences(plotting_data)
    grouped_diffs_plot.show()
    
    merged_df = merge_with_dataset_questions(df_fixed, dataset_args)
    
    # Add new visualizations
    scatter_plot = plot_scatter_with_questions(merged_df)
    scatter_plot.show()
    
    confidence_plot = plot_confidence_changes(merged_df)
    confidence_plot.show()
    
    # Example for a specific question (using first question)
    question_plot = plot_question_specific_responses(merged_df, 0)
    question_plot.show()
    
    return merged_df
