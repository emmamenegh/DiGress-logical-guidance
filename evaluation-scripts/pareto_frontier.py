import os
import re
import argparse
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.stats.proportion import proportions_ztest
from pygmo import hypervolume
import matplotlib.pyplot as plt




def extract_metrics(file_path, lambda_value):
    with open(file_path, 'r') as f:
        content = f.read()

    try:
        return {
            'lambda': lambda_value,
            'Validity': round(float(re.search(r"Validity:\s*([\d.]+)", content).group(1)),2),
            'Uniqueness': round(float(re.search(r"Uniqueness:\s*([\d.]+)", content).group(1)),2),
            'Novelty': round(float(re.search(r"Novelty:\s*([\d.]+)", content).group(1)),2),
            'KL': round(float(re.search(r"KL divergence:\s*([\d.]+)", content).group(1)),2),
            'FCD': round(float(re.search(r"Frechet ChemNet Distance:\s*([\d.]+)", content).group(1)),2),
            'IntDiversity': round(float(re.search(r"IntDiversity:\s*([\d.]+)", content).group(1)),2),
            'ExtDiversity': 1-round(float(re.search(r"ExtDiversity:\s*([\d.]+)", content).group(1)),2),
            'MOLWT': round(float(re.search(r"MOLWT:\s*([\d.]+)", content).group(1)),2),
            'LOGP': round(float(re.search(r"LOGP:\s*([\d.]+)", content).group(1)),2),
            'HBD': round(float(re.search(r"HBD:\s*([\d.]+)", content).group(1)),2),
            'HBA': round(float(re.search(r"HBA:\s*([\d.]+)", content).group(1)),2),
            'COMP': round(float(re.search(r"COMP:\s*([\d.]+)", content).group(1)),2),
            'LRO5': round(float(re.search(r"LRO5:\s*([\d.]+)", content).group(1)),2),
        }
    except AttributeError:
        print(f"Warning: Missing one or more values in {file_path}")
        return None

def scale_to_proportion(df, columns):
    df_scaled = df.copy()
    for col in columns:
        df_scaled[col] = round(df_scaled[col] / 100.0, 2)
    return df_scaled


def is_pareto_efficient(df, objective_columns):
    """
    Identifies Pareto-efficient points in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing objective values.
        objective_columns (list of str): Column names representing objectives to maximize.

    Returns:
        np.ndarray: A boolean array where True indicates a Pareto-efficient point.
    """

    data = df[objective_columns].values
    n = data.shape[0]
    is_efficient = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_efficient[i]:
            continue

        for j in range(n):
            if i == j:
                continue

            # point j dominates point i if:
            greater_equal = data[j] >= data[i]
            strictly_greater = data[j] > data[i]

            if np.all(greater_equal) and np.any(strictly_greater):
                is_efficient[i] = False
                break

    return is_efficient

def compute_pareto_hypervolume(df, metric_columns, epsilon=1e-6):
    """
    Computes normalized hypervolume for a maximization problem.

    Parameters:
        df (pd.DataFrame): DataFrame with Pareto-optimal points (values in [0,1])
        metric_columns (list): Columns used as objectives (maximized)
        epsilon (float): Safety margin to keep points inside bounds

    Returns:
        float: Normalized hypervolume in [0,1]
    """

    # Convert maximization to minimization by 1 - x
    pareto_points = 1.0 - df[metric_columns].values

    # Slightly pull points away from 1.0 to avoid equality with reference
    pareto_points = np.clip(pareto_points, 0.0, 1.0 - epsilon)

    reference_point = [1.0] * len(metric_columns)
    ideal_point = [0.0] * len(metric_columns)
    max_volume = np.prod(np.array(reference_point) - np.array(ideal_point))  # = 1.0 if normalized

    # Compute hypervolume
    hv = hypervolume(pareto_points)
    actual_volume = hv.compute(reference_point)

    # Compute and print exclusive contribution per point
    contributions = hv.contributions(reference_point)

    return actual_volume / max_volume if max_volume > 0 else 0.0, contributions

def plot_pareto(df, dimensions, output_file):

    # Color array: 1 for Pareto (red), 0 for others (black)
    color_array = np.where(df['pareto'], 1, 0)

    colorscale = [
        [0.0, 'black'],
        [1.0, 'red']
    ]

    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=color_array,
            colorscale=colorscale,
            showscale=False
        ),
        dimensions=[dict(label=col, values=df[col]) for col in dimensions],
    ))

    fig.update_layout(
        title="Parallel Coordinates: Pareto (Red) vs Non-Pareto (Black)"
    )

    if output_file:
        if output_file.endswith(".html"):
            fig.write_html(output_file)
        else:
            fig.write_image(output_file)
        print(f"Figure saved to: {output_file}")

    fig.show()


def compute_crowding_distance(df, metric_columns):
    """
    Computes crowding distance for each point based on the given objective columns.

    Distances are normalized per objective and summed across objectives.
    Boundary points receive infinite distance.

    Parameters:
        df (pd.DataFrame): DataFrame with objective values.
        objectives (list of str): Columns used for crowding calculation.

    Returns:
        np.ndarray: Crowding distance for each row in df.
    """

    num_points = df.shape[0]
    distances = np.zeros(num_points)

    for col in metric_columns:
        sorted_idx = np.argsort(df[col].values)
        distances[sorted_idx[0]] = distances[sorted_idx[-1]] = np.inf  # boundaries

        col_range = df[col].max() - df[col].min()
        if col_range == 0:
            continue  # skip if no variation

        for i in range(1, num_points - 1):
            prev_val = df[col].iloc[sorted_idx[i - 1]]
            next_val = df[col].iloc[sorted_idx[i + 1]]
            distances[sorted_idx[i]] += (next_val - prev_val) / col_range

    return distances


def compute_rank_aggregation(df, metric_columns):
    """
    Compute Borda count-based rank aggregation over specified metric columns.
    Assumes higher is better for all metrics.
    """
    ranks = np.zeros((len(df), len(metric_columns)))

    for i, col in enumerate(metric_columns):
        ranks[:, i] = df[col].rank(ascending=False, method='min')

    # Borda count: sum of ranks
    aggregate_rank = ranks.sum(axis=1)
    return aggregate_rank


def compute_topsis(df, metrics):
    """
    Compute TOPSIS scores for rows at `idx` using normalized, maximization metrics.

    Args:
        df (DataFrame): Full dataset.
        metrics (list): Columns to use for scoring (values in [0, 1], higher is better).
        idx (Index or list): Row indices (e.g. Pareto front) to compute scores for.

    Returns:
        np.ndarray: TOPSIS scores in [0, 1], higher is better.
    """
    values = df[metrics].values
    dist_to_ideal = np.linalg.norm(values - 1, axis=1)
    dist_to_anti = np.linalg.norm(values, axis=1)
    return dist_to_anti / (dist_to_ideal + dist_to_anti)



def main(data_folder, selected_property, output_file, output_folder, n_samples):

    # --- Data preparation ---
    
    # Extract metrics
    all_metrics = []
    for filename in os.listdir(data_folder):
        if filename.startswith("guidance") and "results" in filename:
            match = re.search(r"guidance_(\d+)_results", filename)
            lambda_value = float(match.group(1))

            file_path = os.path.join(data_folder, filename)
            metrics = extract_metrics(file_path, lambda_value)
            if metrics:
                all_metrics.append(metrics)

    df = pd.DataFrame(all_metrics)

    # Scale molecular percentages to [0, 1]
    molecular_props = ['MOLWT', 'LOGP', 'HBD', 'HBA', 'COMP', 'LRO5']
    df = scale_to_proportion(df, molecular_props)

    # Sanity check
    if selected_property not in df.columns:
        print(f"Error: Property '{selected_property}' not found in dataset.")
        return

    # Mark and exclude rows that are within 2σ of the baseline
    df['excluded'] = False
    # baseline_row = df[df['lambda'] == 0.0]

    # if not baseline_row.empty:
    #     baseline_value = baseline_row.iloc[0][selected_property]

    #     df['excluded'] = False
    #     for i, row in df.iterrows():
    #         count1 = int(row[selected_property] * n_samples)
    #         count2 = int(baseline_value * n_samples)
    #         stat, pval = proportions_ztest([count1, count2], [n_samples, n_samples], alternative='larger')
    #         df.at[i, 'excluded'] = pval > 0.05  # not significantly better than baseline

    # else:
    #     print("Baseline (λ = 0) not found. Skipping filtering.")
    
    # Select metrics
    metrics = ['lambda', selected_property, 'Validity', 'Uniqueness', 'Novelty', 'KL', 'FCD', 'IntDiversity', 'ExtDiversity']
    metrics_no_lambda = metrics[1:]
    constant_metrics = [col for col in df.columns if df[col].nunique() <= 1]
    metrics_for_plot = [m for m in metrics if m not in constant_metrics]
    

    # --- Calculate Pareto frontier ---
    # Calculate points on the frontier
    df_for_pareto = df[~df['excluded']].copy()
    pareto_mask_subset = is_pareto_efficient(df_for_pareto, metrics_no_lambda)
    df['pareto'] = False
    df.loc[df_for_pareto.index[pareto_mask_subset], 'pareto'] = True

    # Plot the pareto frontier
    plot_pareto(df, metrics_for_plot, os.path.join(output_folder, output_file))

    pareto_df = df[df['pareto']].copy() # dataframe of points on the frontier
    print("Pareto-optimal lambda values:")
    print(pareto_df[['lambda'] + metrics_no_lambda])

    # Compute hypervolume and point-exclusive hypervolume
    if not pareto_df.empty:
        hv_score, contributions = compute_pareto_hypervolume(pareto_df, metrics_no_lambda)
        print(f"\nHypervolume of Pareto front (ref = 0.0): {hv_score:.4f} / 1.0")

        pareto_df['contribution'] = contributions
        for _, row in pareto_df.sort_values(by='contribution', ascending=False).iterrows():
            print(f"λ = {row['lambda']:.2f}, contribution = {row['contribution']:.6f}")
    else:
        print("No Pareto-optimal points found for hypervolume calculation.")

    # Plot metrics used in Pareto frontier vs lambda 
    # (univariate plots including points not on the frontier)
    for col in metrics_no_lambda:
        if col in df.columns:
            df.set_index('lambda').sort_index()[col].plot(marker='o')
            plt.title(f"{col} vs λ")
            plt.xlabel("λ (lambda)")
            plt.ylabel(col)
            plt.grid(True)
            plt.tight_layout()
            plt.show()


    # --- Compute crowding distance ---
    # (only for points on the frontier)
    crowding_distances = compute_crowding_distance(pareto_df, metrics_no_lambda)
    pareto_df['crowding_distance'] = crowding_distances

    # Sort by crowding distance (descending)
    print(pareto_df[['lambda', 'crowding_distance']].sort_values('crowding_distance', ascending=False).round(2).to_string(index=True))


    # ---- Rank Aggregation using Borda count (sum) ----
    if not pareto_df.empty:
        aggregated_ranks = compute_rank_aggregation(pareto_df, metrics_no_lambda)
        pareto_df['borda_rank'] = aggregated_ranks

        print("\nFull Borda Rank Scores (lower is better):")
        print(pareto_df.sort_values('borda_rank')[['lambda'] + ['borda_rank']])


    # --- TOPSIS scoring ---
    if not pareto_df.empty:
        topsis_scores = compute_topsis(pareto_df, metrics_no_lambda)
        pareto_df['topsis_score'] = topsis_scores

        print("\nFull TOPSIS Scores (higher is better):")
        print(pareto_df.sort_values('topsis_score', ascending=False)[['lambda'] + ['topsis_score']])

    else:
        print("No Pareto-optimal points to rank via TOPSIS.")

    
    # --- Save results ---
    df_reduced = df.drop(columns=['pareto', 'excluded'])
    df_reduced.sort_values("lambda").to_csv(
        os.path.join(output_folder, "full_results.csv"), index=False
        )
    
    pareto_df.sort_values("lambda").to_csv(
        os.path.join(output_folder, "pareto_frontier.csv"), index=False
        )

    print("Saved full results to 'full_results.csv' and Pareto frontier to 'pareto_frontier.csv'")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pareto frontier analysis using a molecular property.")
    parser.add_argument("--data_folder", help="Path to the folder containing result files.")
    parser.add_argument("--property", choices=["MOLWT", "LOGP", "HBD", "HBA", "COMP", "LRO5"],
                        help="Molecular property to include in the plot (default: MOLWT)")
    parser.add_argument("--n_samples", type=int, help="Number of samples for each guidance level lambda.")
    parser.add_argument("--pareto_fig", help="File name of the plot (HTML or image format)")
    parser.add_argument("--output_folder", help="Output directory relative path for csv files")


    args = parser.parse_args()

    main(args.data_folder, args.property, args.pareto_fig, args.output_folder, args.n_samples)
