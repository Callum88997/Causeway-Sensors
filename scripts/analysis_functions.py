# Imports required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display
import re
from scipy.optimize import curve_fit

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import linregress

from scripts.setup_functions import collapsible_output

"""def detect_anomalies(df, feature_cols, contamination=0.01):
    '''Applies an Isolation Forest to detect multivariate outliers robustly.
    Evaluates each inferred stage independently to prevent score dilution.
    If a chip is anomalous in ANY single stage, it is flagged as an outlier.

    Args:
        df (pd.DataFrame): The dataframe to analyse.
        feature_cols (list[str]): The list of feature columns to evaluate.
        contamination (str | float, optional): The proportion of outliers. Defaults to 'auto'.

    Returns:
        pd.DataFrame: The dataframe with 'anomaly' (-1 for outlier, 1 for inlier) and 'anomaly_score'.
    '''

    df_clean = df.copy()

    # Initialise trackers for all chips
    worst_scores = pd.Series(index=df_clean.index, data=np.inf)
    is_anomaly = pd.Series(index=df_clean.index, data=1)

    # Group columns dynamically
    stage_groups = {}

    for col in feature_cols:

        stage = str(col).rsplit('_', 1)[0]

        if stage not in stage_groups:

            stage_groups[stage] = []

        stage_groups[stage].append(col)

    # Loop through the pre-grouped dictionary
    for stage, stage_cols in stage_groups.items():
            
        df_stage = df_clean[stage_cols].dropna()
        
        if df_stage.empty or len(df_stage) < 2:
            continue

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(df_stage)
        
        iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=8030, n_jobs=-1)
        preds = iso.fit_predict(X_scaled)
        scores = iso.decision_function(X_scaled)

        # Keep the lowest (most anomalous) score for each chip
        current_scores = pd.Series(scores, index=df_stage.index)
        current_preds = pd.Series(preds, index=df_stage.index)
        
        # Update the master trackers ONLY for those specific chips using
        worst_scores.loc[df_stage.index] = np.minimum(worst_scores.loc[df_stage.index], current_scores)
        is_anomaly.loc[df_stage.index] = np.minimum(is_anomaly.loc[df_stage.index], current_preds)
        
    # Clean up the infinite initialization fallback
    worst_scores.replace(np.inf, 0.0, inplace=True)
    
    df_clean['anomaly'] = is_anomaly
    df_clean['anomaly_score'] = worst_scores

    return df_clean
"""



def detect_anomalies(df, feature_cols, contamination=0.1):
    '''Applies an Isolation Forest to detect multivariate outliers robustly.
    Evaluates each inferred stage independently.
    A chip fails if: 
      1. It is anomalous in >= 2 consecutive stages.
      2. It fails the VERY LAST stage.
      3. It has missing data (NaNs).
    '''
    df_clean = df.copy()

    # Initialise trackers
    worst_scores = pd.Series(index=df_clean.index, data=np.inf)
    
    # Master flag for NaN dropouts and final stage failures
    hard_fail = pd.Series(index=df_clean.index, data=1)
    failed_final_stage = pd.Series(index=df_clean.index, data=0)
    
    # Trackers for consecutive failures
    current_streak = pd.Series(index=df_clean.index, data=0)
    max_streak = pd.Series(index=df_clean.index, data=0)

    # Group columns dynamically (preserves chronological order of feature_cols)
    stage_groups = {}
    for col in feature_cols:
        stage = str(col).rsplit('_', 1)[0]
        if stage not in stage_groups:
            stage_groups[stage] = []
        stage_groups[stage].append(col)

    # Identify the name of the final stage
    final_stage_name = list(stage_groups.keys())[-1]

    # Loop through the pre-grouped dictionary sequentially
    for stage, stage_cols in stage_groups.items():
            
        # 1. Catch missing values (NaNs) and issue a hard fail
        stage_nans = df_clean[stage_cols].isna().any(axis=1)
        nan_chips = stage_nans[stage_nans].index
        
        if len(nan_chips) > 0:
            hard_fail.loc[nan_chips] = -1
            worst_scores.loc[nan_chips] = -999.0
            
        df_stage = df_clean[stage_cols].dropna()
        
        if df_stage.empty or len(df_stage) < 2:
            continue

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(df_stage)
        
        iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=8030, n_jobs=-1)
        preds = iso.fit_predict(X_scaled)
        scores = iso.decision_function(X_scaled)

        current_scores = pd.Series(scores, index=df_stage.index)
        current_preds = pd.Series(preds, index=df_stage.index)
        
        # Keep the lowest (most anomalous) score for each chip
        worst_scores.loc[df_stage.index] = np.minimum(worst_scores.loc[df_stage.index], current_scores)
        
        # 2. Track consecutive failures ("multiple in a row")
        failed_mask = (current_preds == -1)
        
        # Increment streak if it failed this stage
        current_streak.loc[df_stage.index[failed_mask]] += 1
        
        # Reset streak to 0 if it passed this stage
        current_streak.loc[df_stage.index[~failed_mask]] = 0
        
        # Update the maximum streak recorded for each chip
        max_streak = np.maximum(max_streak, current_streak)
        
        # 3. Check if this is the final stage and flag failures
        if stage == final_stage_name:
            failed_final_stage.loc[df_stage.index[failed_mask]] = 1
            
    # Clean up the infinite initialization fallback
    worst_scores.replace(np.inf, 0.0, inplace=True)
    
    # 4. Final Verdict: Fail if NaN dropout, max streak >= 2, OR failed the very last stage
    df_clean['anomaly'] = np.where(
        (hard_fail == -1) | 
        (max_streak >= 3) | 
        (failed_final_stage == 1), 
        -1, 1
    )
    df_clean['anomaly_score'] = worst_scores
        
    return df_clean



"""def detect_anomalies(df, feature_cols, contamination='auto'): # 0.07
    '''Applies an Isolation Forest to detect multivariate outliers.

    Args:
        df (pd.DataFrame): The dataframe to analyse.
        feature_cols (list[str]): The list of feature columns to evaluate.
        contamination (str | float, optional): The proportion of outliers in the data. Defaults to 'auto'.

    Returns:
        pd.DataFrame: The dataframe with 'anomaly' (-1 for outlier, 1 for inlier) and 'anomaly_score' columns.
    '''

    # Creates a copy of the clean dataframe to preserve the original data
    df_clean = df.dropna(subset=feature_cols).copy()
    
    # Checks whether the dataframe contains data
    if df_clean.empty or len(df_clean) < 2:

        df_clean['anomaly'] = 1
        df_clean['anomaly_score'] = 0.0

        return df_clean
    
    # Initialises the scaler and standardises the features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df_clean[feature_cols])
    
    # Initialises and fits the Isolation Forest model
    iso = IsolationForest(n_estimators=500, contamination=contamination, random_state=8030)

    df_clean['anomaly'] = iso.fit_predict(X_scaled)
    df_clean['anomaly_score'] = iso.decision_function(X_scaled)
    
    return df_clean
"""

def pivot_chip_data(df, stage_col, val_col):
    '''Pivots data for anomaly detection.

    Args:
        df (pd.DataFrame): The dataframe to pivot.
        stage_col (str): The column containing stage labels.
        val_col (str): The column containing values to aggregate.

    Returns:
        pd.DataFrame: The pivoted dataframe.
    '''

    return df.pivot_table(index='chip_id', columns=stage_col, values=val_col, aggfunc='mean', observed=False)
   
def summarise_correlations(corr_matrix, label, num=3):
    '''Summarises the strongest positive, strongest negative, and weakest correlations.

    Args:
        corr_matrix (pd.DataFrame): The correlation matrix to summarise.
        label (str): The prefix label for the printed summary.
        num (int, optional): The number of top correlations to display. Defaults to 3.
    '''

    # Prints the correlation insights header
    print(f'\n {label} Correlation Insights')

    # Extracts the upper triangle of the correlation matrix to avoid duplicates
    corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).unstack().dropna()

    # Checks whether the correlation series contains data
    if corr.empty:
        print('Not enough data to compute correlations.')
        
        # Returns control to the calling code
        return

    # Sorts the correlations into positive, negative, and weakest categories
    positive = corr[corr > 0].sort_values(ascending=False)
    negative = corr[corr < 0].sort_values()
    weakest = corr.iloc[corr.abs().argsort()]

    print('Top Positive Correlations:')

    # Checks if positive correlations exist
    if len(positive):

        # Loops through each key-value pair in the positive correlations
        for (s1, s2), val in positive.head(num).items():
            print(f'  - {s1} & {s2} (r = {val:.3f})')
    else:
        print('  None')

    print('\nTop Negative (Inverse) Correlations:')

    # Checks if negative correlations exist
    if len(negative):

        # Loops through each key-value pair in the negative correlations
        for (s1, s2), val in negative.head(num).items():
            print(f'  - {s1} & {s2} (r = {val:.3f})')
    else:
        print('  None')

    print('\nWeakest Relationships (Closest to zero):')

    # Loops through each key-value pair in the weakest correlations
    for (s1, s2), val in weakest.head(num).items():
        print(f'  - {s1} & {s2} (r = {val:.3f})')

def analyse_anomaly_drivers(df, anomalies_df, label, num=3):
    '''Compares anomalous chips to normal chips to identify which stages drive the anomaly.

    Args:
        df (pd.DataFrame): The original dataframe containing the stage data.
        anomalies_df (pd.DataFrame): The dataframe containing anomaly scores and labels.
        label (str): The prefix label for the summary output.
        num (int, optional): The number of top driving stages to display. Defaults to 3.
    '''

    # Extracts the indices for outliers and normal chips
    outliers = anomalies_df[anomalies_df['anomaly'] == -1].index
    normals = anomalies_df[anomalies_df['anomaly'] == 1].index

    # Prints the anomaly breakdown header
    print(f'\n {label} Anomaly Stage Breakdown')

    # Checks if there are no outliers detected
    if len(outliers) == 0:

        print('No anomalies detected. All chips are behaving within normal bounds.')
        
        return

    # Checks if there is a sufficient baseline of normal chips
    if len(normals) < 2:

        print('Not enough normal chips to form a baseline for comparison.')
        
        return

    # Slices the data into normal and anomaly subsets
    normal_data = df.loc[normals]
    anomaly_data = df.loc[outliers]

    # Calculates the statistical means and standard deviations
    normal_mean = normal_data.mean()
    normal_std = normal_data.std().replace(0, 1e-9) 

    anomaly_mean = anomaly_data.mean()
    
    # Calculates the z-scores to identify the largest deviations
    z_scores = ((anomaly_mean - normal_mean) / normal_std).abs().sort_values(ascending=False).dropna()

    print(f'Top {num} stages driving these anomalies (Largest deviation from normal):')

    # Loops through the top z-scores to print the anomaly drivers
    for stage, z in z_scores.head(num).items():

        norm_val = normal_mean[stage]
        anom_val = anomaly_mean[stage]
        print(f'  - {stage}: Anomaly Avg = {anom_val:.3f} | Normal Avg = {norm_val:.3f} (Z-Score: {z:.2f})')

def create_wide_intra(intra_df, val_col, keep_metrics=['std', 'slope', 'range', 'skew']):
    '''Pivots multiple intra-stage metrics and suffixes columns to prevent overlap.
    Dynamically identifies all metric columns belonging to the target channel.

    Args:
        intra_df (pd.DataFrame): The dataframe containing intra-stage features.
        val_col (str): The base value column used to derive metric names.
        keep_metrics (list[str], optional): Suffixes of metrics to keep.

    Returns:
        pd.DataFrame: The pivoted wide-format dataframe.
    '''

    # Dynamically finds all columns that start with the target channel's name
    intra_metrics = [col for col in intra_df.columns if col.startswith(f'{val_col}_') and col.replace(f'{val_col}_', '') in keep_metrics and col not in ['chip_id', 'stage']]

    chronological_stages = intra_df['stage'].unique()

    wide_list = []
    
    # Loops through each required metric
    for metric in intra_metrics:

        wide_metric = pivot_chip_data(intra_df, 'stage', metric)
        
        # Extracts just the statistical suffix (e.g., 'std', 'slope', 'skew')
        metric_suffix = metric.replace(f'{val_col}_', '')
        
        # Appends the suffix to the wide column names to prevent overlap
        wide_metric.columns = [f'{col}_{metric_suffix}' for col in wide_metric.columns]
        wide_list.append(wide_metric)

    # Concatenates the wide list if it contains data
    if not wide_list:
        return pd.DataFrame()
        
    wide_df = pd.concat(wide_list, axis=1)

    ordered_columns = []
    for metric in intra_metrics:
        metric_suffix = metric.replace(f'{val_col}_', '')
        for stage in chronological_stages:
            col_name = f'{stage}_{metric_suffix}'
            if col_name in wide_df.columns:
                ordered_columns.append(col_name)
            
    # Concatenates and returns the wide list if it contains data
    return wide_df[ordered_columns]

def get_outlier_chips(anomalies_df):
    '''Extracts unique chip IDs from the anomaly dataframe, handling both standard and MultiIndex.

    Args:
        anomalies_df (pd.DataFrame): The dataframe containing anomaly labels.

    Returns:
        list: A list of unique outlier chip IDs.
    '''

    # Checks whether the dataframe contains data
    if anomalies_df.empty:
        return []
    
    # Filters the dataframe to isolate identified outliers
    outliers = anomalies_df[anomalies_df['anomaly'] == -1]
    
    # Handles extraction for MultiIndex dataframes
    if isinstance(outliers.index, pd.MultiIndex):
        return outliers.index.get_level_values('chip_id').unique().tolist()
    
    # Returns the standard index list
    return outliers.index.tolist()

def analyse_pel(events_df, changes_df, intra_df, val_col='quad_ch1', change_col='quad_ch1_change', intra_val_col=None, title='PEL Analysis'):
    '''Analyses PEL signals, stage deltas, and intra-stage kinetics for anomalies.

    Args:
        events_df (pd.DataFrame): Stage-level PEL signal measurements.
        changes_df (pd.DataFrame): PEL stage-to-stage signal changes.
        intra_df (pd.DataFrame): PEL intra-stage feature measurements.
        val_col (str, optional): Absolute-signal column to analyse. Defaults to 'quad_ch1'.
        change_col (str, optional): Stage-delta column to analyse. Defaults to 'quad_ch1_change'.
        intra_val_col (str | None, optional): Source signal for intra-stage features. Defaults to None.
        title (str, optional): Heading used for the generated output sections. Defaults to 'PEL Analysis'.

    Returns:
        tuple: Wide tables and their absolute, delta, and intra-stage outliers.
    '''

    # Assigns the appropriate intra-stage target column
    intra_target = intra_val_col if intra_val_col else val_col

    # Filters out non-informative startup stages
    events_df = events_df[~events_df['stage'].isin(['Start', 'Initial'])]

    # Pivots the tables to prepare for anomaly detection
    wide_events = pivot_chip_data(events_df, 'stage', val_col)
    wide_changes = pivot_chip_data(changes_df, 'stage', change_col)
    wide_intra = create_wide_intra(intra_df, intra_target)

    # Detects absolute signal anomalies
    anomalies_abs = detect_anomalies(wide_events, wide_events.columns.tolist())
    outliers_abs = get_outlier_chips(anomalies_abs)

    # Detects stage delta anomalies
    anomalies_delta = detect_anomalies(wide_changes, wide_changes.columns.tolist())
    outliers_delta = get_outlier_chips(anomalies_delta)

    # Detects intra-stage kinetic anomalies
    anomalies_intra = detect_anomalies(wide_intra, wide_intra.columns.tolist())
    outliers_intra = get_outlier_chips(anomalies_intra)

    # Displays the results in a collapsible output section
    with collapsible_output(f'{title} - Absolute Signal'):

        # Plots the absolute signal correlation matrix to show relationships between PEL transitions
        plt.figure(figsize=(10, 6))
        sns.heatmap(wide_events.corr(), cmap='plasma', center=0, annot=True)
        plt.title(f'PEL [{val_col}]: Absolute Signal Correlation')
        plt.tight_layout()
        plt.show()

        summarise_correlations(wide_events.corr(), f'PEL {val_col} Absolute Signal')
        print(f'\nPotential Anomalous Chips (Absolute): {len(outliers_abs)}')
        analyse_anomaly_drivers(wide_events, anomalies_abs, f'PEL {val_col} Absolute Signal')

    # Displays the results in a collapsible output section
    with collapsible_output(f'{title} - Stage Delta'):

        # Plots the stage delta correlation matrix to show relationships between PEL response changes
        plt.figure(figsize=(10, 6))
        sns.heatmap(wide_changes.corr(), cmap='plasma', center=0, annot=True)
        plt.title(f'PEL [{change_col}]: Stage Delta Correlation')
        plt.tight_layout()
        plt.show()

        summarise_correlations(wide_changes.corr(), f'PEL {change_col} Stage Delta')
        print(f'\nPotential Anomalous Chips (Delta): {len(outliers_delta)}')
        analyse_anomaly_drivers(wide_changes, anomalies_delta, f'PEL {change_col} Stage Delta')
    
    # Displays the results in a collapsible output section
    with collapsible_output(f'{title} - Intra-stage Kinetics'):

        # Plots the intra-stage kinetic correlation matrix to show relationships between PEL response features
        plt.figure(figsize=(20, 20))
        sns.heatmap(wide_intra.corr(), cmap='plasma', center=0, annot=True)
        plt.title(f'PEL [{intra_target}]: Intra-stage Kinetics Correlation')
        plt.tight_layout()
        plt.show()

        summarise_correlations(wide_intra.corr(), f'PEL {intra_target} Intra-stage Kinetics')
        print(f'\nPotential Anomalous Chips (Intra-stage Kinetics): {len(outliers_intra)}')
        analyse_anomaly_drivers(wide_intra, anomalies_intra, f'PEL {intra_target} Intra-stage Kinetics')

    return wide_events, wide_changes, wide_intra, outliers_abs, outliers_delta, outliers_intra

def analyse_immob_split(events_df, changes_df, intra_df, split_name, val_col='channel1', change_col='channel1_change', intra_val_col=None, title='Immobilisation Analysis'):
    '''Analyses one immobilisation split across signal, delta, and kinetic features.

    Args:
        events_df (pd.DataFrame): Stage-level immobilisation measurements.
        changes_df (pd.DataFrame): Stage-to-stage immobilisation changes.
        intra_df (pd.DataFrame): Intra-stage feature measurements.
        split_name (str): Descriptive label for the selected split.
        val_col (str, optional): Absolute-signal column to analyse. Defaults to 'channel1'.
        change_col (str, optional): Stage-delta column to analyse. Defaults to 'channel1_change'.
        intra_val_col (str | None, optional): Source signal for intra-stage features. Defaults to None.
        title (str, optional): Heading used for generated output sections. Defaults to 'Immobilisation Analysis'.

    Returns:
        tuple: Wide tables and their absolute, delta, and intra-stage outliers.
    '''

    # Assigns the appropriate intra-stage target column
    intra_target = intra_val_col if intra_val_col else val_col

    # Filters out non-informative startup stages
    events_df = events_df[~events_df['stage'].isin(['Start', 'Initial'])]
    
    # Pivots the tables to prepare for anomaly detection
    wide_events = pivot_chip_data(events_df, 'stage', val_col)
    wide_changes = pivot_chip_data(changes_df, 'stage', change_col)
    wide_intra = create_wide_intra(intra_df, intra_target)

    # Detects absolute signal anomalies
    anomalies_abs = detect_anomalies(wide_events, wide_events.columns.tolist())
    outliers_abs = get_outlier_chips(anomalies_abs)

    # Detects stage delta anomalies
    anomalies_delta = detect_anomalies(wide_changes, wide_changes.columns.tolist())
    outliers_delta = get_outlier_chips(anomalies_delta)

    # Detects intra-stage kinetic anomalies
    anomalies_intra = detect_anomalies(wide_intra, wide_intra.columns.tolist())
    outliers_intra = get_outlier_chips(anomalies_intra)

    # Displays the results in a collapsible output section
    with collapsible_output(f'{title} - Absolute Signal'):

        # Plots the absolute signal correlation matrix to compare immobilisation transitions
        plt.figure(figsize=(20, 20))
        sns.heatmap(wide_events.corr(), cmap='plasma', annot=True, annot_kws={'size': 12})
        plt.title(f'Immob [{split_name}] - {val_col}: Absolute Signal Correlation')
        plt.tight_layout()
        plt.show()

        summarise_correlations(wide_events.corr(), f'Immob [{split_name}] {val_col} Absolute Signal')
        print(f'\nPotential Anomalous Chips (Absolute): {len(outliers_abs)}')
        analyse_anomaly_drivers(wide_events, anomalies_abs, f'Immob [{split_name}] {val_col} Absolute Signal')

    # Displays the results in a collapsible output section
    with collapsible_output(f'{title} - Stage Delta'):

        # Plots the stage delta correlation matrix to compare immobilisation response changes
        plt.figure(figsize=(20, 20))
        sns.heatmap(wide_changes.corr(), cmap='plasma', center=0, annot=True)
        plt.title(f'Immob [{split_name}] - {change_col}: Stage Delta Correlation')
        plt.show()

        summarise_correlations(wide_changes.corr(), f'Immob [{split_name}] {change_col} Stage Delta')
        print(f'\nPotential Anomalous Chips (Delta): {len(outliers_delta)}')
        analyse_anomaly_drivers(wide_changes, anomalies_delta, f'Immob [{split_name}] {change_col} Stage Delta')
    
    # Displays the results in a collapsible output section
    with collapsible_output(f'{title} - Intra-stage Kinetics'):

        # Plots the intra-stage kinetic correlation matrix to compare immobilisation response features
        plt.figure(figsize=(20, 20))
        sns.heatmap(wide_intra.corr(), cmap='plasma', center=0, annot=True)
        plt.title(f'Immob [{split_name}] - {intra_target}: Intra-stage Kinetics Correlation')
        plt.show()

        summarise_correlations(wide_intra.corr(), f'Immob [{split_name}] {intra_target} Intra-stage Kinetics')
        print(f'\nPotential Anomalous Chips (Intra-stage Kinetics): {len(outliers_intra)}')
        analyse_anomaly_drivers(wide_intra, anomalies_intra, f'Immob [{split_name}] {intra_target} Intra-stage Kinetics')
    
    return wide_events, wide_changes, wide_intra, outliers_abs, outliers_delta, outliers_intra

def plot_pel_layer_shifts(change_wide_df, channel_name='quad_ch1'):
    '''Plots PEL signal shifts between layers for each requested channel.

    Args:
        change_wide_df (pd.DataFrame): Wide-format PEL stage-transition measurements (rows=chip_id, cols=stages).
        channel_name (str, optional): Name of the channel for the plot title. Defaults to 'quad_ch1'.
    '''
    
    # Creates a copy of the dataframe to preserve the original data, filtering out the total shift
    layer_cols = [col for col in change_wide_df.columns if col != 'Total_Shift_Initial_to_Final']
    df_plot = change_wide_df[layer_cols].copy()

    # Displays the results in a collapsible output section
    with collapsible_output(f'PEL Layer Shifts: {channel_name}'):
        
        # Plots an interactive view of signal shifts across PEL layer transitions for each chip
        fig = go.Figure()
        
        # Loops through each grouped chip ID and its associated dataframe
        for chip_id, row in df_plot.iterrows():

            fig.add_trace(go.Scatter(
                x=layer_cols, 
                y=row[layer_cols], 
                mode='lines+markers', 
                name=str(chip_id), 
                opacity=0.6, 
                marker=dict(size=6), 
                visible='legendonly'
            ))

        fig.update_layout(
            title=f'PEL Layer Build-up: Shift per Transition ({channel_name})', 
            xaxis_title='Layer Transition', 
            yaxis_title='Signal Shift (Δ)', 
            template='plotly_white', 
            hovermode='x unified', 
            height=600
        )

        fig.update_xaxes(tickangle=45)
        fig.show()

        # Plots a static view of signal shifts across PEL layer transitions for each chip
        plt.figure(figsize=(10, 6))
        
        # Loops through each grouped chip ID and its associated dataframe
        for chip_id, row in df_plot.iterrows():
            plt.plot(layer_cols, row[layer_cols], marker='o', alpha=0.5, label=chip_id)

        plt.title(f'PEL Layer Build-up: Shift per Transition ({channel_name})')
        plt.xlabel('Layer Transition')
        plt.ylabel('Signal Shift (Δ)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # Adds a legend if the number of unique chips is manageable
        if len(df_plot.index) <= 25:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()

def combine_anomalies(*outliers_lists, title='Combined Anomalies Summary'):
    '''Combines an arbitrary number of outlier lists into a single set.

    Args:
        *outliers_lists (list): Variable length argument list of outlier chip IDs.
        title (str, optional): The title for the summary output. Defaults to 'Combined Anomalies Summary'.

    Returns:
        list: A deduplicated list of all outlier chips.
    '''

    all_outlier_chips = set()

    # Loops through each provided outliers list
    for outliers in outliers_lists:
        all_outlier_chips.update(outliers)
        
    all_outlier_chips = list(all_outlier_chips)

    # Displays the results in a collapsible output section
    with collapsible_output(title):

        print(f'Total Unique Anomalous Chips: {len(all_outlier_chips)}')

        # Checks whether any anomalous chips were found
        if all_outlier_chips:
            print(f'Chip IDs: {all_outlier_chips}')
        else:
            print('No anomalies detected in this combination.')

    return all_outlier_chips

def analyse_cross_stage_split(pel_df, immob_df, split_name, channel_name, pel_outliers=None, immob_outliers=None, sc_metrics_df=None, title='Cross-Stage Validation'):
    '''Merges PEL and a specific immobilisation split, runs PCA and clustering, plots the results with anomaly highlights for Channel 1, Channel 2, and overall. Validates against standard curves if provided.

    Args:
        pel_df (pd.DataFrame): The wide-format PEL dataframe.
        immob_df (pd.DataFrame): The wide-format immobilisation dataframe.
        split_name (str): The label for the current data split.
        channel_name (str): The name of the current channel.
        pel_outliers (list, optional): List of outlier chips from PEL analysis. Defaults to None.
        immob_outliers (list, optional): List of outlier chips from immobilisation analysis. Defaults to None.
        sc_metrics_df (pd.DataFrame, optional): Standard curve metrics for validation. Defaults to None.
        title (str, optional): The title for the summary output. Defaults to 'Cross-Stage Validation'.
    '''
    
    # Creates copies to preserve original data
    pel_copy = pel_df.copy()
    immob_copy = immob_df.copy()
    
    # Ensures column names are treated as strings
    pel_copy.columns = pel_copy.columns.astype(str)
    immob_copy.columns = immob_copy.columns.astype(str)

    # Merges the PEL and immobilisation datasets
    merged = pd.merge(pel_copy, immob_copy, left_index=True, right_index=True, how='inner', suffixes=('_PEL', '_Immob')).dropna()
    
    # Checks whether the merged dataframe contains data
    if merged.empty:
        print(f'[{split_name} - {channel_name}] Not enough overlapping chips between PEL and Immobilisation for cross-analysis.')
        
        # Returns control to the calling code
        return
             
    # Displays the results in a collapsible output section
    with collapsible_output(f'{title} - {channel_name}'):

        print(f'[{split_name} - {channel_name}] Overlapping chips analysed: {len(merged)}')
        
        # Standardises the features
        features = merged.columns.tolist()
        X_subset = RobustScaler().fit_transform(merged[features])
        
        # Initialises and applies PCA for dimensionality reduction
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_subset)
        
        # Gets each Principal Component values
        merged['PCA1'] = pca_result[:, 0]
        merged['PCA2'] = pca_result[:, 1]
        
        print(f'[{split_name} - {channel_name}] Explained Variance by first 2 components: {pca.explained_variance_ratio_.sum()*100:.2f}%')
        
        # Initialises and fits KMeans clustering
        kmeans = KMeans(n_clusters=2, random_state=8030, n_init=10)
        merged['Cluster'] = kmeans.fit_predict(X_subset)
        
        # Plots the first two principal components to show cluster separation and flagged anomalies
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=merged, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', s=100, alpha=0.7)

        # Combines outliers from both stages into a single set
        both_anomalies = set(pel_outliers if pel_outliers else []).union(set(immob_outliers if immob_outliers else []))
            
        # Identifies flagged anomalies within the current subset
        anomaly_mask = merged.index.isin(both_anomalies)
        
        # Highlights the identified anomalies on the scatter plot
        if anomaly_mask.any():
            plt.scatter(merged.loc[anomaly_mask, 'PCA1'], merged.loc[anomaly_mask, 'PCA2'], edgecolor='black', facecolor='none', s=200, label='Flagged Anomaly', linewidth=2)
                        
        plt.title(f'Cross-Stage PCA: {split_name} ({channel_name})')
        plt.xlabel('Principal Component 1 (General Signal Variance)')
        plt.ylabel('Principal Component 2 (Stage-to-Stage Dynamic Variance)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print(f'\n Cluster Profiling [{split_name} - {channel_name}]')
        
        # Calculates cluster means and overall statistics to build z-score profiles
        cluster_means = merged[features].groupby(merged['Cluster']).mean()
        overall_mean = merged[features].mean()
        overall_std = merged[features].std().replace(0, 1e-9)
        
        cluster_zscores = (cluster_means - overall_mean) / overall_std
        
        # Loops through each unique cluster ID
        for cluster_id in sorted(merged['Cluster'].unique()):

            print(f"\nCluster {cluster_id} Profile (n={sum(merged['Cluster'] == cluster_id)} observations):")
            
            # Sorts and extracts the highest and lowest z-score features
            top_pos = cluster_zscores.loc[cluster_id].sort_values(ascending=False).head(3)
            top_neg = cluster_zscores.loc[cluster_id].sort_values(ascending=True).head(3)
            
            print('  Defining High Features (Above Average):')
            pos_found = False

            # Loops through each feature and z-score in the top positive items
            for feat, z in top_pos.items():

                if z > 0.5:

                    print(f'    - {feat}: +{z:.2f} standard deviations')
                    pos_found = True
                    
            # Checks if no significant positive features were found
            if not pos_found: 
                print('    - None significantly above average')
                
            print('  Defining Low Features (Below Average):')
            neg_found = False

            # Loops through each feature and z-score in the top negative items
            for feat, z in top_neg.items():

                if z < -0.5:

                    print(f'    - {feat}: {z:.2f} standard deviations')
                    neg_found = True

            # Checks if no significant negative features were found
            if not neg_found: 
                print('    - None significantly below average')
        
        # Proceeds with standard curve mapping if metrics are provided
        if sc_metrics_df is not None:

            print(f'\n--- Cluster Functional Yield (R² Comparison) [{split_name} - {channel_name}] ---')
            
            cluster_mapping = merged[['Cluster']].reset_index()

            # Renames the index column to 'chip_id' if present
            if 'index' in cluster_mapping.columns:
                cluster_mapping = cluster_mapping.rename(columns={'index': 'chip_id'})
                
            # Merges the standard curve metrics with the cluster mapping
            sc_cluster_df = pd.merge(sc_metrics_df.reset_index(), cluster_mapping, on='chip_id', how='inner')
            
            # Checks whether the merged dataframe contains data
            if sc_cluster_df.empty:
                print('No Standard Curve data available to map to clusters.')
            else:

                # Aggregates and displays standard curve R-squared performance per cluster
                cluster_stats = sc_cluster_df.groupby('Cluster')['r2'].agg(['count', 'mean', 'median', 'std']).fillna(0)
                print('Standard Curve R² Performance by Cluster:')
                display(cluster_stats)
                
                # Plots standard-curve R-squared scores by cluster to compare functional yield
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=sc_cluster_df, x='Cluster', y='r2', showmeans=True,  meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'black'})
                sns.stripplot(data=sc_cluster_df, x='Cluster', y='r2', color='black', alpha=0.5, jitter=True)
                plt.axhline(0.95, color='red', linestyle='--', label='Pass Threshold (0.95)')
                
                plt.title(f'Standard Curve R² Distribution by Manufacturing Cluster\n{split_name} ({channel_name})')
                plt.xlabel('Manufacturing Cluster')
                plt.ylabel('Standard Curve R² Score')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
                
            # Creates lists to store the selected identifiers and anomaly results
            anomaly_ids = list(both_anomalies.intersection(set(merged.index)))
            
            # Checks if there are any anomalous chips to validate
            if not anomaly_ids:
                print(f'[{split_name} - {channel_name}] No anomalous chips from the cross-stage analysis to validate against SC.')
            else:

                print(f'\n[{split_name} - {channel_name}] Validating functional performance for anomalous chip(s): {anomaly_ids}\n')
                
                # Extracts standard curve data for the anomalous chips
                anomaly_sc_data = sc_metrics_df[sc_metrics_df.index.isin(anomaly_ids)]
                
                # Checks whether the anomaly data contains records
                if anomaly_sc_data.empty:
                    print('No Standard Curve data found for the flagged anomalous chip(s).')
                else:

                    print(f"{'Chip ID':<15} | {'Curve UUID':<40} | {'R² Score':<10} | {'Status'}")
                    print('-' * 80)

                    # Loops through each row in the anomaly data
                    for chip_id, row in anomaly_sc_data.iterrows():

                        r2 = row['r2']
                        curve_id = row['standard_curve_uuid']
                        status = 'PASSED' if r2 >= 0.95 else 'FAILED'
                        print(f'{chip_id:<15} | {curve_id:<40} | {r2:<10.4f} | {status}')
            
            # Isolates normal chips for baseline comparison
            normal_sc_data = sc_metrics_df[~sc_metrics_df.index.isin(anomaly_ids)]

            # Checks whether the normal data contains records
            if not normal_sc_data.empty:
                print(f"\n>>> Average R² for NORMAL chips: {normal_sc_data['r2'].mean():.4f}")
        else:

            # Creates lists to store the selected identifiers and results
            anomaly_ids = list(both_anomalies.intersection(set(merged.index)))

            print(f'\n--- Anomaly Identifications [{split_name} - {channel_name}] ---')

            # Prints the flagged anomalous chips or a pass message
            if anomaly_ids:
                print(f'Flagged Anomalous Chips in cluster: {anomaly_ids}')
            else:
                print('No anomalous chips detected in this cross-stage split.')

def print_correlation_insights(corr_matrix, split_name, channel_name, top_n=3):
    '''Extracts and prints top positive, negative, and weakest correlations.

    Args:
        corr_matrix (pd.DataFrame): The correlation matrix to analyse.
        split_name (str): The label for the current data split.
        channel_name (str): The name of the channel.
        top_n (int, optional): The number of top correlations to return. Defaults to 3.
    '''

    print(f'{split_name} ({channel_name}) Signal Correlation Insights')
    
    # Extracts the upper triangle of the matrix to avoid duplicated pairs
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_pairs = upper_tri.stack().dropna()
    
    # Checks whether the series contains data
    if corr_pairs.empty:
        print('Not enough variance to calculate correlation insights.\n')
        
        return

    print('Top Positive Correlations:')
    
    # Isolates and sorts the top positive correlations
    top_pos = corr_pairs[corr_pairs > 0].sort_values(ascending=False).head(top_n)

    # Checks whether the top positive series contains data
    if top_pos.empty: 
        print('  None')
    else:

        # Loops through each feature pair and value
        for (feat1, feat2), val in top_pos.items(): 
            print(f'  - {feat1} & {feat2} (r = {val:.3f})')

    print('\nTop Negative (Inverse) Correlations:')
    
    # Isolates and sorts the top negative correlations
    top_neg = corr_pairs[corr_pairs < 0].sort_values(ascending=True).head(top_n)

    # Checks whether the top negative series contains data
    if top_neg.empty: 
        print('  None')
    else:

        # Loops through each feature pair and value
        for (feat1, feat2), val in top_neg.items(): 
            print(f'  - {feat1} & {feat2} (r = {val:.3f})')

    print('\nWeakest Relationships (Closest to zero):')
    
    # Extracts the relationships closest to zero
    weakest = corr_pairs.reindex(corr_pairs.abs().sort_values().index).head(top_n)

    # Loops through each feature pair and value
    for (feat1, feat2), val in weakest.items():
        print(f'  - {feat1} & {feat2} (r = {val:.3f})')

    print('\n')

def cross_stage_correlations(pel_df, immob_df, split_name, channel_name, sc_metrics_df=None, title='Cross-Stage Correlations'):
    '''Compares PEL and immobilisation features using cross-stage correlations.

    Args:
        pel_df (pd.DataFrame): Wide PEL feature table.
        immob_df (pd.DataFrame): Wide immobilisation feature table.
        split_name (str): Label identifying the immobilisation split.
        channel_name (str): The name of the current channel. 
        sc_metrics_df (pd.DataFrame | None, optional): Optional standard-curve metrics. Defaults to None.
        title (str, optional): Heading used for generated output. Defaults to 'Cross-Stage Correlations'.
    '''

    # Creates copies to preserve original data
    pel_copy = pel_df.copy()
    immob_copy = immob_df.copy()

    # Casts the columns to strings to ensure consistent merging
    pel_copy.columns = pel_copy.columns.astype(str)
    immob_copy.columns = immob_copy.columns.astype(str)
    
    # Merges the PEL and immobilisation datasets
    merged = pel_copy.merge(immob_copy, left_index=True, right_index=True, how='inner', suffixes=('_PEL', '_Immob'))

    # Integrates standard curve metrics if they are provided
    if sc_metrics_df is not None:

        # Creates a copy of the standard curve dataframe to preserve the original data
        sc_copy = sc_metrics_df.copy()
        sc_copy.columns = sc_copy.columns.astype(str)
        numeric_cols = sc_copy.select_dtypes(include='number').columns.tolist()
        
        # Ensures datetime is preserved alongside numeric metrics
        if 'datetime' in sc_copy.columns: 
            sc_copy = sc_copy[['datetime'] + numeric_cols]
        else: 
            sc_copy = sc_copy[numeric_cols]
            
        # Selects the earliest standard curve entry per chip to avoid duplicates
        sc_first = sc_copy.sort_values('datetime').groupby(sc_copy.index, sort=False).first().drop(columns='datetime')
        
        merged = merged.merge(sc_first, left_index=True, right_index=True, how="left")

    # Displays the results in a collapsible output section
    with collapsible_output(f'{title} - {channel_name}'):

            
        # Checks whether the merged dataframe contains data
        if merged.empty:

            print(f"[{split_name} - {channel_name}] Not enough overlapping chips for cross-analysis.")

            return

        print(f'[{split_name} - {channel_name}] Overlapping data points analysed: {len(merged)}\n')

        corr_matrix = merged.corr()
        print_correlation_insights(corr_matrix, split_name, channel_name, top_n=3)

        # Plots the cross-stage correlation matrix to show relationships between all selected features
        plt.figure(figsize=(20, 20))
        sns.heatmap(corr_matrix, cmap='plasma', center=0, annot=False)
        plt.title(f'{split_name} ({channel_name}) - Overall Correlation')
        plt.tight_layout()
        plt.show()

def create_interactive_dashboard(data_catalog):
    '''Creates an interactive explorer for comparing data sources and features.

    Args:
        data_catalog (dict): Nested mapping of analysis modes to named dataframes.
    '''

    # Initialises the state dictionary to track the last known values and prevent overwriting active filters
    state = {
        'updating': False,
        'last_mode': None,
        'last_ch': None,
        'last_x_src': None,
        'last_x_col': None,
        'last_y_src': None,
        'last_y_col': None,
        'excluded_chips': set(),
        'valid_chips': []
    }

    # Extracts a list of all available data sources from the absolute analysis mode
    all_sources = list(data_catalog['Absolute'].keys())

    # Initialises the mode toggle buttons for analysis selection
    mode_toggle = widgets.ToggleButtons(options=['Absolute', 'Stage Delta', 'Intra-stage Kinetics'], style={'description_width': 'initial'})
    
    # Initialises the channel dropdown widget
    channel_dropdown = widgets.Dropdown(options=['Channel 1', 'Channel 2'], value='Channel 1', description='Channel:')

    # Initialises the X and Y source dropdown widgets
    x_source_drop = widgets.Dropdown(options=all_sources, value=all_sources[0], description='X Source:')
    y_source_drop = widgets.Dropdown(options=all_sources, value=all_sources[0], description='Y Source:')

    # Initialises the X and Y metric dropdown widgets
    x_col_drop = widgets.Dropdown(description='X Metric:')
    y_col_drop = widgets.Dropdown(description='Y Metric:')

    # Initialises the bounded float text inputs for X and Y minimum and maximum range filters
    x_min_input = widgets.BoundedFloatText(description='Min X:')
    x_max_input = widgets.BoundedFloatText(description='Max X:')
    y_min_input = widgets.BoundedFloatText(description='Min Y:')
    y_max_input = widgets.BoundedFloatText(description='Max Y:')

    # Initialises the buttons to reset X and Y filters to their original bounds
    x_reset_btn = widgets.Button(description='Reset X', button_style='info', tooltip='Reset X filters to original bounds')
    y_reset_btn = widgets.Button(description='Reset Y', button_style='info', tooltip='Reset Y filters to original bounds')

    def reset_x_filters(b):
        '''Resets the X-axis filters to their maximum available limits.'''
        
        # Restores the X input values to the absolute min/max bounds established during the update step
        x_min_input.value = x_min_input.min
        x_max_input.value = x_max_input.max

    def reset_y_filters(b):
        '''Resets the Y-axis filters to their maximum available limits.'''
        
        # Restores the Y input values to the absolute min/max bounds established during the update step
        y_min_input.value = y_min_input.min
        y_max_input.value = y_max_input.max

    # Binds the reset functions to the respective reset button click events
    x_reset_btn.on_click(reset_x_filters)
    y_reset_btn.on_click(reset_y_filters)

    # Groups the X and Y filters into horizontal box containers, initially hidden from view
    x_filters = widgets.HBox([x_min_input, x_max_input, x_reset_btn], layout=widgets.Layout(display='none'))
    y_filters = widgets.HBox([y_min_input, y_max_input, y_reset_btn], layout=widgets.Layout(display='none'))

    # Initialises the text input for filtering the available chips list
    chip_search = widgets.Text(placeholder='Filter available...', layout=widgets.Layout(width='95%'))
    
    # Initialises the selection box for available chips
    chip_select = widgets.Select(options=[], rows=5, layout=widgets.Layout(width='95%'))
    
    # Groups the available chips search and select widgets into the left vertical panel
    left_panel = widgets.VBox([widgets.HTML("<b>Available Chips:</b>"), chip_search, chip_select], layout=widgets.Layout(width='40%'))

    # Initialises the text input for filtering the excluded chips list
    excluded_search = widgets.Text(placeholder='Filter excluded...', layout=widgets.Layout(width='95%'))
    
    # Initialises the selection box for excluded chips
    excluded_select = widgets.Select(options=[], rows=5, layout=widgets.Layout(width='95%'))
    
    # Groups the excluded chips search and select widgets into the right vertical panel
    right_panel = widgets.VBox([widgets.HTML("<b>Excluded Chips:</b>"), excluded_search, excluded_select], layout=widgets.Layout(width='40%'))

    # Initialises the action buttons for moving chips between lists
    exclude_btn = widgets.Button(description='Exclude >', button_style='warning', layout=widgets.Layout(width='120px'))
    exclude_all_btn = widgets.Button(description='Exclude All >>', button_style='danger', layout=widgets.Layout(width='120px'))
    readd_btn = widgets.Button(description='< Re-add', button_style='info', layout=widgets.Layout(width='120px'))
    reset_btn = widgets.Button(description='<< Reset All', button_style='success', layout=widgets.Layout(width='120px'))
    
    # Creates a blank HTML spacer to vertically align the action buttons with the adjacent text inputs
    button_spacer = widgets.HTML("<b>&nbsp;</b>")
    
    # Groups the spacer and action buttons into the central vertical panel
    button_panel = widgets.VBox([button_spacer, exclude_btn, exclude_all_btn, readd_btn, reset_btn], layout=widgets.Layout(width='20%', justify_content='flex-start', align_items='center'))
    
    def refresh_ui(*args):
        '''Updates both chip listboxes based on current exclusions and dynamically valid chips.'''
        
        # Extracts the lowercased search term from the available chips filter
        avail_term = chip_search.value.lower()
        
        # Populates the available chips list matching the search term and omitting excluded chips
        chip_select.options = [c for c in state['valid_chips'] if c not in state['excluded_chips'] and avail_term in c.lower()]
        
        # Extracts the lowercased search term from the excluded chips filter
        excl_term = excluded_search.value.lower()
        
        # Populates the excluded chips list matching the search term and containing only excluded valid chips
        excluded_select.options = [c for c in state['valid_chips'] if c in state['excluded_chips'] and excl_term in c.lower()]

    # Binds the user interface refresh function to changes in either chip search box
    chip_search.observe(refresh_ui, names='value')
    excluded_search.observe(refresh_ui, names='value')

    def exclude_selected(b):
        '''Excludes the currently highlighted chip from the available list.'''
        
        # Checks if a chip is selected in the available chips list
        if chip_select.value:
            
            # Adds the selected chip to the exclusion set
            state['excluded_chips'].add(chip_select.value)
            
            # Refreshes the shuttle lists and redraws the plot
            refresh_ui()
            plot_data()

    def exclude_all_filtered(b):
        '''Excludes all chips currently visible in the filtered available list.'''
        
        # Checks if there are any chips present in the filtered available list
        if chip_select.options:
            
            # Updates the exclusion set with all currently visible chips
            state['excluded_chips'].update(chip_select.options)
            
            # Refreshes the shuttle lists and redraws the plot
            refresh_ui()
            plot_data()

    def readd_selected(b):
        '''Restores the currently highlighted chip from the excluded list back to the active pool.'''
        
        # Checks if a chip is selected and verifies it exists in the exclusion set
        if excluded_select.value and excluded_select.value in state['excluded_chips']:
            
            # Removes the selected chip from the exclusion set
            state['excluded_chips'].remove(excluded_select.value)
            
            # Refreshes the shuttle lists and redraws the plot
            refresh_ui()
            plot_data()

    def reset_exclusions(b):
        '''Clears all current exclusions and restores the default chip lists.'''
        
        # Checks if there are any chips currently excluded
        if state['excluded_chips']:
            
            # Clears the exclusion set entirely
            state['excluded_chips'].clear()
            
            # Empties both search boxes to remove active filters
            chip_search.value = ''
            excluded_search.value = ''
            
            # Refreshes the shuttle lists and redraws the plot
            refresh_ui()
            plot_data()

    # Binds the exclusion and re-addition functions to their respective button click events
    exclude_btn.on_click(exclude_selected)
    exclude_all_btn.on_click(exclude_all_filtered)
    readd_btn.on_click(readd_selected)
    reset_btn.on_click(reset_exclusions)

    # Groups the left, middle, and right panels into the final horizontal exclusion control box
    exclusion_controls = widgets.HBox([left_panel, button_panel, right_panel], layout=widgets.Layout(border='1px solid #ddd', padding='10px', margin='10px 0px', width='100%'))
    
    # Initialises the output widget for capturing and displaying the plotly figure
    out = widgets.Output()

    def update_sources(*args):
        '''Updates available data sources depending on the selected channel (removes Standard Curve for Ch2).'''
        
        # Exits the function if an update is already in progress
        if state['updating']: return
        
        # Sets the updating flag to True to prevent recursive event firing
        state['updating'] = True

        # Retrieves the currently selected channel
        ch = channel_dropdown.value
        
        # Removes the 'Standard Curve' option from the valid sources if Channel 2 is selected
        valid_sources = [s for s in all_sources if s != 'Standard Curve'] if ch == 'Channel 2' else all_sources
        
        # Caches the currently selected X and Y sources
        curr_x, curr_y = x_source_drop.value, y_source_drop.value

        # Updates the dropdown widget options with the valid sources
        x_source_drop.options = valid_sources
        y_source_drop.options = valid_sources

        # Restores the previous selections if valid, otherwise defaults to the first available option
        x_source_drop.value = curr_x if curr_x in valid_sources else valid_sources[0]
        y_source_drop.value = curr_y if curr_y in valid_sources else valid_sources[0]

        # Resets the updating flag to False
        state['updating'] = False
        
        # Triggers a downstream update of the feature dropdowns
        update_dropdowns()

    def update_dropdowns(*args):
        '''Refreshes feature dropdown options after the selected sources change.'''
        
        # Exits the function if an update is already in progress
        if state['updating']: return
        
        # Sets the updating flag to True to prevent recursive event firing
        state['updating'] = True

        # Retrieves the current mode, channel, and source values from the widgets
        mode = mode_toggle.value
        ch = channel_dropdown.value
        x_src, y_src = x_source_drop.value, y_source_drop.value

        # Ensures both X and Y sources are selected before extracting columns
        if x_src and y_src:
            
            # Extracts column names directly from the nested dictionary structure
            x_cols = list(data_catalog[mode][x_src][ch].columns)
            y_cols = list(data_catalog[mode][y_src][ch].columns)

            # Caches the currently selected metrics
            curr_x_col = x_col_drop.value
            curr_y_col = y_col_drop.value

            # Updates the metric dropdown options
            x_col_drop.options = x_cols
            y_col_drop.options = y_cols

            # Restores the previous metrics if valid, otherwise defaults to the first available option
            x_col_drop.value = curr_x_col if curr_x_col in x_cols else (x_cols[0] if x_cols else None)
            y_col_drop.value = curr_y_col if curr_y_col in y_cols else (y_cols[0] if y_cols else None)

        # Resets the updating flag to False
        state['updating'] = False
        
        # Triggers a downstream update of the filter limit bounds
        update_filter_bounds()

    def update_filter_bounds(*args):
        '''Dynamically sets the absolute min/max limits ONLY for the axes that were just modified.'''
        
        # Exits the function if an update is already in progress
        if state['updating']: return
        
        # Sets the updating flag to True to prevent recursive event firing
        state['updating'] = True

        # Retrieves the current core parameters from the widgets
        mode, ch = mode_toggle.value, channel_dropdown.value
        x_src, y_src = x_source_drop.value, y_source_drop.value
        x_col, y_col = x_col_drop.value, y_col_drop.value

        # Determines if the X-axis parameters changed since the last update
        needs_x_update = (mode != state.get('last_mode') or ch != state.get('last_ch') or x_src != state.get('last_x_src') or x_col != state.get('last_x_col'))
        
        # Determines if the Y-axis parameters changed since the last update
        needs_y_update = (mode != state.get('last_mode') or ch != state.get('last_ch') or y_src != state.get('last_y_src') or y_col != state.get('last_y_col'))

        # Clears chip exclusions entirely if axis sources or channels have significantly changed
        if (needs_x_update or needs_y_update) and state.get('last_mode') is not None:
            
            # Empties the exclusion set
            state['excluded_chips'].clear()
            
            # Resets the exclusion search boxes
            chip_search.value = ''
            excluded_search.value = ''

        # Defines a safe large numeric constant to bypass ipywidgets bounding validation errors
        LARGE_NUM = 1e30 

        # Processes X-axis filter bounds if an update is needed
        if needs_x_update:
            
            # Displays the X filter container layout
            x_filters.layout.display = 'flex'
            
            # Evaluates the X column if one is selected
            if x_col:
                
                # Retrieves the target dataframe
                df_x = data_catalog[mode][x_src][ch]
                
                # Checks if the dataframe contains the target column
                if not df_x.empty and x_col in df_x.columns:
                    
                    # Drops missing values from the target column
                    x_data = df_x[x_col].dropna()
                    
                    # Updates the widget constraints if valid data remains
                    if not x_data.empty:
                        
                        # Extracts the absolute minimum and maximum values as floats
                        x_min, x_max = float(x_data.min()), float(x_data.max())
                        
                        # Applies a slight offset if min and max are identical to prevent rendering errors
                        if x_min == x_max: x_max += 1e-9

                        # Expands the widget min/max constraints to the temporary large numbers
                        x_min_input.min, x_max_input.max = -LARGE_NUM, LARGE_NUM
                        x_max_input.min, x_min_input.max = -LARGE_NUM, LARGE_NUM
                        
                        # Sets the widget values to the true data extremes
                        x_min_input.value, x_max_input.value = x_min, x_max
                        
                        # Shrinks the widget constraints down to match the true data extremes
                        x_min_input.min, x_min_input.max = x_min, x_max
                        x_max_input.min, x_max_input.max = x_min, x_max
                        
            # Saves the newly established X configuration to the state dictionary
            state['last_x_src'], state['last_x_col'] = x_src, x_col

        # Processes Y-axis filter bounds if an update is needed
        if needs_y_update:
            
            # Displays the Y filter container layout
            y_filters.layout.display = 'flex'
            
            # Evaluates the Y column if one is selected
            if y_col:
                
                # Retrieves the target dataframe
                df_y = data_catalog[mode][y_src][ch]
                
                # Checks if the dataframe contains the target column
                if not df_y.empty and y_col in df_y.columns:
                    
                    # Drops missing values from the target column
                    y_data = df_y[y_col].dropna()
                    
                    # Updates the widget constraints if valid data remains
                    if not y_data.empty:
                        
                        # Extracts the absolute minimum and maximum values as floats
                        y_min, y_max = float(y_data.min()), float(y_data.max())
                        
                        # Applies a slight offset if min and max are identical to prevent rendering errors
                        if y_min == y_max: y_max += 1e-9

                        # Expands the widget min/max constraints to the temporary large numbers
                        y_min_input.min, y_max_input.max = -LARGE_NUM, LARGE_NUM
                        y_max_input.min, y_min_input.max = -LARGE_NUM, LARGE_NUM
                        
                        # Sets the widget values to the true data extremes
                        y_min_input.value, y_max_input.value = y_min, y_max
                        
                        # Shrinks the widget constraints down to match the true data extremes
                        y_min_input.min, y_min_input.max = y_min, y_max
                        y_max_input.min, y_max_input.max = y_min, y_max
                        
            # Saves the newly established Y configuration to the state dictionary
            state['last_y_src'], state['last_y_col'] = y_src, y_col

        # Saves the core overarching state parameters
        state['last_mode'], state['last_ch'] = mode, ch
        
        # Resets the updating flag to False
        state['updating'] = False
        
        # Triggers a downstream plot rendering update
        plot_data()

    def get_joined_data(mode, x_src, y_src, x_col, y_col, channel_label):
        '''Joins selected dashboard fields and applies an optional channel filter.

        Args:
            mode (str): Analysis mode selected in the dashboard.
            x_src (str): Name of the requested X data source.
            y_src (str): Name of the requested Y data source.
            x_col (str): Feature column selected for the X-axis.
            y_col (str): Feature column selected for the Y-axis.
            channel_label (str): Optional channel label used to filter rows.

        Returns:
            pd.DataFrame: Joined, complete observations for the selected fields.
        '''
        
        # Retrieves and copies the specific dataframes from the catalog using the channel label
        df_x = data_catalog[mode][x_src][channel_label].copy()
        df_y = data_catalog[mode][y_src][channel_label].copy()

        # Checks whether either dataframe is empty and returns an empty frame if so
        if df_x.empty or df_y.empty:
            return pd.DataFrame()

        # Ensures all columns are strings to allow safe merging
        df_x.columns = df_x.columns.astype(str)
        df_y.columns = df_y.columns.astype(str)
        x_col_str, y_col_str = str(x_col), str(y_col)

        # Inner joins the dataframes on their index, applying suffixes for safety
        df_merged = pd.merge(df_x[[x_col_str]], df_y[[y_col_str]], left_index=True, right_index=True, how='inner', suffixes=('_x', '_y'))
        
        # Resolves dynamic column renaming if the X and Y columns share the same name
        actual_x_col = x_col_str + '_x' if x_col_str == y_col_str else x_col_str
        actual_y_col = y_col_str + '_y' if x_col_str == y_col_str else y_col_str
        
        # Renames the columns to standardised keys for downstream plotting
        df_merged = df_merged.rename(columns={actual_x_col: 'x_val', actual_y_col: 'y_val'})

        # Replaces infinities with NaNs and drops missing values before returning
        return df_merged.replace([np.inf, -np.inf], np.nan).dropna()

    def plot_data(*args):
        '''Creates the dashboard scatter plot and handles all active filters.'''
        
        # Prevents redundant plotting cycles during chained updates
        if state['updating']: return

        # Targets the output widget context manager
        with out:
            
            # Clears the previous output before rendering the new plot
            out.clear_output(wait=True)
            
            # Retrieves the current core parameters from the widgets
            mode, channel = mode_toggle.value, channel_dropdown.value
            x_src, y_src = x_source_drop.value, y_source_drop.value
            x_col, y_col = x_col_drop.value, y_col_drop.value

            # Validates that actual columns are selected before proceeding
            if not x_col or not y_col:
                
                # Prints a warning message for invalid columns
                print('Please select valid metrics to plot.')
                
                # Clears the valid chips tracker
                state['valid_chips'] = []
                
                # Empties the chip selection UI
                refresh_ui()

                return

            # Creates a new Plotly figure object
            fig = go.Figure()
            
            # Identifies the active channels based on the dropdown selection
            channels_to_plot = ['Channel 1', 'Channel 2'] if channel == 'Both' else [channel]
            
            # Defines consistent colour mapping for the specific channels
            colors = {'Channel 1': '#1f77b4', 'Channel 2': '#ff1e0e'}
            
            # Initialises a flag to track if any data was successfully plotted
            plotted_any = False

            # Initialises an empty set to collect valid chip IDs found during joining
            current_valid_chips = set()
            
            # Initialises an empty dictionary to cache the joined datasets per channel
            raw_dfs = {}

            # Loops through each channel to fetch and map available data dynamically
            for ch in channels_to_plot:
                
                # Safely attempts to join and extract the necessary data for the channel
                try:
                    
                    # Retrieves the merged dataset for the current axes and channel
                    df = get_joined_data(mode, x_src, y_src, x_col, y_col, ch)
                    
                    # Verifies the dataset is not empty
                    if not df.empty:
                        
                        # Caches the dataframe to prevent duplicate joining downstream
                        raw_dfs[ch] = df
                        
                        # Adds the successfully joined chip indices to the master tracker
                        current_valid_chips.update(df.index.astype(str).tolist())
                        
                # Silently catches and skips extraction errors
                except Exception as e:
                    continue
            
            # Updates the global valid chips state and sorts it alphabetically
            state['valid_chips'] = sorted(list(current_valid_chips))
            
            # Cleans up the exclusions list to discard items that no longer exist in the new dataset
            state['excluded_chips'] = {c for c in state['excluded_chips'] if c in state['valid_chips']}
            
            # Refreshes the shuttle list UI to reflect the available valid chips
            refresh_ui()

            # Loops through each channel again to apply filters and construct the scatter plot
            for ch in channels_to_plot:
                
                # Skips the channel iteration if it lacks cached valid data
                if ch not in raw_dfs: continue
                
                # Retrieves the target dataframe from the local cache
                plot_df = raw_dfs[ch]

                # Creates a baseline mask array allowing all data points through initially
                mask = pd.Series(True, index=plot_df.index)

                # Applies the numeric X-axis widget filter thresholds to the mask
                mask &= (plot_df['x_val'] >= x_min_input.value) & (plot_df['x_val'] <= x_max_input.value)
                
                # Applies the numeric Y-axis widget filter thresholds to the mask
                mask &= (plot_df['y_val'] >= y_min_input.value) & (plot_df['y_val'] <= y_max_input.value)

                # Checks if there are any specific chips selected for exclusion
                if state['excluded_chips']:
                    
                    # Updates the mask to filter out rows whose indices match the excluded chips
                    mask &= ~plot_df.index.astype(str).isin(state['excluded_chips'])

                # Applies the compiled boolean mask to the dataframe
                plot_df = plot_df[mask]

                # Skips the trace creation entirely if there are insufficient data points remaining
                if len(plot_df) < 2: 
                    continue
                
                # Sets the global plot flag to true indicating success
                plotted_any = True
                
                # Extracts the isolated X and Y arrays for the plot
                x_data, y_data = plot_df['x_val'], plot_df['y_val']

                # Adds a scatter trace for the filtered points to the plotly figure
                fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name=f'{ch}', marker=dict(size=8, opacity=0.7, color=colors[ch], line=dict(width=1, color='DarkSlateGrey')), text=plot_df.index, hovertemplate='Chip ID: %{text}<br>X: %{x:.4f}<br>Y: %{y:.4f}<extra></extra>'))

                # Verifies there is variance in the X data to calculate regression safely
                if x_data.nunique() > 1:
                    
                    # Computes linear regression parameters using scipy stats
                    slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
                    
                    # Generates a sequence of points for drawing the linear fit
                    x_fit = np.linspace(x_data.min(), x_data.max(), 100)
                    y_fit = slope * x_fit + intercept

                    # Adds the linear fit trace line to the plotly figure, formatting the legend text
                    fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name=f'{ch} Fit (r={r_value:.3f}, R^2={r_value**2:.3f})', line=dict(color=colors[ch], dash='dash', width=2), hoverinfo='skip'))

            # Checks if the plotting loop completed without rendering any data
            if not plotted_any:
                
                # Prints an error advising the user to relax the filter criteria
                print('Not enough matching chip records to plot these selections within the specified filter bounds.')
                return

            # Formats the dynamic figure title string
            title = f'{y_src} [{y_col}] vs {x_src} [{x_col}]'
            
            # Updates the overarching layout settings and metadata for the final figure
            fig.update_layout(title=title, xaxis_title=f'{x_src} : {x_col}', yaxis_title=f'{y_src} : {y_col}', template='plotly_white', height=600, margin=dict(l=40, r=40, t=60, b=40), hovermode='closest')
            
            # Renders the figure inside the output block
            fig.show()

    # Binds the source update observer to changes in the channel dropdown
    channel_dropdown.observe(update_sources, 'value')
    
    # Binds the dropdown update observer to changes in the mode toggle and axis source dropdowns
    mode_toggle.observe(update_dropdowns, 'value')
    x_source_drop.observe(update_dropdowns, 'value')
    y_source_drop.observe(update_dropdowns, 'value')
    
    # Binds the filter bound update observer to changes in the target metric selections
    x_col_drop.observe(update_filter_bounds, 'value')
    y_col_drop.observe(update_filter_bounds, 'value')
    
    # Binds the plot redraw observer directly to changes in the bounded range input widgets
    x_min_input.observe(plot_data, 'value')
    x_max_input.observe(plot_data, 'value')
    y_min_input.observe(plot_data, 'value')
    y_max_input.observe(plot_data, 'value')

    # Groups all configured user interface elements into the master vertical container
    controls = widgets.VBox([mode_toggle, channel_dropdown, widgets.HBox([x_source_drop, x_col_drop]), x_filters, widgets.HBox([y_source_drop, y_col_drop]), y_filters, exclusion_controls])

    # Displays the dashboard title header element
    display(widgets.HTML(f"<h3 style='margin-bottom:0px; color:#0000FF;'>Master Chip Comparison Dashboard</h3>"))
    
    # Displays the master interface container and the integrated output display block
    display(controls, out)
    
    # Triggers the initial orchestration update to populate parameters and render the first plot
    update_sources()

def get_top_drivers(df, chip_id, top_n=3):
    '''Identifies the features that most strongly distinguish a chip from its peers.

    Args:
        df (pd.DataFrame): Feature table indexed by chip identifier.
        chip_id (str): Identifier of the chip to investigate.
        top_n (int, optional): Number of highest-impact features to return. Defaults to 3.

    Returns:
        list[str]: Ranked absolute z-scores formatted as strings for the selected chip.
    '''

    # Skips evaluation if the chip ID is not found in the standard index
    if chip_id not in df.index: 
        return ["N/A (Chip not in dataset)"]
    
    mean = df.mean()
    std = df.std().replace(0, 1e-9)
    z_scores = ((df.loc[chip_id] - mean) / std).abs().dropna()

    # Checks whether the z-scores series contains data
    if z_scores.empty: 
        return ["N/A"]
    
    top_stages = z_scores.sort_values(ascending=False).head(top_n)

    # Returns the formatted top drivers
    return [f'{stage} (Z: {z:.1f})' for stage, z in top_stages.items()]

def prepare_section(df, stage_col, val_col):

    '''Pivots the chip data and identifies outlier chips through anomaly detection.

    Args:
        df (pd.DataFrame): The input long-format dataframe containing chip metrics.
        stage_col (str): The name of the column representing the processing stage or feature.
        val_col (str): The name of the column containing the values to pivot.

    Returns:
        tuple[pd.DataFrame, list]: A tuple containing the pivoted wide-format dataframe and a list of identified outlier chip identifiers.
    '''

    # Filters out non-informative startup stages
    filtered_df = df[~df[stage_col].isin(['Start', 'Initial'])]

    # Pivots the data into a wide format using the specified stage and value columns
    pivoted = pivot_chip_data(filtered_df, stage_col, val_col)

    # Detects anomalies across all pivoted feature columns using automatic contamination scaling
    anomalies = detect_anomalies(pivoted, pivoted.columns.tolist())

    # Extracts the index identifiers for chips flagged as outliers
    outliers = get_outlier_chips(anomalies)

    return pivoted, outliers

def generate_at_risk_summary(master_df, sc_metrics, sc_raw_df, all_at_risk_chips, section_config, incomplete_chips=None, title='Overall At-Risk Chips Summary'):
    '''Clusters chips based on combined stage data, cross-references against standard curve metrics, and generates a structured summary report across 5 distinct sections.

    Args:
        master_df (pd.DataFrame): The master dataframe containing all features.
        sc_metrics (pd.DataFrame): Standard curve metrics.
        sc_raw_df (pd.DataFrame): Raw standard curve data.
        all_at_risk_chips (list): List of all chips identified as at-risk.
        section_config (dict): Configuration mapping for different summary sections.
        incomplete_chips (set/list, optional): Chips identified as having incomplete datasets.
        title (str, optional): The title for the summary. Defaults to 'Overall At-Risk Chips Summary'.
    '''

    # Initialises incomplete chips as an empty set if not provided
    if incomplete_chips is None:
        incomplete_chips = set()
    else:
        incomplete_chips = set(incomplete_chips)

    # Creates channel-order variables for the comparison results
    pel_unique = set()
    pel_breakdown = {}
    
    immob_unique = set()
    immob_breakdown = {}
    
    # Loops through each section name and outlier list in the configuration
    for section_name, (_, outliers) in section_config.items():

        # Aggregates PEL anomalies
        if 'PEL' in section_name:

            pel_unique.update(outliers)

            if outliers: 
                pel_breakdown[section_name] = outliers
                
        # Aggregates Immobilisation anomalies
        elif 'Immob' in section_name:

            immob_unique.update(outliers)

            if outliers: 
                immob_breakdown[section_name] = outliers

    sc_fails = set()

    # Checks whether the standard curve metrics dataframe contains data and identifies failing scores
    if sc_metrics is not None and not sc_metrics.empty and 'r2' in sc_metrics.columns:
        sc_fails = set(sc_metrics[sc_metrics['r2'] < 0.95].index)
        
    combined_all_fails = sorted(list(set(all_at_risk_chips).union(pel_unique, immob_unique, sc_fails)))

    # Displays the results in a collapsible output section
    with collapsible_output(f'{title} - Overall Summary'):

        print(f'Total Unique Faulty Chips: {len(combined_all_fails)}')

        # Lists the detected faulty chips or prints a clean status
        if combined_all_fails:

            print(f"Chip IDs: {', '.join(map(str, combined_all_fails))}\n")
            print(f'Total Unique PEL Fails: {len(pel_unique)}')
            print(f'Total Unique Immobilisation Fails: {len(immob_unique)}')
            print(f'Total Unique Standard Curve Fails: {len(sc_fails)}')

            print(f'Total Unique Chips with Incomplete Datasets: {len(incomplete_chips)}')
            
            if incomplete_chips:
                print(f"  - Incomplete Chip IDs: {', '.join(map(str, sorted(list(incomplete_chips))))}")
        else:
            print('No faults detected across any stage. Everything looks normal!')

    # Displays the results in a collapsible output section
    with collapsible_output(f'{title} - PEL Breakdown'):

        print(f'Total Unique PEL Faulty Chips: {len(pel_unique)}')
        print(pel_unique, '\n')

        # Expands the individual sections triggered within PEL
        if pel_unique:

            # Loops through each section name and outlier list in the PEL breakdown
            for sec_name, out_list in pel_breakdown.items():
                print(f"  - {sec_name}: {len(out_list)} chips ({', '.join(map(str, out_list))})")
        else:
            print("No anomalies detected in the PEL stage.")

    # Displays the results in a collapsible output section
    with collapsible_output(f'{title} - Immobilisation Breakdown'):

        print(f'Total Unique Immobilisation Faulty Chips: {len(immob_unique)}')
        print(immob_unique, '\n')

        # Expands the individual sections triggered within Immobilisation
        if immob_unique:

            # Loops through each section name and outlier list in the Immobilisation breakdown
            for sec_name, out_list in immob_breakdown.items():
                print(f"  - {sec_name}: {len(out_list)} chips ({', '.join(map(str, out_list))})")
        else:
            print("No anomalies detected in the Immobilisation stage.")

    # Displays the results in a collapsible output section
    with collapsible_output(f'{title} - Standard Curve Breakdown'):

        print(f'Total Unique Standard Curve Faulty Chips: {len(sc_fails)}')

        if sc_fails: 
            print(f"Chip IDs (R^2 < 0.95): {', '.join(map(str, sorted(list(sc_fails))))}")
        else:
            print("No Standard Curve anomalies detected (All R² >= 0.95).")

    # Displays the results in a collapsible output section
    with collapsible_output(f'{title} - Chip Profiles'):

        # Checks whether the master dataframe contains data or if there are combined fails
        if master_df.empty or not combined_all_fails:

            print('No detailed profiles available.')
            
            return

        features = master_df.columns.tolist()

        valid_rows = master_df[features].notna().all(axis=1)
        master_df['Cluster'] = pd.NA

        if valid_rows.sum() >= 2:

            X_master = RobustScaler().fit_transform(master_df.loc[valid_rows, features])
        
            # Re-clusters the master dataset for contextual grouping
            kmeans = KMeans(n_clusters=2, random_state=8030, n_init=10)

            master_df.loc[valid_rows, 'Cluster'] = kmeans.fit_predict(X_master)

        # Attempts to resolve datetime columns from the raw records
        date_col = next((c for c in sc_raw_df.columns if 'date' in c.lower()), None)
        time_col = next((c for c in sc_raw_df.columns if 'time' in c.lower()), None)

        cols_to_merge = ['standard_curve_uuid']

        if date_col: 
            cols_to_merge.append(date_col)

        if time_col:
            cols_to_merge.append(time_col)

        chip_summaries = {}

        # Loops through each faulty chip to build its detailed profile
        for chip in combined_all_fails:
            
            # Identifies the assigned cluster for the current chip
            if chip in master_df.index.get_level_values(master_df.index.name or 'chip_id'):
                
                # Check if it's a MultiIndex (e.g. chip_id + channel)
                if isinstance(master_df.index, pd.MultiIndex):
                    chip_clusters = master_df.xs(chip, level='chip_id')['Cluster'].to_dict()

                    # Handle NaNs in the dictionary
                    cluster_profile = ' | '.join([f"{ch}: {'N/A' if pd.isna(cl) else f'Cluster {int(cl)}'}" for ch, cl in chip_clusters.items()])
                
                # If it's a flat index (just chip_id)
                else:
                    cl = master_df.loc[chip, 'Cluster']

                    if pd.isna(cl):
                        cluster_profile = 'N/A (Incomplete Stage Data)'
                    else:
                        cluster_profile = f'Cluster {int(cl)}'
                    
            else:
                cluster_profile = 'N/A (Incomplete Stage Data)'
                
            # Creates a copy of the standard curve rows to preserve the original data
            sc_rows = sc_metrics[sc_metrics.index == chip].copy()

            # Checks whether the standard curve dataframe contains data and merges date details
            if not sc_rows.empty and len(cols_to_merge) > 1:
                sc_rows = sc_rows.merge(sc_raw_df[cols_to_merge], on='standard_curve_uuid', how='left')
            
            detailed_flags = []

            # Checks whether the standard curve dataframe contains data and appends failing flags
            if not sc_rows.empty and (sc_rows['r2'] < 0.95).any():
                detailed_flags.append('Standard Curve (R² < 0.95)')
                
            drivers_info = {}

            # Loops through each section configuration to aggregate driving metrics
            for section_name, (df_source, outlier_list) in section_config.items():

                if chip in outlier_list:
                    detailed_flags.append(section_name)
                    drivers_info[section_name] = get_top_drivers(df_source, chip)
            
            sc_data = []

            # Checks whether the standard curve dataframe contains data
            if not sc_rows.empty:

                # Loops through each row in the standard curve dataframe
                for _, sc_row in sc_rows.iterrows():

                    r2_val = sc_row['r2']
                    d_str = str(sc_row[date_col]).strip() if date_col and pd.notna(sc_row[date_col]) else ''
                    t_str = str(sc_row[time_col]).strip().replace('-', ':') if time_col and pd.notna(sc_row[time_col]) else ''
                    
                    # Appends the formatted metrics to the summary list
                    sc_data.append({
                        'Date/Time': f'{d_str} {t_str}'.strip() or 'Unknown',
                        'Curve UUID': sc_row['standard_curve_uuid'],
                        'Fit Model': sc_row['fit_model'],
                        'R² Score': r2_val,
                        'Status': 'FAILED' if r2_val < 0.95 else 'PASSED'
                    })

            sc_df = pd.DataFrame(sc_data)
            
            # Checks whether the standard curve dataframe contains data and sorts chronologically
            if not sc_df.empty and 'Date/Time' in sc_df.columns:

                sc_df['Parsed_DateTime'] = pd.to_datetime(sc_df['Date/Time'], errors='coerce')
                sc_df = sc_df.sort_values(by='Parsed_DateTime').drop(columns=['Parsed_DateTime']).reset_index(drop=True)
            
            # Saves the assembled chip profile to the summary dictionary
            chip_summaries[chip] = {
                'Cluster': cluster_profile,
                'Flagged By': detailed_flags,
                'Drivers': drivers_info,
                'SC_DF': sc_df
            }

        def style_status(val):
            '''Returns a CSS style declaration for a standard-curve status value.

            Args:
                val (object): Status value to evaluate.

            Returns:
                str: CSS declaration appropriate for the status.
            '''
        
            if val == 'FAILED': 
                return 'color: red; font-weight: bold;'

            if val == 'PASSED': 
                return 'color: green;'

            return ''
        
        # Loops through each chip and its information dictionary
        for chip, info in chip_summaries.items():

            print('=' * 80)
            print(f'CHIP ID: {chip}')
            print(f"Cluster Profile: {info['Cluster']}")
            
            # Formats and prints the sources flagging this chip
            flag_str = '\n    - '.join(info['Flagged By']) if info['Flagged By'] else 'None (Manual Review)'
            print(f'Flagged By Anomaly In:\n    - {flag_str}')
            
            if info['Drivers']:

                print('\nTop 3 Divergent Regions for Triggered Sections:')

                # Loops through each section and its drivers
                for section, drivers in info['Drivers'].items():

                    print(f'\n  - {section}:')

                    # Loops through each driver item
                    for d in drivers: 
                        print(f'      {d}')

            print('-' * 80)
            
            # Checks whether the standard curve dataframe contains data and displays it
            if not info['SC_DF'].empty:

                # Applies styling formats to the standard curve summary
                styled_df = info['SC_DF'].style.format({'R² Score': '{:.4f}'}).set_properties(**{'text-align': 'left', 'white-space': 'nowrap'}).set_table_styles([dict(selector='th', props=[('text-align', 'left')])]).map(style_status, subset=['Status'])

                display(styled_df)
            else:
                print('No Standard Curve data available for this chip.')
                
            print('\n')

# Binding Kinetics
def parse_conc_flag(conc_str):
    '''Parses a concentration flag label to extract just the numeric concentration.

    Args:
        conc_str (str): The raw concentration string flag.

    Returns:
        float | None: The parsed numeric concentration or None if extraction fails.
    '''
    
    # Extracts the numeric value from the concentration string using regular expressions
    match = re.search(r'([\d.]+)', str(conc_str))

    # Returns None if no numeric match is found
    if not match:
        return None
    
    # Converts the matched string group to a float and returns it
    return float(match.group(1))

def assoc_model(t, Req, kobs, R0, RI):
    '''Calculates Association with Bulk Shift (RI).

    Args:
        t (array-like): Time values.
        Req, kobs, R0, RI (float): Association model parameters.

    Returns:
        array-like: Predicted association response.
    '''

    # Calculates and returns the predicted association response
    return R0 + RI + Req * (1 - np.exp(-kobs * (t - t[0])))

def dissoc_model(t, R0, kd, Rinf, RI):
    '''Calculates Dissociation with Bulk Shift (RI) dropping off.

    Args:
        t (array-like): Time values.
        R0, kd, Rinf, RI (float): Dissociation model parameters.

    Returns:
        array-like: Predicted dissociation response.
    '''

    # Calculates and returns the predicted dissociation response
    return Rinf + (R0 - Rinf - RI) * np.exp(-kd * (t - t[0]))

def fit_association(t, y):
    '''Fits the association-phase model to a time-series response.

    Args:
        t (array-like): Association-phase time values.
        y (array-like): Measured response values.

    Returns:
        tuple | None: Fitted parameters and covariance, or None.
    '''

    # Estimates the initial response value
    R0_guess = y[0]
    
    # Estimates the equilibrium response value
    Req_guess = y[-1] - y[0]

    # Attempts to fit the association model to the data
    try:
        
        # Defines lower bounds for the curve fit parameters
        lower_bounds = [-np.inf, 0.0, -np.inf, -np.inf]
        
        # Defines upper bounds for the curve fit parameters
        upper_bounds = [np.inf, np.inf, np.inf, np.inf]

        # Executes the curve fitting algorithm with the initial guesses and bounds
        popt, _ = curve_fit(assoc_model, t, y, p0=[Req_guess, 0.01, R0_guess, 0.0], bounds=(lower_bounds, upper_bounds), maxfev=10000)

        # Returns the optimised parameters
        return popt

    # Catches runtime errors if optimal parameters cannot be found
    except RuntimeError:
        return None

def fit_dissociation(t, y):
    '''Fits the dissociation-phase model to a time-series response.

    Args:
        t (array-like): Dissociation-phase time values.
        y (array-like): Measured response values.

    Returns:
        tuple | None: Fitted parameters and covariance, or None.
    '''

    # Estimates the initial response value for dissociation
    R0_guess = y[0]
    
    # Estimates the infinite response value
    Rinf_guess = y[-1]

    # Attempts to fit the dissociation model to the data
    try:
        
        # Defines lower bounds for the curve fit parameters
        lower_bounds = [-np.inf, 0.0, -np.inf, -np.inf]
        
        # Defines upper bounds for the curve fit parameters
        upper_bounds = [np.inf, np.inf, np.inf, np.inf]

        # Executes the curve fitting algorithm with the initial guesses and bounds
        popt, _ = curve_fit(dissoc_model, t, y, p0=[R0_guess, 0.01, Rinf_guess, 0.0], bounds=(lower_bounds, upper_bounds), maxfev=10000)

        # Returns the optimised parameters
        return popt

    # Catches runtime errors if optimal parameters cannot be found
    except RuntimeError:
        return None

def plot_kinetic_curves(data, title):
    '''Plots fitted association and dissociation curves for one experiment, grouped by measurement for interactive toggling.'''

    # Initialises the Matplotlib figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Initialises the Plotly figure for interactive plotting
    fig_plotly = go.Figure()
    
    # Defines the colour map for the plotted curves
    cmap = plt.cm.tab20
    
    # Loops through each index and fit dictionary in the data
    for i, f in enumerate(data['fits']):

        # Assigns a specific colour from the colour map based on the index
        colour = cmap(i % 20)
        hex_colour = mcolors.to_hex(colour)
        
        # Extracts the association and dissociation zones
        assoc_zone = f['assoc_zone']
        dissoc_zone = f['dissoc_zone']

        # Skips the current iteration if the association zone contains no data
        if assoc_zone.empty:
            continue

        # Identifies the start time of the association zone
        t0 = assoc_zone['time'].iloc[0]
        
        # Prepares the raw time and response data, adjusting for the dissociation zone if present
        if not dissoc_zone.empty:

            t_raw = pd.concat([assoc_zone['time'], dissoc_zone['time']]) - t0
            y_raw = pd.concat([assoc_zone['response'], dissoc_zone['response']])
        else:
            t_raw = assoc_zone['time'] - t0
            y_raw = assoc_zone['response']
            
        # Defines a unique legend group for this specific measurement to bind the raw data and fits together
        group_id = str(f['meas_id'])
            
        # Adds the raw data trace to the Plotly figure
        fig_plotly.add_trace(go.Scatter(x=t_raw, y=y_raw, mode='lines', line=dict(color=hex_colour, width=1.5), name=f['meas_id'], legendgroup=group_id))
        
        # Plots the raw association data on the Matplotlib axes
        ax.plot(assoc_zone['time'] - t0, assoc_zone['response'], color=colour, lw=1.5)

        # Plots the raw dissociation data on the Matplotlib axes if present
        if not dissoc_zone.empty:
            ax.plot(dissoc_zone['time'] - t0, dissoc_zone['response'], color=colour, lw=1.5)

        # Calculates and plots the fitted association model line if successfully derived
        if f['assoc_fit'] is not None:

            Req, kobs, R0, RI_a = f['assoc_fit']
            t_fit = assoc_zone['time'].values
            y_fit = assoc_model(t_fit, Req, kobs, R0, RI_a)

            # Adds the fit line to the same legend group, hiding the duplicate legend entry
            fig_plotly.add_trace(go.Scatter(x=t_fit - t0, y=y_fit, mode='lines', line=dict(color='black', dash='dash', width=1.5), showlegend=False, hoverinfo='skip', legendgroup=group_id))

            ax.plot(t_fit - t0, y_fit, '--', color='black', lw=1.2)

        # Calculates and plots the fitted dissociation model line if successfully derived
        if f['dissoc_fit'] is not None:

            R0d, kd, Rinf, RI_d = f['dissoc_fit']
            t_fit_d = dissoc_zone['time'].values
            y_fit_d = dissoc_model(t_fit_d, R0d, kd, Rinf, RI_d)

            # Adds the fit line to the same legend group, hiding the duplicate legend entry
            fig_plotly.add_trace(go.Scatter(x=t_fit_d - t0, y=y_fit_d, mode='lines', line=dict(color='black', dash='dash', width=1.5), showlegend=False, hoverinfo='skip', legendgroup=group_id))

            ax.plot(t_fit_d - t0, y_fit_d, '--', color='black', lw=1.2)

        # Adds an empty plot element to generate the legend entry
        ax.plot([], [], color=colour, label=f['meas_id'])

    # Updates and displays the final Plotly layout
    fig_plotly.update_layout(title=title, xaxis_title='Time from injection start (s)', yaxis_title='Response (RU, ref-subtracted)', template='plotly_white', legend_title_text='Measurement')
    fig_plotly.show()

    # Formats and displays the Matplotlib graph
    ax.set_xlabel('Time from injection start (s)')
    ax.set_ylabel('Response (RU, ref-subtracted)')
    ax.set_title(title)
    
    ax.legend(title='Measurement', fontsize=8, loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_rates_vs_conc(data, title_prefix):
    '''Plots k_obs, k_a, k_d, and K_D against concentration as sequential plots.
    Includes all data points, even physically impossible negative values, for diagnostic purposes.

    Args:
        data (dict): Kinetic results containing fits and global rates.
        title_prefix (str): Prefix for the plot titles.
    '''

    # Extracts concentrations and observed association rates for successful fits
    concs_kobs = [f['conc'] for f in data['fits'] if f['assoc_fit'] is not None]
    kobs_vals = [f['assoc_fit'][1] for f in data['fits'] if f['assoc_fit'] is not None]

    if concs_kobs:

        # Initialises the figure and plots the k_obs scatter points
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(concs_kobs, kobs_vals, color='tab:blue')

        # Generates and plots the linear fit line if sufficient data exists
        if len(concs_kobs) >= 2 and 'ka' in data:
            
            x_fit = np.linspace(0, max(concs_kobs), 50)
            y_fit = data['ka'] * x_fit + data['kd']
            ax.plot(x_fit, y_fit, '--', color='black', label=f"ka={data['ka']:.2e}, kd={data['kd']:.2e}")
            ax.legend()

        # Formats and displays the k_obs plot
        ax.set_xlabel('Concentration (ug/ml)')
        ax.set_ylabel('k_obs (s$^{-1}$)')
        ax.set_title(f'{title_prefix} - k_obs')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    # Initialises empty lists to store individual parameters
    concs_params = []
    ka_vals = []
    kd_vals = []
    KD_vals = []

    # Loops through each generated curve fit to calculate individual rates
    for f in data['fits']:

        # Extracts parameters if both association and dissociation fits are successful
        if f['assoc_fit'] is not None and f['dissoc_fit'] is not None and f['conc'] > 0:

            kobs = f['assoc_fit'][1]
            kd = f['dissoc_fit'][1]
            ka_indiv = (kobs - kd) / f['conc']
            
            # Calculates KD, avoiding division by zero
            KD_indiv = kd / ka_indiv if ka_indiv != 0 else np.nan
            
            concs_params.append(f['conc'])
            ka_vals.append(ka_indiv)
            kd_vals.append(kd)
            KD_vals.append(KD_indiv)

    # Skips the remaining plots if no valid data points exist
    if not concs_params:

        print("No valid fits available for additional parameter plots.")
        return

    # Initialises the figure and plots the ka scatter points
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(concs_params, ka_vals, color='tab:green')
    
    # Formats and displays the ka plot (Linear scale allows negatives)
    ax.set_xlabel('Concentration (ug/ml)')
    ax.set_ylabel('k_a (ug/ml*s)$^{-1}$')
    ax.set_title(f'{title_prefix} - Association Rate (k_a)')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    # Initialises the figure and plots the kd scatter points
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(concs_params, kd_vals, color='tab:red')
    
    # Formats and displays the kd plot
    ax.set_xlabel('Concentration (ug/ml)')
    ax.set_ylabel('k_d (s$^{-1}$)')
    ax.set_title(f'{title_prefix} - Dissociation Rate (k_d)')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    # Initialises  figure and plots the KD scatter points
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(concs_params, KD_vals, color='tab:purple')
    
    # Formats and displays the KD plot with a symmetrical logarithmic y-axis
    ax.set_xlabel('Concentration (ug/ml)')
    ax.set_ylabel('K_D (ug/ml)')
    ax.set_title(f'{title_prefix} - Equilibrium Constant (K_D)')
    ax.set_yscale('symlog')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_binding_curves(data, title, pre_baseline_seconds=10, tail_avg_seconds=5):
    '''Plots baseline-aligned binding curves using configurable sampling windows.

    Args:
        data (dict): Sensorgram measurements to align and plot.
        title (str): Plot title.
        pre_baseline_seconds (float, optional): Duration of the pre-event baseline window. Defaults to 10.
        tail_avg_seconds (float, optional): Duration of the tail-averaging window. Defaults to 5.
    '''

    # Extracts the raw sensorgram data
    sensor = data['sensor']
    
    # Initialises the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Initialises the Plotly figure
    fig_plotly = go.Figure()
    
    # Defines the colour map for the plotted curves
    cmap = plt.cm.viridis
    
    # Determines the total number of fits for colour scaling
    n = len(data['fits'])

    curve_id = 0

    # Loops through each index and fit dictionary in the data
    for i, f in enumerate(data['fits']):

        assoc_zone = f['assoc_zone']

        # Skips the current iteration if the association zone contains no data
        if assoc_zone.empty:
            continue

        # Assigns a specific colour from the colour map based on the index
        colour = cmap(i / max(n - 1, 1)) 
        hex_colour = mcolors.to_hex(colour)
        
        # Identifies the start time of the association zone
        t0 = assoc_zone['time'].iloc[0]

        # Slices out a specific pre-event window to calculate the standard baseline
        window = sensor[(sensor['time'] >= t0 - pre_baseline_seconds) & (sensor['time'] <= assoc_zone['time'].iloc[-1])]
        pre_zone = window[window['time'] < t0]
        baseline = pre_zone['response'].mean() if not pre_zone.empty else window['response'].iloc[0]

        # Aligns the time and response data relative to the baseline and event start
        t_plot = window['time'] - t0
        y_plot = window['response'] - baseline

        label = f['meas_id']
        
        # Adds the aligned trace to the Plotly figure
        fig_plotly.add_trace(go.Scatter(x=t_plot, y=y_plot, mode='markers', marker=dict(color=hex_colour, size=4), name=label))

        # Plots the aligned trace on the Matplotlib axes
        ax.scatter(t_plot, y_plot, s=18, label=label, color=colour)

        # Calculates plateau levels based on the configured tail-averaging duration
        tail_zone = y_plot[t_plot >= t_plot.max() - tail_avg_seconds]
        plateau_val = tail_zone.mean()

        # Determines coordinates for the plateau annotation lines
        t_max = t_plot.max()
        t_range = t_max - t_plot.min()
        x0_pl = t_max - (0.15 * t_range) 
        
        # Adds plateau indicators and annotations to the Plotly figure
        fig_plotly.add_trace(go.Scatter(x=[x0_pl, t_max], y=[plateau_val, plateau_val], mode='lines', line=dict(color=hex_colour, dash='dot', width=2), showlegend=False, hoverinfo='skip'))
        fig_plotly.add_annotation(x=t_max, y=plateau_val, text=f'{plateau_val:.3f}', showarrow=False, xanchor='left', xshift=5, font=dict(size=11, color=hex_colour))

        # Adds plateau indicators and annotations to the Matplotlib axes
        ax.axhline(y=plateau_val, xmin=0.85, xmax=1.0, linestyle=':', color=colour, linewidth=1.2)
        ax.annotate(f'{plateau_val:.3f}', xy=(1.0, plateau_val), xycoords=('axes fraction', 'data'), xytext=(5, 0), textcoords='offset points', va='center', ha='left', fontsize=9)

        curve_id += 1

    # Updates and displays the final Plotly layout
    fig_plotly.update_layout(title=title, xaxis_title='Time (sec)', yaxis_title='Signal (RU)', template='plotly_white', legend_title_text='Measurement')
    fig_plotly.show()

    # Formats and displays the fallback Matplotlib graph
    ax.set_xlabel('Time(sec)')
    ax.set_ylabel('Signal (RU)')
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_overall_rates_vs_conc(sc_data, title_prefix):
    '''Plots boxplots and stripplots for individual kinetic rates (k_obs, k_a, k_d, K_D)
    grouped by concentration across all assay measurements.

    Args:
        sc_data (list): A list of data dictionaries containing 'fits'.
        title_prefix (str): Prefix for the plot titles.
    '''
    
    records = []
    
    # Extracts the individual per-concentration rates from every chip in the dataset
    for data in sc_data:

        if 'fits' not in data:
            continue
            
        for f in data['fits']:

            if f['assoc_fit'] is not None and f['dissoc_fit'] is not None and f['conc'] > 0:

                kobs = f['assoc_fit'][1]
                kd = f['dissoc_fit'][1]
                ka_indiv = (kobs - kd) / f['conc']
                KD_indiv = kd / ka_indiv if ka_indiv != 0 else np.nan
                
                records.append({
                    'Concentration': f['conc'],
                    'k_obs': kobs,
                    'k_a': ka_indiv,
                    'k_d': kd,
                    'K_D': KD_indiv
                })
    
    # Validates that we successfully extracted data
    if not records:

        print("No individual per-concentration rates found across the dataset.")
        return

    # Converts to a DataFrame and sorts by concentration to ensure ordered x-axis categories
    df = pd.DataFrame(records)
    df = df.sort_values('Concentration')

    # Defines the metrics, labels, and formatting rules
    metrics = [
        ('k_obs', 'k_obs (s$^{-1}$)', 'tab:blue', 'lightblue', False),
        ('k_a', 'k_a (ug/ml*s)$^{-1}$', 'tab:green', 'lightgreen', False),
        ('k_d', 'k_d (s$^{-1}$)', 'tab:red', 'lightcoral', False),
        ('K_D', 'K_D (ug/ml)', 'tab:purple', '#d8b4e2', True)
    ]

    # Generates a separate grouped boxplot/stripplot for each metric
    for col, ylabel, color, boxcolor, is_symlog in metrics:
        
        # Drops missing values for the specific metric being plotted
        df_plot = df.dropna(subset=[col])

        if df_plot.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Seaborn automatically groups discrete x-values (Concentration) into categories
        sns.boxplot(data=df_plot, x='Concentration', y=col, color=boxcolor, width=0.4, showfliers=False, ax=ax)
        sns.stripplot(data=df_plot, x='Concentration', y=col, color=color, size=6, jitter=True, alpha=0.8, ax=ax)
        
        if is_symlog:
            ax.set_yscale('symlog')
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        elif col == 'k_a':
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
            
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Concentration (ug/ml)')
        ax.set_title(f'{title_prefix} - {col} grouped by Concentration')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()

def run_binding_kinetics_analysis(files, baseline='- 2'):
    '''Computes binding kinetics (association/dissociation, global ka/kd/KD).
    
    Uses calculated baseline flags to dynamically define fitting windows:
    1. Association: From injection start to the absolute peak.
    2. Dissociation: From the pre-drop peak strictly to the specified baseline.

    Args:
        files (dict): Processed sensorgram and flag data files.
        baseline (str): The baseline flag to use for the end of dissociation (default: '- 2').

    Returns:
        dict: The updated kinetics results.
    '''

    kinetics_results = {}
    
    # Clean the input target to guarantee matching regardless of spacing (e.g., '- 2' becomes '-2')
    target_baseline = str(baseline).replace(' ', '')

    for folder in files.keys():

        if 'baseline' in folder:
            continue

        kinetics_results[folder] = {}
        previous_file = ''

        for file in list(files[folder].keys()):

            if 'sensorgram' in file:
       
                sensor = files[folder][file].sort_values('time').reset_index(drop=True)
                flags = files[folder][previous_file].sort_values('time').reset_index(drop=True)
                sensor['response'] = sensor['channel1']

                file_name_flags = file.split('_')[1] + '_baseline_flags'
                baseline_flags = files[folder].get(file_name_flags)
                parsed = []

                for _, row in flags.iterrows():
                    conc_val = parse_conc_flag(row['conc'])

                    if conc_val is not None:
                        parsed.append({'time': row['time'], 'conc': conc_val, 'label': str(row['conc'])})

                segments = []

                if baseline_flags is not None:

                    # DYNAMIC BASELINE FILTER: Strips spaces from the data to match the cleaned target
                    info_str = baseline_flags['information'].astype(str).str.replace(' ', '')
                    target_flags = baseline_flags[info_str.str.contains(target_baseline, na=False)]

                    for i, event in enumerate(parsed):

                        t_start = event['time']
                        t_next = parsed[i+1]['time'] if i + 1 < len(parsed) else sensor['time'].max()

                        # Matches the event window to the filtered target baseline timings
                        valid_flags = target_flags[(target_flags['absolute_peak_time'] >= t_start) & (target_flags['absolute_peak_time'] <= t_next)]
                        
                        if valid_flags.empty:
                            continue
                            
                        flag_row = valid_flags.iloc[0]
                        
                        t_assoc_start = t_start
                        t_assoc_end = flag_row['absolute_peak_time']
                        
                        t_dissoc_start = flag_row['peak_time']
                        t_dissoc_end = flag_row['Window_Start']

                        if pd.isna(t_assoc_end) or pd.isna(t_dissoc_start) or pd.isna(t_dissoc_end):
                            continue

                        segments.append({
                            'conc': event['conc'], 
                            'meas_id': event['label'], 
                            't_assoc_start': t_assoc_start,
                            't_assoc_end': t_assoc_end,
                            't_dissoc_start': t_dissoc_start,
                            't_dissoc_end': t_dissoc_end
                        })

                fits = []

                for seg in segments:

                    assoc_zone = sensor[(sensor['time'] >= seg['t_assoc_start']) & (sensor['time'] <= seg['t_assoc_end'])]
                    dissoc_zone = sensor[(sensor['time'] >= seg['t_dissoc_start']) & (sensor['time'] <= seg['t_dissoc_end'])]

                    assoc_fit = fit_association(assoc_zone['time'].values, assoc_zone['response'].values) if len(assoc_zone) > 5 else None
                    dissoc_fit = fit_dissociation(dissoc_zone['time'].values, dissoc_zone['response'].values) if len(dissoc_zone) > 5 else None

                    fits.append({
                        'conc': seg['conc'], 'meas_id': seg['meas_id'],
                        'assoc_fit': assoc_fit, 'dissoc_fit': dissoc_fit,
                        'assoc_zone': assoc_zone, 'dissoc_zone': dissoc_zone
                    })

                file_data = {'sensor': sensor, 'segments': segments, 'fits': fits}
                concs = [f['conc'] for f in fits if f['assoc_fit'] is not None]
                kobs_vals = [f['assoc_fit'][1] for f in fits if f['assoc_fit'] is not None]

                if len(concs) >= 2:
                    slope, intercept, r_value, _, _ = linregress(concs, kobs_vals)
                    ka = slope
                    kd_from_intercept = intercept
                    KD = kd_from_intercept / ka if ka != 0 else np.nan

                    file_data['ka'] = ka
                    file_data['kd'] = kd_from_intercept
                    file_data['KD'] = KD
                    file_data['kobs_r2'] = r_value ** 2

                    dissoc_kds = [f['dissoc_fit'][1] for f in fits if f['dissoc_fit'] is not None]

                    if dissoc_kds:
                        file_data['kd_dissoc_mean'] = np.mean(dissoc_kds)

                kinetics_results[folder][file] = file_data

            previous_file = file

    all_global_data = []

    for folder, folder_files in kinetics_results.items():
        for file, data in folder_files.items():
            if 'fits' in data:
                all_global_data.append(data)
                
    if all_global_data:
        
        print(f"\n{'='*40}\nOverall Kinetics Distributions\n{'='*40}")
            
        with collapsible_output('Overall Distributions by Concentration'):
            plot_overall_rates_vs_conc(all_global_data, 'Overall')
            
    for folder, folder_files in kinetics_results.items():

        print(f"\n{'='*40}\nKinetics Analysis: {folder}\n{'='*40}")

        summary_rows = []

        for file, data in folder_files.items():
            if 'ka' in data:
                summary_rows.append({
                    'File': file, 'k_a (ug/ml*s)^-1': data['ka'], 'k_d_intercept (s^-1)': data['kd'],
                    'k_d_dissoc_mean (s^-1)': data.get('kd_dissoc_mean', np.nan),
                    'K_D (ug/ml)': data['KD'], 'k_obs_fit_R2': data['kobs_r2']
                })

        if summary_rows:
            with collapsible_output(f'Global Kinetics Summary: {folder}'):
                display(pd.DataFrame(summary_rows))

        for file, data in folder_files.items():

            if 'fits' not in data: 
                continue

            clean_filename = file.replace('Copy of ', '').strip()
            parts = clean_filename.split('_')
            chip_id = parts[0] if len(parts) > 0 else 'Unknown'
            measurement_id = parts[1] if len(parts) > 1 else 'Unknown'
            
            print(f'\n--- Chip ID: {chip_id} | Measurement ID: {measurement_id} ---')
            per_conc_rows = []

            for f in data['fits']:

                warning = ''
                                
                Req, kobs, R0 = (f['assoc_fit'][0], f['assoc_fit'][1], f['assoc_fit'][2]) if f['assoc_fit'] is not None else (np.nan, np.nan, np.nan)

                if f['assoc_fit'] is None: 
                    warning += 'Assoc fit failed; '
                    
                R0d, kd, Rinf = (f['dissoc_fit'][0], f['dissoc_fit'][1], f['dissoc_fit'][2]) if f['dissoc_fit'] is not None else (np.nan, np.nan, np.nan)

                if f['dissoc_fit'] is None: 
                    warning += 'Dissoc fit failed; '
                    
                if not np.isnan(kobs) and not np.isnan(kd) and f['conc'] > 0:

                    ka_indiv = (kobs - kd) / f['conc']
                    KD_indiv = kd / ka_indiv if ka_indiv != 0 else np.nan

                    if kd >= kobs: 
                        warning += 'kd >= kobs (Negative k_a); '
                else:
                    ka_indiv, KD_indiv = np.nan, np.nan
                    
                per_conc_rows.append({
                    'Measurement': f['meas_id'], 'Concentration (ug/ml)': f['conc'], 'k_obs (s^-1)': kobs,
                    'k_d (s^-1)': kd, 'k_a (calc)': ka_indiv, 'K_D': KD_indiv, 'R_eq': Req, 'Warning': warning.strip('; ')
                })

            if per_conc_rows:

                df_per_conc = pd.DataFrame(per_conc_rows).sort_values(by=['Concentration (ug/ml)', 'Measurement']).reset_index(drop=True)

                with collapsible_output(f'Per-Concentration Details: {file}'):
                    display(df_per_conc)

            with collapsible_output(f'Kinetic Fit Curves: {file}'):
                plot_kinetic_curves(data, f'Kinetic Curves - {chip_id}')

            if 'ka' in data:
                with collapsible_output(f'Rates vs Concentration: {file}'):
                    plot_rates_vs_conc(data, f'Rates vs Concentration - {chip_id}')

            with collapsible_output(f'Binding Curves: {file}'):
                plot_binding_curves(data, f'Binding Curves Overlay - {chip_id}')
     
    return kinetics_results
