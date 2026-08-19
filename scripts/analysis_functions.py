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

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import linregress

from scripts.setup_functions import collapsible_output

def detect_anomalies(df, feature_cols, contamination='auto'):
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
    if df_clean.empty:
        return df_clean
        
    # Initialises the scaler and standardises the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[feature_cols])
    
    # Initialises and fits the Isolation Forest model
    iso = IsolationForest(contamination=contamination, random_state=8030)
    df_clean['anomaly'] = iso.fit_predict(X_scaled)
    df_clean['anomaly_score'] = iso.decision_function(X_scaled)
    
    return df_clean

def pivot_chip_data(df, stage_col, val_col):
    '''Pivots data for anomaly detection. 
    
    If a 'Channel' column exists, it uses both chip_id and Channel as the row index, 
    effectively treating each channel as an independent observation (doubling the dataset size).

    Args:
        df (pd.DataFrame): The dataframe to pivot.
        stage_col (str): The column containing stage labels.
        val_col (str): The column containing values to aggregate.

    Returns:
        pd.DataFrame: The pivoted dataframe.
    '''

    # Checks for the presence of the 'Channel' column to determine the pivot index
    if 'Channel' in df.columns:
        return df.pivot_table(index=['chip_id', 'Channel'], columns=stage_col, values=val_col, aggfunc='mean', observed=False)
    else:
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
        
        # Returns control to the calling code
        return

    # Checks if there is a sufficient baseline of normal chips
    if len(normals) < 2:

        print('Not enough normal chips to form a baseline for comparison.')
        
        # Returns control to the calling code
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

def create_wide_intra(intra_df, val_col):
    '''Pivots multiple intra-stage metrics and suffixes columns to prevent overlap.

    Args:
        intra_df (pd.DataFrame): The dataframe containing intra-stage features.
        val_col (str): The base value column used to derive metric names.

    Returns:
        pd.DataFrame: The pivoted wide-format dataframe.
    '''

    # Defines the metrics to extract based on the target column
    intra_metrics = [f'{val_col}_std', f'{val_col}_spike_to_noise_ratio', f'{val_col}_max_residual_zscore']

    wide_list = []
    
    # Loops through each required metric
    for metric in intra_metrics:

        # Processes the metric if it exists in the dataframe
        if metric in intra_df.columns:

            wide_metric = pivot_chip_data(intra_df, 'stage', metric)
            metric_suffix = metric.replace(f'{val_col}_', '')
            wide_metric.columns = [f'{col}_{metric_suffix}' for col in wide_metric.columns]
            wide_list.append(wide_metric)
            
    # Concatenates and returns the wide list if it contains data
    return pd.concat(wide_list, axis=1) if wide_list else pd.DataFrame()

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

def plot_pel_layer_shifts(pel_changes, channels_dict=None):
    '''Plots PEL signal shifts between layers for each requested channel.

    Args:
        pel_changes (pd.DataFrame): PEL stage-transition measurements.
        channels_dict (dict | None, optional): Display-name to signal-column mapping. Defaults to None.
    '''

    # Initialises the default channel dictionary if none is provided
    if channels_dict is None:
        channels_dict = {
            'Channel 1': 'quad_ch1_change',
            'Channel 2': 'quad_ch2_change'
        }
    
    # Creates a copy of the dataframe to preserve the original data, filtering out the total shift
    df_plot = pel_changes[pel_changes['stage'] != 'Total_Shift_Initial_to_Final'].copy()

    # Loops through each key-value pair in the channel dictionary
    for ch_name, ch_col in channels_dict.items():

        # Displays the results in a collapsible output section
        with collapsible_output(f'PEL Layer Shifts: {ch_name}'):
            
            # Plots an interactive view of signal shifts across PEL layer transitions for each chip
            fig = go.Figure()
            
            # Loops through each grouped chip ID and its associated dataframe
            for chip_id, df_chip in df_plot.groupby('chip_id'):

                fig.add_trace(go.Scatter(
                    x=df_chip['stage'], 
                    y=df_chip[ch_col], 
                    mode='lines+markers', 
                    name=str(chip_id), 
                    opacity=0.6, 
                    marker=dict(size=6), 
                    visible='legendonly'
                ))

            fig.update_layout(
                title=f'PEL Layer Build-up: Shift per Transition ({ch_name})', 
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
            for chip_id, df_chip in df_plot.groupby('chip_id'):
                plt.plot(df_chip['stage'], df_chip[ch_col], marker='o', alpha=0.5, label=chip_id)

            plt.title(f'PEL Layer Build-up: Shift per Transition ({ch_name})')
            plt.xlabel('Layer Transition')
            plt.ylabel('Signal Shift (Δ)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

            # Adds a legend if the number of unique chips is manageable
            if df_plot['chip_id'].nunique() <= 25:
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

def analyse_cross_stage_split(pel_df, immob_df, split_name, pel_outliers=None, immob_outliers=None, sc_metrics_df=None, title='Cross-Stage Validation'):
    '''Merges PEL and a specific immobilisation split, runs PCA and clustering, plots the results with anomaly highlights for Channel 1, Channel 2, and overall. Validates against standard curves if provided.

    Args:
        pel_df (pd.DataFrame): The wide-format PEL dataframe.
        immob_df (pd.DataFrame): The wide-format immobilisation dataframe.
        split_name (str): The label for the current data split.
        pel_outliers (list, optional): List of outlier chips from PEL analysis. Defaults to None.
        immob_outliers (list, optional): List of outlier chips from immobilisation analysis. Defaults to None.
        sc_metrics_df (pd.DataFrame, optional): Standard curve metrics for validation. Defaults to None.
        title (str, optional): The title for the summary output. Defaults to 'Cross-Stage Validation'.
    '''
    
    # Creates a copy of the PEL dataframe to preserve the original data
    pel_copy = pel_df.copy()
    # Creates a copy of the immobilisation dataframe to preserve the original data
    immob_copy = immob_df.copy()
    
    # Ensures column names are treated as strings
    pel_copy.columns = pel_copy.columns.astype(str)
    immob_copy.columns = immob_copy.columns.astype(str)

    # Defines a mapping to normalise channel names across datasets
    channel_mapping = {
        'quad_ch1': 'Ch1', 'channel1': 'Ch1',
        'quad_ch2': 'Ch2', 'channel2': 'Ch2',
        'quad_ch1_change': 'Ch1', 'channel1_change': 'Ch1',
        'quad_ch2_change': 'Ch2', 'channel2_change': 'Ch2'
    }

    # Renames MultiIndex levels if they contain a Channel index
    if isinstance(pel_copy.index, pd.MultiIndex) and 'Channel' in pel_copy.index.names:
        pel_copy = pel_copy.rename(index=channel_mapping, level='Channel')
        
    if isinstance(immob_copy.index, pd.MultiIndex) and 'Channel' in immob_copy.index.names:
        immob_copy = immob_copy.rename(index=channel_mapping, level='Channel')

    # Merges the PEL and immobilisation datasets
    merged = pd.merge(pel_copy, immob_copy, left_index=True, right_index=True, how='inner', suffixes=('_PEL', '_Immob')).dropna()
    
    # Checks whether the merged dataframe contains data
    if merged.empty:
        print(f'[{split_name}] Not enough overlapping chips between PEL and Immobilisation for cross-analysis.')
        
        # Returns control to the calling code
        return
        
    # Combines outliers from both stages into a single set
    both_anomalies = set(pel_outliers if pel_outliers else []).union(set(immob_outliers if immob_outliers else []))

    def analyse_subset(subset_df, label):
        '''Runs PCA, clustering, and validation for one channel subset.

        Args:
            subset_df (pd.DataFrame): Merged features for the selected subset.
            label (str): Human-readable subset label.
        '''

        # Checks whether the subset dataframe contains data
        if subset_df.empty:
            
            # Returns control to the calling code
            return
            
        # Displays the results in a collapsible output section
        with collapsible_output(f'{title} - {label}'):

            print(f'[{split_name} - {label}] Overlapping chips analysed: {len(subset_df)}')
            
            # Standardises the features
            features = subset_df.columns.tolist()
            X_subset = StandardScaler().fit_transform(subset_df[features])
            
            # Initialises and applies PCA for dimensionality reduction
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X_subset)
            
            # Creates a copy of the dataframe to plot to preserve the original data
            df_to_plot = subset_df.copy()
            df_to_plot['PCA1'] = pca_result[:, 0]
            df_to_plot['PCA2'] = pca_result[:, 1]
            
            print(f'[{split_name} - {label}] Explained Variance by first 2 components: {pca.explained_variance_ratio_.sum()*100:.2f}%')
            
            # Initialises and fits KMeans clustering
            kmeans = KMeans(n_clusters=2, random_state=8030, n_init=10)
            df_to_plot['Cluster'] = kmeans.fit_predict(X_subset)
            
            # Plots the first two principal components to show cluster separation and flagged anomalies
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df_to_plot, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', s=100, alpha=0.7)
            
            # Extracts chip IDs based on the index type
            if isinstance(df_to_plot.index, pd.MultiIndex):
                subset_chip_ids = df_to_plot.index.get_level_values('chip_id')
            else:
                subset_chip_ids = df_to_plot.index
                
            # Identifies flagged anomalies within the current subset
            anomaly_mask = subset_chip_ids.isin(both_anomalies)
            
            # Highlights the identified anomalies on the scatter plot
            if anomaly_mask.any():
                plt.scatter(df_to_plot.loc[anomaly_mask, 'PCA1'], df_to_plot.loc[anomaly_mask, 'PCA2'], edgecolor='black', facecolor='none', s=200, label='Flagged Anomaly', linewidth=2)
                            
            plt.title(f'Cross-Stage PCA: {split_name} ({label})')
            plt.xlabel('Principal Component 1 (General Signal Variance)')
            plt.ylabel('Principal Component 2 (Stage-to-Stage Dynamic Variance)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
            print(f'\n Cluster Profiling [{split_name} - {label}]')
            
            # Calculates cluster means and overall statistics to build z-score profiles
            cluster_means = df_to_plot[features].groupby(df_to_plot['Cluster']).mean()
            overall_mean = df_to_plot[features].mean()
            overall_std = df_to_plot[features].std().replace(0, 1e-9)
            
            cluster_zscores = (cluster_means - overall_mean) / overall_std
            
            # Loops through each unique cluster ID
            for cluster_id in sorted(df_to_plot['Cluster'].unique()):

                print(f"\nCluster {cluster_id} Profile (n={sum(df_to_plot['Cluster'] == cluster_id)} observations):")
                
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

                print(f'\n--- Cluster Functional Yield (R² Comparison) [{split_name} - {label}] ---')
                
                cluster_mapping = df_to_plot[['Cluster']].reset_index()

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
                    
                    plt.title(f'Standard Curve R² Distribution by Manufacturing Cluster\n{split_name} ({label})')
                    plt.xlabel('Manufacturing Cluster')
                    plt.ylabel('Standard Curve R² Score')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.show()
                    
                # Creates lists to store the selected identifiers and anomaly results
                subset_unique_chips = set(subset_chip_ids)
                anomaly_ids = list(both_anomalies.intersection(subset_unique_chips))
                
                # Checks if there are any anomalous chips to validate
                if not anomaly_ids:
                    print(f'[{split_name} - {label}] No anomalous chips from the cross-stage analysis to validate against SC.')
                else:

                    print(f'\n[{split_name} - {label}] Validating functional performance for anomalous chip(s): {anomaly_ids}\n')
                    
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
                subset_unique_chips = set(subset_chip_ids)
                anomaly_ids = list(both_anomalies.intersection(subset_unique_chips))
                print(f'\n--- Anomaly Identifications [{split_name} - {label}] ---')

                # Prints the flagged anomalous chips or a pass message
                if anomaly_ids:
                    print(f'Flagged Anomalous Chips in cluster: {anomaly_ids}')
                else:
                    print('No anomalous chips detected in this cross-stage split.')

    # Triggers the subset analysis for Channel 1 if it exists
    if isinstance(merged.index, pd.MultiIndex) and 'Ch1' in merged.index.get_level_values('Channel'):
        analyse_subset(merged.xs('Ch1', level='Channel', drop_level=False), 'Channel 1')
        
    # Triggers the subset analysis for Channel 2 if it exists
    if isinstance(merged.index, pd.MultiIndex) and 'Ch2' in merged.index.get_level_values('Channel'):
        analyse_subset(merged.xs('Ch2', level='Channel', drop_level=False), 'Channel 2')
        
    # Triggers the pooled overall subset analysis
    analyse_subset(merged, 'Overall (Pooled)')

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
        
        # Returns control to the calling code
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

def cross_stage_correlations(pel_df, immob_df, split_name, sc_metrics_df=None, title='Cross-Stage Correlations'):
    '''Compares PEL and immobilisation features using cross-stage correlations.

    Args:
        pel_df (pd.DataFrame): Wide PEL feature table.
        immob_df (pd.DataFrame): Wide immobilisation feature table.
        split_name (str): Label identifying the immobilisation split.
        sc_metrics_df (pd.DataFrame | None, optional): Optional standard-curve metrics. Defaults to None.
        title (str, optional): Heading used for generated output. Defaults to 'Cross-Stage Correlations'.
    '''

    # Creates a copy of the PEL dataframe to preserve the original data
    pel_copy = pel_df.copy()
    
    # Creates a copy of the immobilisation dataframe to preserve the original data
    immob_copy = immob_df.copy()

    # Casts the columns to strings to ensure consistent merging
    pel_copy.columns = pel_copy.columns.astype(str)
    immob_copy.columns = immob_copy.columns.astype(str)

    # Defines a mapping to normalise channel names across datasets
    channel_mapping = {
        'quad_ch1': 'Ch1', 'channel1': 'Ch1',
        'quad_ch2': 'Ch2', 'channel2': 'Ch2',
        'quad_ch1_change': 'Ch1', 'channel1_change': 'Ch1',
        'quad_ch2_change': 'Ch2', 'channel2_change': 'Ch2'
    }
    
    # Renames MultiIndex levels if they contain a Channel index
    if isinstance(pel_copy.index, pd.MultiIndex) and 'Channel' in pel_copy.index.names:
        pel_copy = pel_copy.rename(index=channel_mapping, level='Channel')
        
    if isinstance(immob_copy.index, pd.MultiIndex) and 'Channel' in immob_copy.index.names:
        immob_copy = immob_copy.rename(index=channel_mapping, level='Channel')

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
        
        # Left joins the standard curve metrics onto the merged dataframe
        if isinstance(merged.index, pd.MultiIndex):
            merged = merged.merge(sc_first, left_on='chip_id', right_index=True, how='left')
        else:
            merged = merged.merge(sc_first, left_index=True, right_index=True, how='left')

    # Checks whether the merged dataframe contains data
    if merged.empty:

        print(f'[{split_name}] Not enough overlapping data points for cross-analysis.')

        # Returns control to the calling code
        return
    
    def analyse_subset(subset_df, label):
        '''Calculates and displays correlation results for one channel subset.

        Args:
            subset_df (pd.DataFrame): Merged features for the selected subset.
            label (str): Human-readable subset label.
        '''

        # Checks whether the subset dataframe contains data
        if subset_df.empty:
            
            # Returns control to the calling code
            return
            
        # Displays the results in a collapsible output section
        with collapsible_output(f'{title} - {label}'):

            print(f'[{split_name} - {label}] Overlapping data points analysed: {len(subset_df)}\n')

            corr_matrix = subset_df.corr()
            print_correlation_insights(corr_matrix, split_name, label, top_n=3)

            # Plots the cross-stage correlation matrix to show relationships between all selected features
            plt.figure(figsize=(20, 20))
            sns.heatmap(corr_matrix, cmap='plasma', center=0, annot=False)
            plt.title(f'{split_name} ({label}) - Overall Correlation')
            plt.tight_layout()
            plt.show()

    # Triggers the subset analysis for Channel 1 if it exists
    if isinstance(merged.index, pd.MultiIndex) and 'Ch1' in merged.index.get_level_values('Channel'):
        analyse_subset(merged.xs('Ch1', level='Channel', drop_level=False), 'Channel 1')
        
    # Triggers the subset analysis for Channel 2 if it exists
    if isinstance(merged.index, pd.MultiIndex) and 'Ch2' in merged.index.get_level_values('Channel'):
        analyse_subset(merged.xs('Ch2', level='Channel', drop_level=False), 'Channel 2')
        
    # Triggers the pooled overall subset analysis
    analyse_subset(merged, 'Overall (Pooled)')

def create_interactive_dashboard(data_catalog):
    '''Creates an interactive explorer for comparing data sources and features.

    Args:
        data_catalog (dict): Nested mapping of analysis modes to named dataframes.
    '''

    # Initialises widget controls for the dashboard
    mode_toggle = widgets.ToggleButtons(options=['Absolute', 'Stage Delta', 'Intra-stage Kinetics'], style={'description_width': 'initial'})
    channel_dropdown = widgets.Dropdown(options=['Channel 1', 'Channel 2', 'Both'], value='Channel 1', description='Channel:')
    sources = list(data_catalog['Absolute'].keys())

    x_source_drop = widgets.Dropdown(options=sources, value=sources[0], description='X Source:')
    y_source_drop = widgets.Dropdown(options=sources, value=sources[0], description='Y Source:')

    x_col_drop = widgets.Dropdown(description='X Metric:')
    y_col_drop = widgets.Dropdown(description='Y Metric:')
    
    out = widgets.Output()
    
    def update_dropdowns(*args):
        '''Refreshes feature dropdown options after the selected sources change.'''

        mode = mode_toggle.value
        x_src, y_src = x_source_drop.value, y_source_drop.value
        
        x_cols = list(data_catalog[mode][x_src].columns)
        y_cols = list(data_catalog[mode][y_src].columns)
        
        # Updates the X metric dropdown options
        x_col_drop.options = x_cols

        # Resets the X metric value if the current selection is invalid
        if x_col_drop.value not in x_cols: 

            x_col_drop.value = x_cols[0] if x_cols else None
            
        # Updates the Y metric dropdown options
        y_col_drop.options = y_cols
        
        # Resets the Y metric value if the current selection is invalid
        if y_col_drop.value not in y_cols: 

            y_col_drop.value = y_cols[0] if y_cols else None
            
    # Binds observers to trigger dropdown updates on value changes
    mode_toggle.observe(update_dropdowns, 'value')
    x_source_drop.observe(update_dropdowns, 'value')
    y_source_drop.observe(update_dropdowns, 'value')
    update_dropdowns()
    
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

        # Creates a copy of the X dataframe to preserve the original data
        df_x = data_catalog[mode][x_src].copy()
        
        # Creates a copy of the Y dataframe to preserve the original data
        df_y = data_catalog[mode][y_src].copy()
        
        def slice_channel(df, ch_label):
            '''Returns rows for a selected channel when the table uses a MultiIndex.
            
            Args:
                df (pd.DataFrame): The dataframe to slice.
                ch_label (str): The label of the channel to isolate.
                
            Returns:
                pd.DataFrame: The sliced dataframe.
            '''

            target_ch = 'Ch1' if ch_label == 'Channel 1' else 'Ch2'

            # Performs the slice if the index is a MultiIndex containing 'Channel'
            if isinstance(df.index, pd.MultiIndex) and 'Channel' in df.index.names:
                
                # Defines a standard mapping for channel names
                ch_map = {
                    'quad_ch1': 'Ch1', 
                    'channel1': 'Ch1', 
                    'quad_ch2': 'Ch2', 
                    'channel2': 'Ch2',
                    'quad_ch1_change': 'Ch1', 
                    'channel1_change': 'Ch1',
                    'quad_ch2_change': 'Ch2', 
                    'channel2_change': 'Ch2'
                }

                # Renames the Channel level to ensure consistent lookups
                df = df.rename(index=ch_map, level='Channel')
                
                # Returns the isolated target channel if it exists
                if target_ch in df.index.get_level_values('Channel'):
                    return df.xs(target_ch, level='Channel')
                else:
                    return pd.DataFrame()
                
            return df

        # Slices both dataframes using the requested channel label
        df_x = slice_channel(df_x, channel_label)
        df_y = slice_channel(df_y, channel_label)
        
        # Checks whether either dataframe is empty
        if df_x.empty or df_y.empty:
            return pd.DataFrame()

        # Ensures all columns are strings for safe merging
        df_x.columns = df_x.columns.astype(str)
        df_y.columns = df_y.columns.astype(str)
        x_col_str, y_col_str = str(x_col), str(y_col)
        
        # Inner joins the dataframes on their index
        df_merged = pd.merge(df_x[[x_col_str]], df_y[[y_col_str]], left_index=True, right_index=True, how='inner', suffixes=('_x', '_y'))
        
        # Handles column renaming if X and Y columns share the same name
        actual_x_col = x_col_str + '_x' if x_col_str == y_col_str else x_col_str
        actual_y_col = y_col_str + '_y' if x_col_str == y_col_str else y_col_str
        df_merged = df_merged.rename(columns={actual_x_col: 'x_val', actual_y_col: 'y_val'})

        return df_merged.replace([np.inf, -np.inf], np.nan).dropna()

    def plot_data(*args):
        '''Creates the dashboard scatter plot using the current widget selections.'''

        # Opens the resource safely for processing
        with out:

            # Clears the previous output before rendering the new plot
            out.clear_output(wait=True)

            mode, channel = mode_toggle.value, channel_dropdown.value
            x_src, y_src = x_source_drop.value, y_source_drop.value
            x_col, y_col = x_col_drop.value, y_col_drop.value
            
            # Displays an error if the selected columns are invalid
            if not x_col or not y_col:

                print('Please select valid metrics to plot.')

                # Returns control to the calling code
                return
                
            # Creates a new Plotly figure for the visualisation
            fig = go.Figure()
            
            # Sets the active channels based on the dropdown selection
            channels_to_plot = ['Channel 1', 'Channel 2'] if channel == 'Both' else [channel]
            colors = {'Channel 1': '#1f77b4', 'Channel 2': '#ff1e0e'} 
            plotted_any = False
            
            # Loops through each channel to plot its data
            for ch in channels_to_plot:

                # Safely attempts to join and extract the necessary data
                try:
                    plot_df = get_joined_data(mode, x_src, y_src, x_col, y_col, ch)
                except Exception as e:
                    continue
                    
                # Skips the channel if there are insufficient points for plotting
                if len(plot_df) < 2: 
                    continue

                plotted_any = True
                x_data, y_data = plot_df['x_val'], plot_df['y_val']
                
                # Adds a scatter trace for the data points
                fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name=f'{ch}', marker=dict(size=8, opacity=0.7, color=colors[ch], line=dict(width=1, color='DarkSlateGrey')), text=plot_df.index, hovertemplate='Chip ID: %{text}<br>X: %{x:.4f}<br>Y: %{y:.4f}<extra></extra>'))
                
                # Calculates and plots a linear regression fit if variance exists
                if x_data.nunique() > 1:

                    slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
                    x_fit = np.linspace(x_data.min(), x_data.max(), 100)
                    y_fit = slope * x_fit + intercept
                    
                    fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name=f'{ch} Fit (r={r_value:.3f}, R^2={r_value**2:.3f})', line=dict(color=colors[ch], dash='dash', width=2), hoverinfo='skip'))            
            
            # Reports an error if no valid data was found for any channel
            if not plotted_any:

                print('Not enough matching chip records to plot these selections.')

                # Returns control to the calling code
                return
                
            # Formats and displays the final interactive dashboard
            title = f'{y_src} [{y_col}] vs {x_src} [{x_col}]'
            fig.update_layout(title=title, xaxis_title=f'{x_src} : {x_col}', yaxis_title=f'{y_src} : {y_col}', template='plotly_white', height=600, margin=dict(l=40, r=40, t=60, b=40), hovermode='closest')
            fig.show()

    # Binds observers to trigger plotting on widget changes
    mode_toggle.observe(plot_data, 'value')
    channel_dropdown.observe(plot_data, 'value')
    x_source_drop.observe(plot_data, 'value')
    y_source_drop.observe(plot_data, 'value')
    x_col_drop.observe(plot_data, 'value')
    y_col_drop.observe(plot_data, 'value')
    
    # Organises the controls vertically
    controls = widgets.VBox([mode_toggle, channel_dropdown, widgets.HBox([x_source_drop, x_col_drop]), widgets.HBox([y_source_drop, y_col_drop])])
    
    # Displays the dashboard elements
    display(widgets.HTML(f"<h3 style='margin-bottom:0px; color:#0000FF;'>Master Chip Comparison Dashboard</h3>"))
    display(controls, out)

    # Invokes the initial plot
    plot_data()

def get_top_drivers(df, chip_id, top_n=3):
    '''Identifies the features that most strongly distinguish a chip from its peers.

    Args:
        df (pd.DataFrame): Feature table indexed by chip identifier.
        chip_id (str): Identifier of the chip to investigate.
        top_n (int, optional): Number of highest-impact features to return. Defaults to 3.

    Returns:
        list[str]: Ranked absolute z-scores formatted as strings for the selected chip.
    '''

    mean = df.mean()
    std = df.std().replace(0, 1e-9)
    
    # Evaluates the drivers based on the type of index structure
    if isinstance(df.index, pd.MultiIndex):

        if chip_id not in df.index.get_level_values('chip_id'): 
            return ['N/A (Chip not in dataset)']
        
        chip_rows = df.xs(chip_id, level='chip_id')
        drivers = []

        # Loops through each row in the dataframe for the targeted chip
        for ch, row in chip_rows.iterrows():

            z_scores = ((row - mean) / std).abs().dropna()

            # Checks whether the z-scores series contains data
            if not z_scores.empty:

                top_stages = z_scores.sort_values(ascending=False).head(top_n)
                
                # Formats the top driving stages for the current channel
                top_str = ' | '.join([f'{stage} (Z: {z:.1f})' for stage, z in top_stages.items()])
                drivers.append(f'[{ch}] {top_str}')

        return drivers
    else:

        # Skips evaluation if the chip ID is not found in the standard index
        if chip_id not in df.index: 
            return ['N/A (Chip not in dataset)']
        
        z_scores = ((df.loc[chip_id] - mean) / std).abs().dropna()

        # Checks whether the z-scores series contains data
        if z_scores.empty: 
            return ['N/A']
        
        top_stages = z_scores.sort_values(ascending=False).head(top_n)

        # Returns the formatted top drivers
        return [f'{stage} (Z: {z:.1f})' for stage, z in top_stages.items()]

def generate_at_risk_summary(master_df, sc_metrics, sc_raw_df, all_at_risk_chips, section_config, title='Overall At-Risk Chips Summary'):
    '''Clusters chips based on combined stage data, cross-references against standard curve metrics, and generates a structured summary report across 5 distinct sections.

    Args:
        master_df (pd.DataFrame): The master dataframe containing all features.
        sc_metrics (pd.DataFrame): Standard curve metrics.
        sc_raw_df (pd.DataFrame): Raw standard curve data.
        all_at_risk_chips (list): List of all chips identified as at-risk.
        section_config (dict): Configuration mapping for different summary sections.
        title (str, optional): The title for the summary. Defaults to 'Overall At-Risk Chips Summary'.
    '''

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
        else:
            print('No faults detected across any stage. Everything looks normal!')

    # Displays the results in a collapsible output section
    with collapsible_output(f'{title} - PEL Breakdown'):

        print(f'Total Unique PEL Faulty Chips: {len(pel_unique)}')

        # Expands the individual sections triggered within PEL
        if pel_unique:

            # Loops through each section name and outlier list in the PEL breakdown
            for sec_name, out_list in pel_breakdown.items():
                print(f"  - {sec_name}: {len(out_list)} chips ({', '.join(map(str, out_list))})")

    # Displays the results in a collapsible output section
    with collapsible_output(f'{title} - Immobilisation Breakdown'):

        print(f'Total Unique Immobilisation Faulty Chips: {len(immob_unique)}')

        # Expands the individual sections triggered within Immobilisation
        if immob_unique:

            # Loops through each section name and outlier list in the Immobilisation breakdown
            for sec_name, out_list in immob_breakdown.items():
                print(f"  - {sec_name}: {len(out_list)} chips ({', '.join(map(str, out_list))})")

    # Displays the results in a collapsible output section
    with collapsible_output(f'{title} - Standard Curve Breakdown'):

        print(f'Total Unique Standard Curve Faulty Chips: {len(sc_fails)}')

        if sc_fails: 
            print(f"Chip IDs (R^2 < 0.95): {', '.join(map(str, sorted(list(sc_fails))))}")

    # Displays the results in a collapsible output section
    with collapsible_output(f'{title} - Chip Profiles'):

        # Checks whether the master dataframe contains data or if there are combined fails
        if master_df.empty or not combined_all_fails:

            print('No detailed profiles available.')
            
            # Returns control to the calling code
            return

        features = master_df.columns.tolist()
        X_master = StandardScaler().fit_transform(master_df[features])
        
        # Re-clusters the master dataset for contextual grouping
        kmeans = KMeans(n_clusters=2, random_state=8030, n_init=10)
        master_df['Cluster'] = kmeans.fit_predict(X_master)

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
            if chip in master_df.index.get_level_values('chip_id'):

                chip_clusters = master_df.xs(chip, level='chip_id')['Cluster'].to_dict()
                cluster_profile = ' | '.join([f'{ch}: Cluster {cl}' for ch, cl in chip_clusters.items()])
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
    '''Plots fitted association and dissociation curves for one experiment.

    Args:
        data (dict): Kinetic measurements and fitted-model values.
        title (str): Plot title.
    '''

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
            
        # Adds the raw data trace to the Plotly figure
        fig_plotly.add_trace(go.Scatter(x=t_raw, y=y_raw, mode='lines', line=dict(color=hex_colour, width=1.5), name=f['meas_id']))
        
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

            fig_plotly.add_trace(go.Scatter(x=t_fit - t0, y=y_fit, mode='lines', line=dict(color='black', dash='dash', width=1.5), showlegend=False, hoverinfo='skip'))

            ax.plot(t_fit - t0, y_fit, '--', color='black', lw=1.2)

        # Calculates and plots the fitted dissociation model line if successfully derived
        if f['dissoc_fit'] is not None:

            R0d, kd, Rinf, RI_d = f['dissoc_fit']
            t_fit_d = dissoc_zone['time'].values
            y_fit_d = dissoc_model(t_fit_d, R0d, kd, Rinf, RI_d)

            fig_plotly.add_trace(go.Scatter(x=t_fit_d - t0, y=y_fit_d, mode='lines', line=dict(color='black', dash='dash', width=1.5), showlegend=False, hoverinfo='skip'))

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

def plot_kobs_vs_conc(data, title):
    '''Plots observed association rates against concentration.

    Args:
        data (dict): Kinetic results containing concentration and kobs.
        title (str): Plot title.
    '''

    # Extracts concentrations and observed association rates for successful fits
    concs = [f['conc'] for f in data['fits'] if f['assoc_fit'] is not None]
    kobs_vals = [f['assoc_fit'][1] for f in data['fits'] if f['assoc_fit'] is not None]

    # Initialises the Matplotlib figure and plots the scatter points
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(concs, kobs_vals, color='tab:blue')

    # Generates values for the linear fit line
    x_fit = np.linspace(0, max(concs), 50)
    y_fit = data['ka'] * x_fit + data['kd']

    # Plots the linear fit line and adds the equation to the legend
    ax.plot(x_fit, y_fit, '--', color='black', label=f"ka={data['ka']:.2e}, kd={data['kd']:.2e}")

    # Formats and displays the plot
    ax.set_xlabel('Concentration (ug/ml)')
    ax.set_ylabel('k_obs (s$^{-1}$)')
    ax.set_title(title)
    ax.legend()
    plt.show()

def plot__binding_curves(data, title, pre_baseline_seconds=10, tail_avg_seconds=5):
    '''Plots baseline-aligned binding curves using configurable sampling windows.

    Args:
        data (dict): Sensorgram measurements to align and plot.
        title (str): Plot title.
        pre_baseline_seconds (float, optional): Duration of the pre-event baseline window. Defaults to 10.
        tail_avg_seconds (float, optional): Duration of the tail-averaging window. Defaults to 5.
    '''

    # Extracts the raw sensorgram data
    sensor = data['sensor']
    
    # Initialises the Matplotlib figure
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

def run_binding_kinetics_analysis(files):
    '''Computes binding kinetics (association/dissociation, global ka/kd/KD).
    
    Correctly treats every flag as a new injection, automatically finding the 
    peak to split the window into Association and Dissociation phases.

    Args:
        files (dict): Processed sensorgram and flag data files.

    Returns:
        dict: The updated kinetics results.
    '''

    # Initialises an empty dictionary to store the kinetic analysis results
    kinetics_results = {}

    # Loops through each folder in the files dictionary
    for folder in files.keys():

        # Skips folders containing baseline flags
        if 'baseline' in folder:
            continue
            
        # Initialises an empty dictionary for the current folder
        kinetics_results[folder] = {}
        previous_file = ''
        
        # Loops through each file in the folder
        for file in list(files[folder].keys()):

            # Processes only sensorgram files
            if 'sensorgram' in file:

                # Extracts and sorts the sensorgram and flag data chronologically
                sensor = files[folder][file].sort_values('time').reset_index(drop=True)
                flags = files[folder][previous_file].sort_values('time').reset_index(drop=True)
                sensor['response'] = sensor['channel1']

                parsed = []

                # Loops through each row in the flags dataframe to extract concentrations
                for _, row in flags.iterrows():

                    conc_val = parse_conc_flag(row['conc'])

                    # Appends valid concentration events to the parsed list
                    if conc_val is not None:
                        parsed.append({'time': row['time'], 'conc': conc_val, 'label': str(row['conc'])})

                segments = []
                median_time_step = sensor['time'].diff().median()
                lookahead_rows = max(1, int(5 / median_time_step))
                
                # Loops through each parsed event to delineate association and dissociation segments
                for i, event in enumerate(parsed):

                    t_start = event['time']
                    t_next = parsed[i+1]['time'] if i + 1 < len(parsed) else sensor['time'].max()
                    
                    # Isolates the sensorgram zone for the current event
                    zone = sensor[(sensor['time'] >= t_start) & (sensor['time'] <= t_next)]

                    # Skips the segment if there are insufficient data points
                    if len(zone) < lookahead_rows + 5: 
                        continue
                        
                    # Calculates the drop size to identify the peak
                    drop_size = zone['response'] - zone['response'].shift(-lookahead_rows)

                    # Falls back to an immediate shift if the initial drop size is empty
                    if drop_size.dropna().empty:
                        drop_size = zone['response'] - zone['response'].shift(-1)
                        
                    # Identifies the index of the maximum drop size
                    peak_idx = drop_size.idxmax()
                    
                    # Defaults to the absolute maximum peak if the delta check yields NaN
                    if pd.isna(peak_idx):
                        peak_idx = zone['response'].idxmax()
                        
                    t_peak = sensor.loc[peak_idx, 'time']
                    
                    # Isolates the candidate dissociation zone
                    dissoc_candidate = sensor[(sensor['time'] > t_peak) & (sensor['time'] <= t_next)]

                    # Processes the dissociation zone if it contains data
                    if not dissoc_candidate.empty:
                        
                        # Caps the dissociation window at 60 seconds
                        t_cap = t_peak + 60.0
                        dissoc_candidate = dissoc_candidate[dissoc_candidate['time'] <= t_cap]
                        
                        # Pinpoints a secondary dissociation event using differential gradients
                        diffs = dissoc_candidate['response'].diff()
                        spike_idx = diffs[diffs > 0.5].first_valid_index()
                        
                        # Sets the dissociation end time based on the detected spike or maximum time
                        if spike_idx is not None:
                            t_dissoc_end = sensor.loc[spike_idx, 'time'] - 2.0
                        else:
                            t_dissoc_end = dissoc_candidate['time'].max()
                    else:
                        t_dissoc_end = t_next
                    
                    # Appends the calculated segment boundaries
                    segments.append({
                        'conc': event['conc'], 
                        'meas_id': event['label'], 
                        't_assoc_start': t_start,
                        't_assoc_end': t_peak,
                        't_dissoc_start': t_peak,
                        't_dissoc_end': t_dissoc_end
                    })

                fits = []

                # Loops through each evaluated segment to perform curve fitting
                for seg in segments:

                    # Extracts the specific zones for association and dissociation
                    assoc_zone = sensor[(sensor['time'] >= seg['t_assoc_start']) & (sensor['time'] <= seg['t_assoc_end'])]
                    dissoc_zone = sensor[(sensor['time'] >= seg['t_dissoc_start']) & (sensor['time'] <= seg['t_dissoc_end'])]

                    # Attempts to fit the models if there are sufficient data points
                    assoc_fit = fit_association(assoc_zone['time'].values, assoc_zone['response'].values) if len(assoc_zone) > 5 else None
                    dissoc_fit = fit_dissociation(dissoc_zone['time'].values, dissoc_zone['response'].values) if len(dissoc_zone) > 5 else None

                    # Appends the fit results and zone data
                    fits.append({
                        'conc': seg['conc'], 'meas_id': seg['meas_id'],
                        'assoc_fit': assoc_fit, 'dissoc_fit': dissoc_fit,
                        'assoc_zone': assoc_zone, 'dissoc_zone': dissoc_zone
                    })

                # Compiles the final file data structure
                file_data = {'sensor': sensor, 'segments': segments, 'fits': fits}
                concs = [f['conc'] for f in fits if f['assoc_fit'] is not None]
                kobs_vals = [f['assoc_fit'][1] for f in fits if f['assoc_fit'] is not None]

                # Attempts a linear regression if multiple valid concentrations exist
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

                    # Determines the overall dissociation rate from individual fits if available
                    if dissoc_kds:
                        file_data['kd_dissoc_mean'] = np.mean(dissoc_kds)

                # Assigns the compiled data to the kinetics results dictionary
                kinetics_results[folder][file] = file_data
            
            # Updates the previous file tracker
            previous_file = file


    # Loops through each folder and its assigned files in the kinetics results
    for folder, folder_files in kinetics_results.items():

        # Prints the section header
        print(f"\n{'='*40}\nKinetics Analysis: {folder}\n{'='*40}")
        
        summary_rows = []

        # Loops through each file and its data content
        for file, data in folder_files.items():

            # Appends valid kinetic summaries
            if 'ka' in data:

                summary_rows.append({
                    'File': file, 'ka (ug/ml*s)^-1': data['ka'], 'kd_intercept (s^-1)': data['kd'],
                    'kd_dissoc_mean (s^-1)': data.get('kd_dissoc_mean', np.nan),
                    'KD (ug/ml)': data['KD'], 'kobs_fit_R2': data['kobs_r2']
                })
        
        # Displays the results in a collapsible output section if data exists
        if summary_rows:

            with collapsible_output(f'Global Kinetics Summary: {folder}'):
                display(pd.DataFrame(summary_rows))

        # Loops through each file to generate per-concentration details and plots
        for file, data in folder_files.items():

            # Skips files without valid fits
            if 'fits' not in data: 
                continue
            
            # Extracts the chip and measurement identifiers from the filename
            clean_filename = file.replace('Copy of ', '').strip()
            parts = clean_filename.split('_')
            chip_id = parts[0] if len(parts) > 0 else 'Unknown'
            measurement_id = parts[1] if len(parts) > 1 else 'Unknown'
            
            print(f'\n--- Chip ID: {chip_id} | Measurement ID: {measurement_id} ---')

            per_conc_rows = []

            # Loops through each generated curve fit
            for f in data['fits']:

                warning = ''
                
                # Unpacks association parameters
                Req, kobs, R0 = (f['assoc_fit'][0], f['assoc_fit'][1], f['assoc_fit'][2]) if f['assoc_fit'] is not None else (np.nan, np.nan, np.nan)

                # Appends a warning if the association fit failed
                if f['assoc_fit'] is None: 
                    warning += 'Assoc fit failed; '
                    
                # Unpacks dissociation parameters
                R0d, kd, Rinf = (f['dissoc_fit'][0], f['dissoc_fit'][1], f['dissoc_fit'][2]) if f['dissoc_fit'] is not None else (np.nan, np.nan, np.nan)

                # Appends a warning if the dissociation fit failed
                if f['dissoc_fit'] is None: 
                    warning += 'Dissoc fit failed; '
                    
                # Extracts standard rates if parameters are viable
                if not np.isnan(kobs) and not np.isnan(kd) and f['conc'] > 0:

                    ka_indiv = (kobs - kd) / f['conc']
                    KD_indiv = kd / ka_indiv if ka_indiv != 0 else np.nan

                    # Appends a warning if a negative ka is inferred
                    if kd >= kobs: 
                        warning += 'kd >= kobs (Negative k_a); '
                else:
                    ka_indiv, KD_indiv = np.nan, np.nan
                    
                # Appends the concentration row metrics
                per_conc_rows.append({
                    'Measurement': f['meas_id'], 'Concentration (ug/ml)': f['conc'], 'k_obs (s^-1)': kobs,
                    'k_d (s^-1)': kd, 'k_a (calc)': ka_indiv, 'K_D': KD_indiv, 'R_eq': Req, 'Warning': warning.strip('; ')
                })
                
            # Displays the results in a collapsible output section if concentration rows exist
            if per_conc_rows:

                df_per_conc = pd.DataFrame(per_conc_rows).sort_values(by=['Concentration (ug/ml)', 'Measurement']).reset_index(drop=True)

                with collapsible_output(f'Per-Concentration Details: {file}'):
                    display(df_per_conc)

            # Displays the kinetic fit curves in a collapsible output section
            with collapsible_output(f'Kinetic Fit Curves: {file}'):
                plot_kinetic_curves(data, f'Kinetic Curves — {chip_id}')

            # Displays the observed rates against concentration in a collapsible output section
            if 'ka' in data:

                with collapsible_output(f'K_obs vs Concentration: {file}'):
                    plot_kobs_vs_conc(data, f'k_obs vs Concentration — {chip_id}')

            # Displays the aligned binding curves in a collapsible output section
            with collapsible_output(f'Aligned Binding Curves: {file}'):
                plot_binding_curves(data, f'Binding Curves Overlay — {chip_id}')
                
    # Returns the final updated dictionary
    return kinetics_results
