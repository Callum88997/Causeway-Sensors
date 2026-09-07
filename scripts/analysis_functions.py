# Imports required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

def detect_anomalies(df, feature_cols, contamination='auto'):
    '''Applies an Isolation Forest to detect multivariate outliers robustly.
    
    Evaluates each inferred stage independently.
    
    Args:
        df (pd.DataFrame): The input dataframe containing the dataset.
        feature_cols (list[str]): The list of feature columns to evaluate.
        contamination (str | float, optional): The expected proportion of outliers. Defaults to 'auto'.
        
    Returns:
        tuple: (annotated_df, stage_flags_df) where stage_flags_df contains 
               a 1 if the chip failed that specific stage, else 0.
    '''

    # Creates a copy of the dataframe to prevent modification of the original data
    df_clean = df.copy()

    # Initialises a series to track the worst anomaly scores with infinity
    worst_scores = pd.Series(index=df_clean.index, data=np.inf)

    # Initialises a series to track the anomaly status defaulting to 1 (normal)
    is_anomaly = pd.Series(index=df_clean.index, data=1)

    # Initialises an empty dictionary to dynamically group columns by stage
    stage_groups = {}

    # Loops through each column in the feature columns list
    for col in feature_cols:

        # Extracts the stage name by splitting the column string from the right
        stage = str(col).rsplit('_', 1)[0]

        # Adds the stage to the dictionary if it has not been encountered yet
        if stage not in stage_groups:
            
            # Initialises an empty list for the new stage
            stage_groups[stage] = []

        # Appends the current column to its corresponding stage group
        stage_groups[stage].append(col)

    # Initialises a tracker matrix to hold stage-level failure flags
    stage_flags = pd.DataFrame(0, index=df_clean.index, columns=list(stage_groups.keys()))

    # Loops through each stage and its associated columns in the grouped dictionary
    for stage, stage_cols in stage_groups.items():
            
        # Filters the dataframe for the specific stage columns and drops missing values
        df_stage = df_clean[stage_cols].dropna()
        
        # Skips the stage if the resulting dataframe is empty or lacks sufficient data points
        if df_stage.empty or len(df_stage) < 2:
            continue

        # Initialises the robust scaler for feature normalisation
        scaler = RobustScaler()

        # Fits and transforms the stage data using the scaler
        X_scaled = scaler.fit_transform(df_stage)
        
        # Initialises the isolation forest model with defined parameters
        iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=8030, n_jobs=-1)

        # Fits the model and predicts the anomaly labels for the scaled data
        preds = iso.fit_predict(X_scaled)

        # Calculates the anomaly decision function scores for the scaled data
        scores = iso.decision_function(X_scaled)

        # Converts the generated scores into a pandas series aligned with the dataframe index
        current_scores = pd.Series(scores, index=df_stage.index)

        # Converts the predicted labels into a pandas series aligned with the dataframe index
        current_preds = pd.Series(preds, index=df_stage.index)
        
        # Updates the worst scores by taking the minimum between existing and current scores
        worst_scores.loc[df_stage.index] = np.minimum(worst_scores.loc[df_stage.index], current_scores)

        # Updates the global anomaly flag by taking the minimum between existing and current predictions
        is_anomaly.loc[df_stage.index] = np.minimum(is_anomaly.loc[df_stage.index], current_preds)
        
        # Identifies the indices of the chips that failed the current stage
        failed_chips = df_stage.index[current_preds == -1]

        # Flags the failed chips with a 1 in the corresponding stage column of the tracker matrix
        stage_flags.loc[failed_chips, stage] = 1
        
    # Replaces any remaining infinity values in the worst scores series with 0.0
    worst_scores.replace(np.inf, 0.0, inplace=True)
    
    # Assigns the aggregated anomaly statuses as a new column in the clean dataframe
    df_clean['anomaly'] = is_anomaly

    # Assigns the aggregated anomaly scores as a new column in the clean dataframe
    df_clean['anomaly_score'] = worst_scores

    # Returns the annotated dataframe alongside the stage failure flags matrix
    return df_clean, stage_flags

def pivot_chip_data(df, stage_col, val_col):
    '''Pivots data for anomaly detection.

    Args:
        df (pd.DataFrame): The dataframe to pivot.
        stage_col (str): The column containing stage labels.
        val_col (str): The column containing values to aggregate.

    Returns:
        pd.DataFrame: The pivoted dataframe.
    '''

    # Returns the pivoted dataframe grouped by chip ID and stages, averaging the specified values
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

    # Extracts the upper triangle of the correlation matrix to avoid duplicate pairs and self-correlations
    corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).unstack().dropna()

    # Checks whether the extracted correlation series contains data
    if corr.empty:
        
        # Prints a missing data warning
        print('Not enough data to compute correlations.')
        
        return

    # Extracts and sorts the positive correlations in descending order
    positive = corr[corr > 0].sort_values(ascending=False)

    # Extracts and sorts the negative correlations in ascending order
    negative = corr[corr < 0].sort_values()

    # Extracts and sorts the weakest correlations based on their absolute values closest to zero
    weakest = corr.iloc[corr.abs().argsort()]

    # Prints the positive correlations header
    print('Top Positive Correlations:')

    # Checks if positive correlations exist
    if len(positive):

        # Loops through each key-value pair in the top positive correlations
        for (s1, s2), val in positive.head(num).items():
            
            # Prints the paired stages and their correlation value
            print(f'  - {s1} & {s2} (r = {val:.3f})')
    else:
        
        # Prints a null message if no positive correlations are found
        print('  None')

    # Prints the negative correlations header
    print('\nTop Negative (Inverse) Correlations:')

    # Checks if negative correlations exist
    if len(negative):

        # Loops through each key-value pair in the top negative correlations
        for (s1, s2), val in negative.head(num).items():
            
            # Prints the paired stages and their correlation value
            print(f'  - {s1} & {s2} (r = {val:.3f})')
    else:
        
        # Prints a null message if no negative correlations are found
        print('  None')

    # Prints the weakest relationships header
    print('\nWeakest Relationships (Closest to zero):')

    # Loops through each key-value pair in the top weakest correlations
    for (s1, s2), val in weakest.head(num).items():
        
        # Prints the paired stages and their correlation value
        print(f'  - {s1} & {s2} (r = {val:.3f})')

def analyse_anomaly_drivers(df, anomalies_df, label, num=3, use_ensemble=True, ensemble_chips=None):
    '''Compares anomalous chips to normal chips to identify which stages drive the anomaly.

    Args:
        df (pd.DataFrame): The original dataframe containing the stage data.
        anomalies_df (pd.DataFrame): The dataframe containing individual anomaly scores and labels.
        label (str): The prefix label for the summary output.
        num (int, optional): The number of top driving stages to display. Defaults to 3.
        use_ensemble (bool, optional): If True, uses ensemble_chips for analysis. Defaults to True.
        ensemble_chips (list, optional): List of ensemble failed chip IDs.
    '''

    # Determines which set of outliers and normals to use based on the toggle
    if use_ensemble and ensemble_chips is not None:
        outliers = df.index.intersection(ensemble_chips)
        normals = df.index.difference(ensemble_chips)
        mode_label = "Ensemble"
    else:
        outliers = anomalies_df[anomalies_df['anomaly'] == -1].index
        normals = anomalies_df[anomalies_df['anomaly'] == 1].index
        mode_label = "Isolated"

    # Prints the anomaly breakdown header formatted with the provided label and mode
    print(f'\n {label} Anomaly Stage Breakdown ({mode_label} Mode)')

    # Checks if there are no outliers detected in the dataset
    if len(outliers) == 0:
        print('No anomalies detected. All chips are behaving within normal bounds.')
        return

    # Checks if there is a sufficient baseline of normal chips for comparison
    if len(normals) < 2:
        print('Not enough normal chips to form a baseline for comparison.')
        return

    # Slices the source dataframe to isolate the normal data subset
    normal_data = df.loc[normals]

    # Slices the source dataframe to isolate the anomaly data subset
    anomaly_data = df.loc[outliers]

    # Calculates the statistical column means for the normal baseline
    normal_mean = normal_data.mean()

    # Calculates the standard deviations for the normal baseline, substituting zeros to prevent division errors
    normal_std = normal_data.std().replace(0, 1e-9) 

    # Calculates the statistical column means for the anomaly subset
    anomaly_mean = anomaly_data.mean()
    
    # Calculates the absolute z-scores to identify the largest deviations from the baseline
    z_scores = ((anomaly_mean - normal_mean) / normal_std).abs().sort_values(ascending=False).dropna()

    # Prints the header indicating the top driving stages
    print(f'Top {num} stages driving these anomalies (Largest deviation from normal):')

    # Loops through the top z-scores to output the specific anomaly drivers
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
        keep_metrics (list[str], optional): Suffixes of metrics to keep. Defaults to ['std', 'slope', 'range', 'skew'].

    Returns:
        pd.DataFrame: The pivoted wide-format dataframe.
    '''

    # Dynamically finds all columns that start with the target channel's name and match the requested metrics
    intra_metrics = [col for col in intra_df.columns if col.startswith(f'{val_col}_') and col.replace(f'{val_col}_', '') in keep_metrics and col not in ['chip_id', 'stage']]

    # Extracts the unique chronological stages from the source dataframe
    chronological_stages = intra_df['stage'].unique()

    # Initialises an empty list to store the pivoted metric dataframes
    wide_list = []
    
    # Loops through each required metric identified in the dataset
    for metric in intra_metrics:

        # Pivots the dataframe to transform the specific metric into a wide format
        wide_metric = pivot_chip_data(intra_df, 'stage', metric)
        
        # Extracts just the statistical suffix from the metric name
        metric_suffix = metric.replace(f'{val_col}_', '')
        
        # Appends the suffix to the wide column names to prevent naming collisions
        wide_metric.columns = [f'{col}_{metric_suffix}' for col in wide_metric.columns]

        # Appends the pivoted and renamed dataframe to the collection list
        wide_list.append(wide_metric)

    # Evaluates if the collected wide list is empty
    if not wide_list:

        # Returns an empty dataframe if no metrics were processed
        return pd.DataFrame()
        
    # Concatenates all collected wide dataframes side-by-side
    wide_df = pd.concat(wide_list, axis=1)

    # Initialises an empty list to enforce sequential column ordering
    ordered_columns = []

    # Loops through each extracted metric to build the chronological column names
    for metric in intra_metrics:

        # Isolates the metric suffix for column mapping
        metric_suffix = metric.replace(f'{val_col}_', '')

        # Loops through the sequential chronological stages
        for stage in chronological_stages:

            # Constructs the expected column name combining stage and suffix
            col_name = f'{stage}_{metric_suffix}'

            # Checks if the constructed column exists in the concatenated dataframe
            if col_name in wide_df.columns:

                # Appends the validated column name to the ordering list
                ordered_columns.append(col_name)
            
    # Filters and returns the wide dataframe with chronologically ordered columns
    return wide_df[ordered_columns]

def get_outlier_chips(anomalies_df):
    '''Extracts unique chip IDs from the anomaly dataframe, handling both standard and MultiIndex.

    Args:
        anomalies_df (pd.DataFrame): The dataframe containing anomaly labels.

    Returns:
        list: A list of unique outlier chip IDs.
    '''

    # Checks whether the anomalies dataframe contains data
    if anomalies_df.empty:

        # Returns an empty list if there is no data
        return []
    
    # Filters the dataframe to isolate rows flagged as outliers
    outliers = anomalies_df[anomalies_df['anomaly'] == -1]
    
    # Evaluates if the index is a MultiIndex structure
    if isinstance(outliers.index, pd.MultiIndex):

        # Extracts and returns unique chip IDs from the MultiIndex level
        return outliers.index.get_level_values('chip_id').unique().tolist()
    
    # Returns the standard index as a list
    return outliers.index.tolist()

def get_related_columns(query_col, abs_cols, change_cols, intra_cols):
    '''A unified method to find all related columns across Absolute, Change, and Intra datasets.
    
    Args:
        query_col (str): The column name you want to look up.
        abs_cols (list): List of all Absolute column names.
        change_cols (list): List of all Change/Delta column names.
        intra_cols (list): List of all Intra-stage column names.
        
    Returns:
        dict: A dictionary containing the detected type and its linked columns across all sets.
    '''
    
    # Casts the query column to a string and strips leading and trailing whitespace
    query_str = str(query_col).strip()
    
    # Initialises a dictionary to store the parsed relationships and detected types
    result = {
        'Query': query_str,
        'Detected_Type': None,
        'Absolute_Columns': [],
        'Change_Columns': [],
        'Intra_Columns': []
    }
    
    # Initialises a variable to track the identified base change column
    base_change_col = None

    # Initialises an empty list to store the extracted underlying stage nodes
    extracted_nodes = []

    # Evaluates whether the query string exists within the intra-stage columns list
    if query_str in intra_cols:

        # Assigns the detected type as an intra-stage feature
        result['Detected_Type'] = 'Intra'

        # Loops through each column in the change columns list to find the parent
        for c_col in change_cols:

            # Casts the current change column to a string
            c_str = str(c_col)

            # Evaluates whether the query string exactly matches or starts with the current change column prefix
            if query_str == c_str or query_str.startswith(f"{c_str}_"):

                # Assigns the matched change column as the base reference
                base_change_col = c_str

                # Breaks the loop once the parent change column is successfully identified
                break
                
    # Evaluates whether the query string exists directly within the change columns list
    elif query_str in change_cols:

        # Assigns the detected type as a change feature
        result['Detected_Type'] = 'Change'

        # Assigns the query string directly as the base change column
        base_change_col = query_str
        
    # Evaluates whether the query string exists directly within the absolute columns list
    elif query_str in abs_cols:

        # Assigns the detected type as an absolute feature
        result['Detected_Type'] = 'Absolute'

        # Assigns the query string as the sole extracted node
        extracted_nodes = [query_str]
        
    # Handles cases where the query string is not found in any provided list
    else:

        # Returns an error dictionary indicating the missing column
        return {"Error": f"Column '{query_str}' not found in any provided list."}
    
    # Evaluates whether a base change column was successfully identified
    if base_change_col:

        # Checks whether the base change column contains a standard transition arrow
        if '->' in base_change_col:

            # Splits the base change column string at the transition arrow
            parts = base_change_col.split('->')

            # Extracts and strips the origin and destination nodes into the tracking list
            extracted_nodes = [parts[0].strip(), parts[1].strip()]

        # Checks whether the base change column contains a custom shift identifier
        elif '_to_' in base_change_col:

            # Strips the custom prefixes from the column string
            clean_col = base_change_col.replace('Net_Shift_', '').replace('Total_Shift_', '')

            # Splits the cleaned column string at the custom transition identifier
            parts = clean_col.split('_to_')

            # Extracts and strips the origin and destination nodes into the tracking list
            extracted_nodes = [parts[0].strip(), parts[1].strip()]

    # Evaluates whether the detected type is an absolute column
    if result['Detected_Type'] == 'Absolute':

        # Appends the query string directly to the absolute columns list
        result['Absolute_Columns'] = [query_str]

    # Handles configurations for change and intra-stage types
    else:

        # Filters and assigns the extracted nodes that exist within the absolute columns list
        result['Absolute_Columns'] = [n for n in extracted_nodes if n in abs_cols]

    # Evaluates whether the detected type is an absolute column to populate change relationships
    if result['Detected_Type'] == 'Absolute':

        # Initialises an empty list to store matching change columns
        matched_changes = []

        # Loops through each column in the provided change columns list
        for c_col in change_cols:

            # Casts the current change column to a string
            c_str = str(c_col)

            # Checks whether the current change column contains a standard transition arrow
            if '->' in c_str:

                # Splits the string and strips whitespace to extract the component parts
                parts = [p.strip() for p in c_str.split('->')]

                # Evaluates whether the query string is present in the extracted parts
                if query_str in parts:

                    # Appends the matching change column to the tracking list
                    matched_changes.append(c_str)

            # Checks whether the current change column contains a custom shift identifier
            elif '_to_' in c_str:

                # Strips the custom prefixes from the column string
                clean_col = c_str.replace('Net_Shift_', '').replace('Total_Shift_', '')

                # Splits the string and strips whitespace to extract the component parts
                parts = [p.strip() for p in clean_col.split('_to_')]

                # Evaluates whether the query string is present in the extracted parts
                if query_str in parts:

                    # Appends the matching change column to the tracking list
                    matched_changes.append(c_str)

        # Assigns the compiled list of matched changes to the result dictionary
        result['Change_Columns'] = matched_changes

    # Handles configurations for change and intra-stage types
    else:

        # Evaluates whether the identified base change column exists in the provided change columns list
        if base_change_col in change_cols:

            # Assigns the base change column to the change columns list in the result dictionary
            result['Change_Columns'] = [base_change_col]

    # Evaluates whether the detected type is an absolute column to populate intra-stage relationships
    if result['Detected_Type'] == 'Absolute':

        # Initialises an empty list to store matching intra-stage columns
        matched_intras = []

        # Loops through each matched change column identified in the previous step
        for matched_c in result['Change_Columns']:

            # Finds and appends all intra-stage columns derived from or exactly matching the current change column
            matched_intras.extend([i for i in intra_cols if i == matched_c or str(i).startswith(f"{matched_c}_")])

        # Assigns the compiled list of matched intra-stage columns to the result dictionary
        result['Intra_Columns'] = matched_intras

    # Handles configurations for change and intra-stage types
    else:

        # Evaluates whether a base change column was successfully identified
        if base_change_col:

            # Finds and assigns all intra-stage columns derived from or exactly matching the base change column
            result['Intra_Columns'] = [i for i in intra_cols if i == base_change_col or str(i).startswith(f"{base_change_col}_")]

    # Returns the fully populated relationships dictionary
    return result

def get_ensemble_failures(flags_abs, flags_delta, flags_intra, min_overlaps=1):
    '''Identifies chips that failed absolute, delta and intra metrics on the same underlying base stages.
    
    Args:
        flags_abs (pd.DataFrame): The dataframe tracking absolute stage failures.
        flags_delta (pd.DataFrame): The dataframe tracking delta transition failures.
        flags_intra (pd.DataFrame): The dataframe tracking intra-stage dynamic failures.
        min_overlaps (int, optional): The minimum number of intersecting failure modes required. Defaults to 1.
        
    Returns:
        list: A list of chip IDs flagged as ensemble failures.
    '''
    
    # Initialises an empty list to store chips that fail across multiple paradigms
    ensemble_chips = []

    # Extracts and assigns only the final evaluated column in the absolute flags matrix
    critical_stages = flags_abs.columns[-1:].tolist()

    # Extracts the full list of absolute columns to pass to the relationship mapper
    abs_cols = flags_abs.columns.tolist()

    # Extracts the full list of change columns to pass to the relationship mapper
    change_cols = flags_delta.columns.tolist()

    # Extracts the full list of intra-stage columns to pass to the relationship mapper
    intra_cols = flags_intra.columns.tolist()

    # Initialises an empty dictionary to pre-compute relationships for every base stage
    stage_map = {}

    # Loops through each absolute stage column
    for stage in abs_cols:

        # Computes and maps the related columns for the current absolute stage
        stage_map[stage] = get_related_columns(stage, abs_cols, change_cols, intra_cols)

    # Identifies initial candidate chips by finding the union of chips failing at least once in any mode
    candidate_chips = set(flags_abs.index[flags_abs.any(axis=1)]) | set(flags_delta.index[flags_delta.any(axis=1)]) | set(flags_intra.index[flags_intra.any(axis=1)])

    # Loops through each identified candidate chip
    for chip in candidate_chips:

        # Initialises an empty set to track master stages that failed the intersection rule
        failed_stages = set()

        # Loops through every master stage and its mapped relationships
        for stage, rels in stage_map.items():
            
            # Initialises a boolean flag to track failure in the absolute dataset
            failed_in_abs = False

            # Initialises a boolean flag to track failure in the delta dataset
            failed_in_delta = False

            # Initialises a boolean flag to track failure in the intra-stage dataset
            failed_in_intra = False
            
            # Evaluates whether the chip failed the specific absolute stage
            if chip in flags_abs.index and flags_abs.loc[chip, stage] == 1:

                # Updates the absolute failure flag to true
                failed_in_abs = True
                
            # Evaluates whether the chip exists in the delta flags dataframe
            if chip in flags_delta.index:

                # Loops through each related change column
                for c_col in rels['Change_Columns']:

                    # Evaluates whether the chip failed the current related change column
                    if c_col in flags_delta.columns and flags_delta.loc[chip, c_col] == 1:

                        # Updates the delta failure flag to true
                        failed_in_delta = True

                        # Breaks the loop once a delta failure is confirmed
                        break 
                        
            # Evaluates whether the chip exists in the intra-stage flags dataframe
            if chip in flags_intra.index:

                # Loops through each related intra-stage column
                for i_col in rels['Intra_Columns']:

                    # Evaluates whether the chip failed the current related intra-stage column
                    if i_col in flags_intra.columns and flags_intra.loc[chip, i_col] == 1:

                        # Updates the intra-stage failure flag to true
                        failed_in_intra = True

                        # Breaks the loop once an intra-stage failure is confirmed
                        break 

            # Calculates the total number of datasets flagging a failure for the current stage
            datasets_failed = sum([failed_in_abs, failed_in_delta, failed_in_intra])
            
            # Evaluates whether the stage failed across all three evaluated datasets
            if datasets_failed == 3:

                # Appends the fully failed stage to the tracking set
                failed_stages.add(stage)

        # Evaluates whether the chip failed any of the defined critical stages
        failed_critical = any(stage in failed_stages for stage in critical_stages)

        # Evaluates whether the total failed stages meet the minimum threshold or target critical stages
        if len(failed_stages) >= min_overlaps:#or failed_critical:

            # Appends the confirmed chip to the final ensemble failures list
            ensemble_chips.append(chip)
            
    # Returns the compiled list containing all identified ensemble failure chips
    return ensemble_chips

def analyse_pel(events_df, changes_df, intra_df, val_col='quad_ch1', change_col='quad_ch1_change', intra_val_col=None, title='PEL Analysis', use_ensemble=True):
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

    # Assigns the appropriate intra-stage target column based on the provided inputs
    intra_target = intra_val_col if intra_val_col else val_col

    # Filters out non-informative startup stages from the events dataframe
    events_df = events_df[~events_df['stage'].isin(['Start', 'Initial'])]

    # Pivots the events dataframe to prepare for absolute signal anomaly detection
    wide_events = pivot_chip_data(events_df, 'stage', val_col)

    # Pivots the changes dataframe to prepare for stage delta anomaly detection
    wide_changes = pivot_chip_data(changes_df, 'stage', change_col)

    # Pivots the intra-stage dataframe to prepare for kinetic anomaly detection
    wide_intra = create_wide_intra(intra_df, intra_target)

    # Detects absolute signal anomalies across all stages
    anomalies_abs, flags_abs = detect_anomalies(wide_events, wide_events.columns.tolist())

    # Extracts the unique chip IDs flagged as absolute signal anomalies
    outliers_abs = get_outlier_chips(anomalies_abs)

    # Detects stage delta anomalies across all transitions
    anomalies_delta, flags_delta = detect_anomalies(wide_changes, wide_changes.columns.tolist())

    # Extracts the unique chip IDs flagged as stage delta anomalies
    outliers_delta = get_outlier_chips(anomalies_delta)

    # Detects intra-stage kinetic anomalies across all features
    anomalies_intra, flags_intra = detect_anomalies(wide_intra, wide_intra.columns.tolist())

    # Extracts the unique chip IDs flagged as intra-stage kinetic anomalies
    outliers_intra = get_outlier_chips(anomalies_intra)

    # Identifies chips that failed across absolute, delta, and intra metrics simultaneously
    ensemble_failed_chips = get_ensemble_failures(flags_abs, flags_delta, flags_intra)

    # Displays the absolute signal results in a collapsible output section
    with collapsible_output(f'{title} - Absolute Signal'):

        # Initialises the matplotlib figure with specific dimensions
        plt.figure(figsize=(10, 6))

        # Plots a heatmap of the absolute signal correlation matrix to show relationships between PEL transitions
        sns.heatmap(wide_events.corr(), cmap='plasma', center=0, annot=True)

        # Sets the title for the absolute signal heatmap
        plt.title(f'PEL [{val_col}]: Absolute Signal Correlation')

        # Adjusts the layout to prevent clipping of plot elements
        plt.tight_layout()

        # Renders the absolute signal heatmap
        plt.show()

        # Summarises and prints the top absolute signal correlations
        summarise_correlations(wide_events.corr(), f'PEL {val_col} Absolute Signal')

        # Prints the total count of potential absolute signal anomalous chips
        print(f'\nPotential Anomalous Chips (Absolute): {len(outliers_abs)}')

        # Analyses and prints the specific stages driving the absolute signal anomalies
        analyse_anomaly_drivers(wide_events, anomalies_abs, f'PEL {val_col} Absolute Signal', use_ensemble=use_ensemble, ensemble_chips=ensemble_failed_chips)

    # Displays the stage delta results in a collapsible output section
    with collapsible_output(f'{title} - Stage Delta'):

        # Initialises the matplotlib figure with specific dimensions
        plt.figure(figsize=(10, 6))

        # Plots a heatmap of the stage delta correlation matrix to show relationships between PEL response changes
        sns.heatmap(wide_changes.corr(), cmap='plasma', center=0, annot=True)

        # Sets the title for the stage delta heatmap
        plt.title(f'PEL [{change_col}]: Stage Delta Correlation')

        # Adjusts the layout to prevent clipping of plot elements
        plt.tight_layout()

        # Renders the stage delta heatmap
        plt.show()

        # Summarises and prints the top stage delta correlations
        summarise_correlations(wide_changes.corr(), f'PEL {change_col} Stage Delta')

        # Prints the total count of potential stage delta anomalous chips
        print(f'\nPotential Anomalous Chips (Delta): {len(outliers_delta)}')

        # Analyses and prints the specific stages driving the stage delta anomalies
        analyse_anomaly_drivers(wide_changes, anomalies_delta, f'PEL {change_col} Stage Delta', use_ensemble=use_ensemble, ensemble_chips=ensemble_failed_chips)
    
    # Displays the intra-stage kinetics results in a collapsible output section
    with collapsible_output(f'{title} - Intra-stage Kinetics'):

        # Initialises the matplotlib figure with specific dimensions
        plt.figure(figsize=(20, 20))

        # Plots a heatmap of the intra-stage kinetic correlation matrix to show relationships between PEL response features
        sns.heatmap(wide_intra.corr(), cmap='plasma', center=0, annot=True)

        # Sets the title for the intra-stage kinetics heatmap
        plt.title(f'PEL [{intra_target}]: Intra-stage Kinetics Correlation')

        # Adjusts the layout to prevent clipping of plot elements
        plt.tight_layout()

        # Renders the intra-stage kinetics heatmap
        plt.show()

        # Summarises and prints the top intra-stage kinetic correlations
        summarise_correlations(wide_intra.corr(), f'PEL {intra_target} Intra-stage Kinetics')

        # Prints the total count of potential intra-stage kinetic anomalous chips
        print(f'\nPotential Anomalous Chips (Intra-stage Kinetics): {len(outliers_intra)}')

        # Analyses and prints the specific stages driving the intra-stage kinetic anomalies
        analyse_anomaly_drivers(wide_intra, anomalies_intra, f'PEL {intra_target} Intra-stage Kinetics', use_ensemble=use_ensemble, ensemble_chips=ensemble_failed_chips)

    # Returns the generated wide tables and their corresponding outlier lists
    return wide_events, wide_changes, wide_intra, outliers_abs, outliers_delta, outliers_intra, ensemble_failed_chips

def analyse_immob_split(events_df, changes_df, intra_df, split_name, val_col='channel1', change_col='channel1_change', intra_val_col=None, title='Immobilisation Analysis', use_ensemble=True):
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

    # Assigns the appropriate intra-stage target column based on the provided inputs
    intra_target = intra_val_col if intra_val_col else val_col

    # Filters out non-informative startup stages from the events dataframe
    events_df = events_df[~events_df['stage'].isin(['Start', 'Initial'])]
    
    # Pivots the events dataframe to prepare for absolute signal anomaly detection
    wide_events = pivot_chip_data(events_df, 'stage', val_col)

    # Pivots the changes dataframe to prepare for stage delta anomaly detection
    wide_changes = pivot_chip_data(changes_df, 'stage', change_col)

    # Pivots the intra-stage dataframe to prepare for kinetic anomaly detection
    wide_intra = create_wide_intra(intra_df, intra_target)

    # Detects absolute signal anomalies across all stages
    anomalies_abs, flags_abs = detect_anomalies(wide_events, wide_events.columns.tolist())

    # Extracts the unique chip IDs flagged as absolute signal anomalies
    outliers_abs = get_outlier_chips(anomalies_abs)

    # Detects stage delta anomalies across all transitions
    anomalies_delta, flags_delta = detect_anomalies(wide_changes, wide_changes.columns.tolist())

    # Extracts the unique chip IDs flagged as stage delta anomalies
    outliers_delta = get_outlier_chips(anomalies_delta)

    # Detects intra-stage kinetic anomalies across all features
    anomalies_intra, flags_intra = detect_anomalies(wide_intra, wide_intra.columns.tolist())

    # Extracts the unique chip IDs flagged as intra-stage kinetic anomalies
    outliers_intra = get_outlier_chips(anomalies_intra)

    # Identifies chips that failed across absolute, delta, and intra metrics simultaneously
    ensemble_failed_chips = get_ensemble_failures(flags_abs, flags_delta, flags_intra)
    
    # Displays the absolute signal results in a collapsible output section
    with collapsible_output(f'{title} - Absolute Signal'):

        # Initialises the matplotlib figure with specific dimensions
        plt.figure(figsize=(20, 20))

        # Plots a heatmap of the absolute signal correlation matrix to compare immobilisation transitions
        sns.heatmap(wide_events.corr(), cmap='plasma', annot=True, annot_kws={'size': 12})

        # Sets the title for the absolute signal heatmap
        plt.title(f'Immob [{split_name}] - {val_col}: Absolute Signal Correlation')

        # Adjusts the layout to prevent clipping of plot elements
        plt.tight_layout()

        # Renders the absolute signal heatmap
        plt.show()

        # Summarises and prints the top absolute signal correlations
        summarise_correlations(wide_events.corr(), f'Immob [{split_name}] {val_col} Absolute Signal')

        # Prints the total count of potential absolute signal anomalous chips
        print(f'\nPotential Anomalous Chips (Absolute): {len(outliers_abs)}')

        # Analyses and prints the specific stages driving the absolute signal anomalies
        analyse_anomaly_drivers(wide_events, anomalies_abs, f'Immob [{split_name}] {val_col} Absolute Signal', use_ensemble=use_ensemble, ensemble_chips=ensemble_failed_chips)

    # Displays the stage delta results in a collapsible output section
    with collapsible_output(f'{title} - Stage Delta'):

        # Initialises the matplotlib figure with specific dimensions
        plt.figure(figsize=(20, 20))

        # Plots a heatmap of the stage delta correlation matrix to compare immobilisation response changes
        sns.heatmap(wide_changes.corr(), cmap='plasma', center=0, annot=True)

        # Sets the title for the stage delta heatmap
        plt.title(f'Immob [{split_name}] - {change_col}: Stage Delta Correlation')

        # Renders the stage delta heatmap
        plt.show()

        # Summarises and prints the top stage delta correlations
        summarise_correlations(wide_changes.corr(), f'Immob [{split_name}] {change_col} Stage Delta')

        # Prints the total count of potential stage delta anomalous chips
        print(f'\nPotential Anomalous Chips (Delta): {len(outliers_delta)}')

        # Analyses and prints the specific stages driving the stage delta anomalies
        analyse_anomaly_drivers(wide_changes, anomalies_delta, f'Immob [{split_name}] {change_col} Stage Delta', use_ensemble=use_ensemble, ensemble_chips=ensemble_failed_chips)
    
    # Displays the intra-stage kinetics results in a collapsible output section
    with collapsible_output(f'{title} - Intra-stage Kinetics'):

        # Initialises the matplotlib figure with specific dimensions
        plt.figure(figsize=(20, 20))

        # Plots a heatmap of the intra-stage kinetic correlation matrix to compare immobilisation response features
        sns.heatmap(wide_intra.corr(), cmap='plasma', center=0, annot=True)

        # Sets the title for the intra-stage kinetics heatmap
        plt.title(f'Immob [{split_name}] - {intra_target}: Intra-stage Kinetics Correlation')

        # Renders the intra-stage kinetics heatmap
        plt.show()

        # Summarises and prints the top intra-stage kinetic correlations
        summarise_correlations(wide_intra.corr(), f'Immob [{split_name}] {intra_target} Intra-stage Kinetics')

        # Prints the total count of potential intra-stage kinetic anomalous chips
        print(f'\nPotential Anomalous Chips (Intra-stage Kinetics): {len(outliers_intra)}')

        # Analyses and prints the specific stages driving the intra-stage kinetic anomalies
        analyse_anomaly_drivers(wide_intra, anomalies_intra, f'Immob [{split_name}] {intra_target} Intra-stage Kinetics', use_ensemble=use_ensemble, ensemble_chips=ensemble_failed_chips)
    
    # Returns the generated wide tables and their corresponding outlier lists
    return wide_events, wide_changes, wide_intra, outliers_abs, outliers_delta, outliers_intra, ensemble_failed_chips

def plot_pel_layer_shifts(change_wide_df, channel_name='quad_ch1'):
    '''Plots PEL signal shifts between layers for each requested channel.

    Args:
        change_wide_df (pd.DataFrame): Wide-format PEL stage-transition measurements (rows=chip_id, cols=stages).
        channel_name (str, optional): Name of the channel for the plot title. Defaults to 'quad_ch1'.
    '''
    
    # Generates a list of layer columns by filtering out the total shift summary
    layer_cols = [col for col in change_wide_df.columns if col != 'Total_Shift_Initial_to_Final']

    # Creates a copy of the dataframe to preserve the original data, retaining only the layer columns
    df_plot = change_wide_df[layer_cols].copy()

    # Displays the visualisations within a collapsible output section
    with collapsible_output(f'PEL Layer Shifts: {channel_name}'):
        
        # Initialises a plotly graph object figure for interactive viewing
        fig = go.Figure()
        
        # Loops through each grouped chip ID and its associated row data
        for chip_id, row in df_plot.iterrows():

            # Adds a scatter trace for the current chip's layer shifts
            fig.add_trace(go.Scatter(
                x=layer_cols, 
                y=row[layer_cols], 
                mode='lines+markers', 
                name=str(chip_id), 
                opacity=0.6, 
                marker=dict(size=6), 
                visible='legendonly'
            ))

        # Updates the plotly figure layout with titles and interactive styling
        fig.update_layout(
            title=f'PEL Layer Build-up: Shift per Transition ({channel_name})', 
            xaxis_title='Layer Transition', 
            yaxis_title='Signal Shift (Δ)', 
            template='plotly_white', 
            hovermode='x unified', 
            height=600
        )

        # Angles the x-axis tick labels to prevent text overlap
        fig.update_xaxes(tickangle=45)

        # Renders the interactive plotly figure
        fig.show()

        # Initialises a matplotlib figure for static viewing with specific dimensions
        plt.figure(figsize=(10, 6))
        
        # Loops through each grouped chip ID and its associated row data again
        for chip_id, row in df_plot.iterrows():

            # Plots a static line for the current chip's layer shifts
            plt.plot(layer_cols, row[layer_cols], marker='o', alpha=0.5, label=chip_id)

        # Sets the title for the static layer shift plot
        plt.title(f'PEL Layer Build-up: Shift per Transition ({channel_name})')

        # Sets the x-axis label
        plt.xlabel('Layer Transition')

        # Sets the y-axis label
        plt.ylabel('Signal Shift (Δ)')

        # Rotates the x-axis ticks to improve readability
        plt.xticks(rotation=45)

        # Adds a subtle background grid to the plot
        plt.grid(True, alpha=0.3)

        # Evaluates if the number of unique chips is small enough to display a readable legend
        if len(df_plot.index) <= 25:

            # Adds the legend positioned outside the main plotting area
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Adjusts the layout to accommodate the legend and angled labels
        plt.tight_layout()

        # Renders the static matplotlib plot
        plt.show()

def combine_anomalies(*outliers_lists, title='Combined Anomalies Summary'):
    '''Combines an arbitrary number of outlier lists into a single set.

    Args:
        *outliers_lists (list): Variable length argument list of outlier chip IDs.
        title (str, optional): The title for the summary output. Defaults to 'Combined Anomalies Summary'.

    Returns:
        list: A deduplicated list of all outlier chips.
    '''

    # Initialises an empty set to store and automatically deduplicate unique outlier chip IDs
    all_outlier_chips = set()

    # Loops through each provided list of outliers passed to the function
    for outliers in outliers_lists:

        # Adds the current batch of outliers into the tracking set
        all_outlier_chips.update(outliers)
        
    # Converts the aggregated deduplicated set back into a standard list
    all_outlier_chips = list(all_outlier_chips)

    # Displays the compiled results within a collapsible output section
    with collapsible_output(title):

        # Prints the total count of unique anomalous chips identified across all lists
        print(f'Total Unique Anomalous Chips: {len(all_outlier_chips)}')

        # Evaluates whether any anomalous chips were compiled
        if all_outlier_chips:

            # Prints the list of identified chip IDs
            print(f'Chip IDs: {all_outlier_chips}')

        else:

            # Prints a confirmation message if no anomalies were detected in any list
            print('No anomalies detected in this combination.')

    # Returns the final unified list of outlier chips
    return all_outlier_chips

def analyse_cross_stage_split(pel_df, immob_df, split_name, channel_name, pel_outliers=None, immob_outliers=None, sc_metrics_df=None, title='Cross-Stage Validation'):
    '''Merges PEL and a specific immobilisation split, runs PCA and clustering, plots the results with anomaly highlights for Channel 1, Channel 2, and overall. 
    
    Validates against standard curves if provided.

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
    
    # Creates a copy of the PEL dataframe to preserve the original data
    pel_copy = pel_df.copy()

    # Creates a copy of the immobilisation dataframe to preserve the original data
    immob_copy = immob_df.copy()
    
    # Casts the PEL column names to strings to ensure consistent merging
    pel_copy.columns = pel_copy.columns.astype(str)

    # Casts the immobilisation column names to strings to ensure consistent merging
    immob_copy.columns = immob_copy.columns.astype(str)

    # Merges the PEL and immobilisation datasets using an inner join on their indices
    merged = pd.merge(pel_copy, immob_copy, left_index=True, right_index=True, how='inner', suffixes=('_PEL', '_Immob')).dropna()
    
    # Checks whether the merged dataframe contains data
    if merged.empty:

        # Prints a warning message indicating insufficient overlapping chips
        print(f'[{split_name} - {channel_name}] Not enough overlapping chips between PEL and Immobilisation for cross-analysis.')
        
        return
             
    # Displays the results within a collapsible output section
    with collapsible_output(f'{title} - {channel_name}'):

        # Prints the count of overlapping chips analysed
        print(f'[{split_name} - {channel_name}] Overlapping chips analysed: {len(merged)}')
        
        # Extracts the list of feature column names from the merged dataframe
        features = merged.columns.tolist()

        # Standardises the extracted features using a robust scaler
        X_subset = RobustScaler().fit_transform(merged[features])
        
        # Initialises the PCA model to reduce the data to two principal components
        pca = PCA(n_components=2)

        # Fits the PCA model and transforms the standardised feature subset
        pca_result = pca.fit_transform(X_subset)
        
        # Assigns the first principal component values to the merged dataframe
        merged['PCA1'] = pca_result[:, 0]

        # Assigns the second principal component values to the merged dataframe
        merged['PCA2'] = pca_result[:, 1]
        
        # Prints the percentage of variance explained by the first two principal components
        print(f'[{split_name} - {channel_name}] Explained Variance by first 2 components: {pca.explained_variance_ratio_.sum()*100:.2f}%')
        
        # Initialises the KMeans clustering model with two target clusters
        kmeans = KMeans(n_clusters=2, random_state=8030, n_init=10)

        # Fits the KMeans model and assigns the predicted cluster labels to the dataframe
        merged['Cluster'] = kmeans.fit_predict(X_subset)
        
        # Initialises a matplotlib figure with specific dimensions for the scatter plot
        plt.figure(figsize=(10, 6))

        # Plots the first two principal components to visualise cluster separation
        sns.scatterplot(data=merged, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', s=100, alpha=0.7)

        # Combines outlier chips from both stages into a unified deduplicated set
        both_anomalies = set(pel_outliers if pel_outliers else []).union(set(immob_outliers if immob_outliers else []))
            
        # Creates a boolean mask identifying flagged anomalies within the current subset index
        anomaly_mask = merged.index.isin(both_anomalies)
        
        # Evaluates whether any flagged anomalies exist in the current subset
        if anomaly_mask.any():

            # Highlights the identified anomalies on the scatter plot with distinct markers
            plt.scatter(merged.loc[anomaly_mask, 'PCA1'], merged.loc[anomaly_mask, 'PCA2'], edgecolor='black', facecolor='none', s=200, label='Flagged Anomaly', linewidth=2)
                        
        # Sets the title for the cross-stage PCA scatter plot
        plt.title(f'Cross-Stage PCA: {split_name} ({channel_name})')

        # Sets the x-axis label for the first principal component
        plt.xlabel('Principal Component 1 (General Signal Variance)')

        # Sets the y-axis label for the second principal component
        plt.ylabel('Principal Component 2 (Stage-to-Stage Dynamic Variance)')

        # Adds a legend to the scatter plot
        plt.legend()

        # Adds a subtle background grid to improve plot readability
        plt.grid(True, alpha=0.3)

        # Renders the PCA scatter plot
        plt.show()
        
        # Prints the cluster profiling header
        print(f'\n Cluster Profiling [{split_name} - {channel_name}]')
        
        # Calculates the statistical means of the features grouped by cluster
        cluster_means = merged[features].groupby(merged['Cluster']).mean()

        # Calculates the overall statistical means of the features across all clusters
        overall_mean = merged[features].mean()

        # Calculates the overall standard deviations, substituting zeros to prevent division errors
        overall_std = merged[features].std().replace(0, 1e-9)
        
        # Calculates the z-scores to profile how each cluster deviates from the overall mean
        cluster_zscores = (cluster_means - overall_mean) / overall_std
        
        # Loops through each unique and sorted cluster ID
        for cluster_id in sorted(merged['Cluster'].unique()):

            # Prints the profile header and observation count for the current cluster
            print(f"\nCluster {cluster_id} Profile (n={sum(merged['Cluster'] == cluster_id)} observations):")
            
            # Extracts and sorts the highest positive z-score features for the current cluster
            top_pos = cluster_zscores.loc[cluster_id].sort_values(ascending=False).head(3)

            # Extracts and sorts the lowest negative z-score features for the current cluster
            top_neg = cluster_zscores.loc[cluster_id].sort_values(ascending=True).head(3)
            
            # Prints the header for defining high features
            print('  Defining High Features (Above Average):')

            # Initialises a boolean flag to track if significant positive features are found
            pos_found = False

            # Loops through each feature and its corresponding z-score in the top positive list
            for feat, z in top_pos.items():

                # Evaluates if the positive z-score exceeds the significance threshold
                if z > 0.5:

                    # Prints the significantly high feature and its z-score
                    print(f'    - {feat}: +{z:.2f} standard deviations')

                    # Updates the positive flag to indicate significant features were found
                    pos_found = True
                    
            # Checks if no significant positive features were identified
            if not pos_found: 

                # Prints a message indicating the absence of significantly high features
                print('    - None significantly above average')
                
            # Prints the header for defining low features
            print('  Defining Low Features (Below Average):')

            # Initialises a boolean flag to track if significant negative features are found
            neg_found = False

            # Loops through each feature and its corresponding z-score in the top negative list
            for feat, z in top_neg.items():

                # Evaluates if the negative z-score falls below the significance threshold
                if z < -0.5:

                    # Prints the significantly low feature and its z-score
                    print(f'    - {feat}: {z:.2f} standard deviations')

                    # Updates the negative flag to indicate significant features were found
                    neg_found = True

            # Checks if no significant negative features were identified
            if not neg_found: 

                # Prints a message indicating the absence of significantly low features
                print('    - None significantly below average')
        
        # Evaluates whether standard curve metrics are provided for validation
        if sc_metrics_df is not None:

            # Prints the functional yield comparison header
            print(f'\n--- Cluster Functional Yield (R² Comparison) [{split_name} - {channel_name}] ---')
            
            # Extracts the cluster assignments mapping
            cluster_mapping = merged[['Cluster']].reset_index()

            # Checks if the reset index column is named 'index'
            if 'index' in cluster_mapping.columns:

                # Renames the index column to match the standard chip ID naming convention
                cluster_mapping = cluster_mapping.rename(columns={'index': 'chip_id'})
                
            # Merges the standard curve metrics with the cluster mappings based on chip IDs
            sc_cluster_df = pd.merge(sc_metrics_df.reset_index(), cluster_mapping, on='chip_id', how='inner')
            
            # Checks whether the merged standard curve dataframe contains data
            if sc_cluster_df.empty:

                # Prints a message indicating no standard curve data is available for mapping
                print('No Standard Curve data available to map to clusters.')

            else:

                # Aggregates and calculates standard curve R-squared performance statistics per cluster
                cluster_stats = sc_cluster_df.groupby('Cluster')['r2'].agg(['count', 'mean', 'median', 'std']).fillna(0)

                # Prints the standard curve performance header
                print('Standard Curve R² Performance by Cluster:')

                # Displays the aggregated standard curve statistics
                display(cluster_stats)
                
                # Initialises a matplotlib figure for the standard curve boxplot
                plt.figure(figsize=(10, 6))

                # Plots the standard curve R-squared distribution by cluster using a boxplot
                sns.boxplot(data=sc_cluster_df, x='Cluster', y='r2', showmeans=True,  meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'black'})

                # Overlays a stripplot to show individual data points within the clusters
                sns.stripplot(data=sc_cluster_df, x='Cluster', y='r2', color='black', alpha=0.5, jitter=True)

                # Draws a horizontal line to indicate the pass threshold
                plt.axhline(0.95, color='red', linestyle='--', label='Pass Threshold (0.95)')
                
                # Sets the title for the standard curve distribution plot
                plt.title(f'Standard Curve R² Distribution by Manufacturing Cluster\n{split_name} ({channel_name})')

                # Sets the x-axis label for the manufacturing clusters
                plt.xlabel('Manufacturing Cluster')

                # Sets the y-axis label for the standard curve scores
                plt.ylabel('Standard Curve R² Score')

                # Adds a legend to the standard curve distribution plot
                plt.legend()

                # Adds a subtle background grid to the standard curve plot
                plt.grid(True, alpha=0.3)

                # Adjusts the layout to prevent clipping of plot elements
                plt.tight_layout()

                # Renders the standard curve distribution plot
                plt.show()
                
            # Extracts the specific flagged anomaly chip IDs present within the current subset
            anomaly_ids = list(both_anomalies.intersection(set(merged.index)))
            
            # Checks if there are no anomalous chips to validate
            if not anomaly_ids:

                # Prints a message indicating the absence of anomalies for functional validation
                print(f'[{split_name} - {channel_name}] No anomalous chips from the cross-stage analysis to validate against SC.')

            else:

                # Prints the validation header for the flagged anomalous chips
                print(f'\n[{split_name} - {channel_name}] Validating functional performance for anomalous chip(s): {anomaly_ids}\n')
                
                # Extracts the standard curve metrics corresponding to the anomalous chips
                anomaly_sc_data = sc_metrics_df[sc_metrics_df.index.isin(anomaly_ids)]
                
                # Checks whether the extracted anomaly standard curve data contains records
                if anomaly_sc_data.empty:

                    # Prints a message indicating missing standard curve data for the anomalies
                    print('No Standard Curve data found for the flagged anomalous chip(s).')

                else:

                    # Prints the formatted table header for the validation results
                    print(f"{'Chip ID':<15} | {'Curve UUID':<40} | {'R² Score':<10} | {'Status'}")

                    # Prints a separator line for the table
                    print('-' * 80)

                    # Loops through each row in the anomalous standard curve data
                    for chip_id, row in anomaly_sc_data.iterrows():

                        # Extracts the R-squared score from the current row
                        r2 = row['r2']

                        # Extracts the standard curve UUID from the current row
                        curve_id = row['standard_curve_uuid']

                        # Determines the pass or fail status based on the R-squared threshold
                        status = 'PASSED' if r2 >= 0.95 else 'FAILED'

                        # Prints the formatted validation results for the current anomalous chip
                        print(f'{chip_id:<15} | {curve_id:<40} | {r2:<10.4f} | {status}')
            
            # Isolates the standard curve metrics for the normal baseline chips
            normal_sc_data = sc_metrics_df[~sc_metrics_df.index.isin(anomaly_ids)]

            # Checks whether the normal baseline data contains records
            if not normal_sc_data.empty:

                # Prints the average R-squared score for the normal baseline chips
                print(f"\n>>> Average R² for NORMAL chips: {normal_sc_data['r2'].mean():.4f}")

        else:

            # Extracts the specific flagged anomaly chip IDs present within the current subset
            anomaly_ids = list(both_anomalies.intersection(set(merged.index)))

            # Prints the anomaly identification header
            print(f'\n--- Anomaly Identifications [{split_name} - {channel_name}] ---')

            # Checks whether any anomalous chips were identified
            if anomaly_ids:

                # Prints the list of flagged anomalous chips
                print(f'Flagged Anomalous Chips in cluster: {anomaly_ids}')

            else:

                # Prints a message indicating no anomalies were found in the current split
                print('No anomalous chips detected in this cross-stage split.')

def print_correlation_insights(corr_matrix, split_name, channel_name, top_n=3):
    '''Extracts and prints top positive, negative, and weakest correlations.

    Args:
        corr_matrix (pd.DataFrame): The correlation matrix to analyse.
        split_name (str): The label for the current data split.
        channel_name (str): The name of the channel.
        top_n (int, optional): The number of top correlations to return. Defaults to 3.
    '''

    # Prints the correlation insights header
    print(f'{split_name} ({channel_name}) Signal Correlation Insights')
    
    # Extracts the upper triangle of the correlation matrix to avoid duplicated pairs
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Stacks the upper triangle into a series and drops missing values
    corr_pairs = upper_tri.stack().dropna()
    
    # Checks whether the stacked correlation series contains data
    if corr_pairs.empty:

        # Prints a warning message indicating insufficient variance
        print('Not enough variance to calculate correlation insights.\n')
        
        return

    # Prints the header for top positive correlations
    print('Top Positive Correlations:')
    
    # Isolates and sorts the strongest positive correlations in descending order
    top_pos = corr_pairs[corr_pairs > 0].sort_values(ascending=False).head(top_n)

    # Checks whether the top positive correlations series is empty
    if top_pos.empty: 

        # Prints a null message if no positive correlations are found
        print('  None')

    else:

        # Loops through each feature pair and value in the top positive correlations
        for (feat1, feat2), val in top_pos.items(): 

            # Prints the paired features and their correlation value
            print(f'  - {feat1} & {feat2} (r = {val:.3f})')

    # Prints the header for top negative correlations
    print('\nTop Negative (Inverse) Correlations:')
    
    # Isolates and sorts the strongest negative correlations in ascending order
    top_neg = corr_pairs[corr_pairs < 0].sort_values(ascending=True).head(top_n)

    # Checks whether the top negative correlations series is empty
    if top_neg.empty: 

        # Prints a null message if no negative correlations are found
        print('  None')

    else:

        # Loops through each feature pair and value in the top negative correlations
        for (feat1, feat2), val in top_neg.items(): 

            # Prints the paired features and their negative correlation value
            print(f'  - {feat1} & {feat2} (r = {val:.3f})')

    # Prints the header for weakest relationships
    print('\nWeakest Relationships (Closest to zero):')
    
    # Extracts the relationships with absolute values closest to zero
    weakest = corr_pairs.reindex(corr_pairs.abs().sort_values().index).head(top_n)

    # Loops through each feature pair and value in the weakest relationships
    for (feat1, feat2), val in weakest.items():

        # Prints the paired features and their weak correlation value
        print(f'  - {feat1} & {feat2} (r = {val:.3f})')

    # Prints a newline for visual spacing
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

    # Creates a copy of the PEL dataframe to preserve the original data
    pel_copy = pel_df.copy()

    # Creates a copy of the immobilisation dataframe to preserve the original data
    immob_copy = immob_df.copy()

    # Casts the PEL column names to strings to ensure consistent merging
    pel_copy.columns = pel_copy.columns.astype(str)

    # Casts the immobilisation column names to strings to ensure consistent merging
    immob_copy.columns = immob_copy.columns.astype(str)
    
    # Merges the PEL and immobilisation datasets using an inner join on their indices
    merged = pel_copy.merge(immob_copy, left_index=True, right_index=True, how='inner', suffixes=('_PEL', '_Immob'))

    # Evaluates whether standard curve metrics are provided for integration
    if sc_metrics_df is not None:

        # Creates a copy of the standard curve dataframe to preserve the original data
        sc_copy = sc_metrics_df.copy()

        # Casts the standard curve column names to strings
        sc_copy.columns = sc_copy.columns.astype(str)

        # Extracts a list of all numeric columns from the standard curve dataframe
        numeric_cols = sc_copy.select_dtypes(include='number').columns.tolist()
        
        # Checks if a datetime column exists in the standard curve dataframe
        if 'datetime' in sc_copy.columns: 

            # Filters the dataframe to include the datetime column alongside the numeric metrics
            sc_copy = sc_copy[['datetime'] + numeric_cols]

        else: 

            # Filters the dataframe to include only the numeric metrics
            sc_copy = sc_copy[numeric_cols]
            
        # Selects the earliest standard curve entry per chip to avoid duplicate merges
        sc_first = sc_copy.sort_values('datetime').groupby(sc_copy.index, sort=False).first().drop(columns='datetime')
        
        # Merges the deduplicated standard curve metrics into the primary dataset using a left join
        merged = merged.merge(sc_first, left_index=True, right_index=True, how='left')

    # Displays the results within a collapsible output section
    with collapsible_output(f'{title} - {channel_name}'):
            
        # Checks whether the merged dataframe contains data
        if merged.empty:

            # Prints a warning message indicating insufficient overlapping chips
            print(f'[{split_name} - {channel_name}] Not enough overlapping chips for cross-analysis.')

            # Returns control to the calling function
            return

        # Prints the count of overlapping data points analysed
        print(f'[{split_name} - {channel_name}] Overlapping data points analysed: {len(merged)}\n')

        # Calculates the correlation matrix for the merged dataset
        corr_matrix = merged.corr()

        # Extracts and prints the top correlation insights from the matrix
        print_correlation_insights(corr_matrix, split_name, channel_name, top_n=3)

        # Initialises a matplotlib figure with specific dimensions for the correlation heatmap
        plt.figure(figsize=(20, 20))

        # Plots the cross-stage correlation matrix to show relationships between all selected features
        sns.heatmap(corr_matrix, cmap='plasma', center=0, annot=False)

        # Sets the title for the overall correlation heatmap
        plt.title(f'{split_name} ({channel_name}) - Overall Correlation')

        # Adjusts the layout to prevent clipping of plot elements
        plt.tight_layout()

        # Renders the correlation heatmap
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
    
    # Initialises the checkbox for filtering Protein G stages in immobilisation sources
    prg_checkbox = widgets.Checkbox(value=False, description='PrG Stages Only (Immob)', tooltip='Filter Immobilisation metrics to PrG stages only', style={'description_width': 'initial'})
    
    # Groups the mode toggle and Protein G checkbox into a horizontal box
    top_controls = widgets.HBox([mode_toggle, prg_checkbox], layout=widgets.Layout(align_items='center', grid_gap='20px'))

    # Initialises the channel dropdown widget
    channel_dropdown = widgets.Dropdown(options=['Channel 1', 'Channel 2', 'Both'], value='Channel 1', description='Channel:')

    # Initialises the horizontal axis source dropdown widget
    x_source_drop = widgets.Dropdown(options=all_sources, value=all_sources[0], description='X Source:')

    # Initialises the vertical axis source dropdown widget
    y_source_drop = widgets.Dropdown(options=all_sources, value=all_sources[0], description='Y Source:')

    # Initialises the horizontal axis metric dropdown widget
    x_col_drop = widgets.Dropdown(description='X Metric:')

    # Initialises the vertical axis metric dropdown widget
    y_col_drop = widgets.Dropdown(description='Y Metric:')

    # Initialises the bounded float text input for the horizontal axis minimum filter
    x_min_input = widgets.BoundedFloatText(description='Min X:')

    # Initialises the bounded float text input for the horizontal axis maximum filter
    x_max_input = widgets.BoundedFloatText(description='Max X:')

    # Initialises the bounded float text input for the vertical axis minimum filter
    y_min_input = widgets.BoundedFloatText(description='Min Y:')

    # Initialises the bounded float text input for the vertical axis maximum filter
    y_max_input = widgets.BoundedFloatText(description='Max Y:')

    # Initialises the button to reset the horizontal axis filters to their original bounds
    x_reset_btn = widgets.Button(description='Reset X', button_style='info', tooltip='Reset X filters to original bounds')

    # Initialises the button to reset the vertical axis filters to their original bounds
    y_reset_btn = widgets.Button(description='Reset Y', button_style='info', tooltip='Reset Y filters to original bounds')

    # Initialises the integer slider for the horizontal axis histogram bins
    x_bins_slider = widgets.IntSlider(value=30, min=5, max=150, step=5, description='X Hist Bins:', layout=widgets.Layout(width='300px'))

    # Initialises the integer slider for the vertical axis histogram bins
    y_bins_slider = widgets.IntSlider(value=30, min=5, max=150, step=5, description='Y Hist Bins:', layout=widgets.Layout(width='300px'))

    def reset_x_filters(b):
        '''Resets the X-axis filters to their maximum available limits.'''

        # Assigns the absolute minimum allowed value to the horizontal minimum input
        x_min_input.value = x_min_input.min

        # Assigns the absolute maximum allowed value to the horizontal maximum input
        x_max_input.value = x_max_input.max

    def reset_y_filters(b):
        '''Resets the Y-axis filters to their maximum available limits.'''

        # Assigns the absolute minimum allowed value to the vertical minimum input
        y_min_input.value = y_min_input.min

        # Assigns the absolute maximum allowed value to the vertical maximum input
        y_max_input.value = y_max_input.max

    # Binds the horizontal reset function to the respective reset button click event
    x_reset_btn.on_click(reset_x_filters)

    # Binds the vertical reset function to the respective reset button click event
    y_reset_btn.on_click(reset_y_filters)

    # Groups the horizontal axis filters into a horizontal box container initially hidden from view
    x_filters = widgets.HBox([x_min_input, x_max_input, x_reset_btn], layout=widgets.Layout(display='none'))

    # Groups the vertical axis filters into a horizontal box container initially hidden from view
    y_filters = widgets.HBox([y_min_input, y_max_input, y_reset_btn], layout=widgets.Layout(display='none'))

    # Initialises the text input for dynamically filtering the available chips list
    chip_search = widgets.Text(placeholder='Filter available...', layout=widgets.Layout(width='95%'))
    
    # Initialises the selection box for displaying available chips
    chip_select = widgets.Select(options=[], rows=5, layout=widgets.Layout(width='95%'))
    
    # Groups the available chips search and selection widgets into the left vertical panel
    left_panel = widgets.VBox([widgets.HTML('<b>Available Chips:</b>'), chip_search, chip_select], layout=widgets.Layout(width='40%'))

    # Initialises the text input for dynamically filtering the excluded chips list
    excluded_search = widgets.Text(placeholder='Filter excluded...', layout=widgets.Layout(width='95%'))
    
    # Initialises the selection box for displaying excluded chips
    excluded_select = widgets.Select(options=[], rows=5, layout=widgets.Layout(width='95%'))
    
    # Groups the excluded chips search and selection widgets into the right vertical panel
    right_panel = widgets.VBox([widgets.HTML('<b>Excluded Chips:</b>'), excluded_search, excluded_select], layout=widgets.Layout(width='40%'))

    # Initialises the action button to exclude a single highlighted chip
    exclude_btn = widgets.Button(description='Exclude >', button_style='warning', layout=widgets.Layout(width='120px'))

    # Initialises the action button to exclude all currently valid chips
    exclude_all_btn = widgets.Button(description='Exclude All >>', button_style='danger', layout=widgets.Layout(width='120px'))

    # Initialises the action button to re-add a single highlighted chip to the available pool
    readd_btn = widgets.Button(description='< Re-add', button_style='info', layout=widgets.Layout(width='120px'))

    # Initialises the action button to clear all current exclusions
    reset_btn = widgets.Button(description='<< Reset All', button_style='success', layout=widgets.Layout(width='120px'))
    
    # Creates a blank HTML spacer to vertically align the action buttons with the adjacent text inputs
    button_spacer = widgets.HTML('<b>&nbsp;</b>')
    
    # Groups the spacer and all action buttons into the central vertical panel
    button_panel = widgets.VBox([button_spacer, exclude_btn, exclude_all_btn, readd_btn, reset_btn], layout=widgets.Layout(width='20%', justify_content='flex-start', align_items='center'))
    
    def refresh_ui(*args):
        '''Updates both chip listboxes based on current exclusions and dynamically valid chips.'''

        # Extracts and converts the available chips search term to lowercase
        avail_term = chip_search.value.lower()

        # Filters and updates the available chips selection list based on exclusions and the search term
        chip_select.options = [c for c in state['valid_chips'] if c not in state['excluded_chips'] and avail_term in c.lower()]
        
        # Extracts and converts the excluded chips search term to lowercase
        excl_term = excluded_search.value.lower()

        # Filters and updates the excluded chips selection list based on exclusions and the search term
        excluded_select.options = [c for c in state['valid_chips'] if c in state['excluded_chips'] and excl_term in c.lower()]

    # Binds the refresh function to the available chips search input changes
    chip_search.observe(refresh_ui, names='value')

    # Binds the refresh function to the excluded chips search input changes
    excluded_search.observe(refresh_ui, names='value')

    def exclude_selected(b):
        '''Excludes the currently highlighted chip from the available list.'''

        # Evaluates whether a valid chip is currently highlighted in the available list
        if chip_select.value:

            # Adds the highlighted chip to the excluded chips tracking set
            state['excluded_chips'].add(chip_select.value)

            # Refreshes the user interface to reflect the updated exclusion
            refresh_ui()

            # Triggers a redrawing of the plots to exclude the selected data
            plot_data()

    def exclude_all_filtered(b):
        '''Excludes all valid chips, regardless of the active search filter.'''
        
        # Evaluates whether there are any valid chips currently in the dataset
        if state['valid_chips']:
            
            # Adds all valid chips to the excluded chips tracking set
            state['excluded_chips'].update(state['valid_chips'])
            
            # Refreshes the user interface to reflect the complete exclusion
            refresh_ui()

            # Triggers a redrawing of the plots to exclude all data
            plot_data()

    def readd_selected(b):
        '''Restores the currently highlighted chip from the excluded list back to the active pool.'''

        # Evaluates whether a valid chip is currently highlighted in the excluded list and exists in the tracking set
        if excluded_select.value and excluded_select.value in state['excluded_chips']:

            # Removes the highlighted chip from the excluded chips tracking set
            state['excluded_chips'].remove(excluded_select.value)

            # Refreshes the user interface to reflect the restoration
            refresh_ui()

            # Triggers a redrawing of the plots to include the restored data
            plot_data()

    def reset_exclusions(b):
        '''Clears all current exclusions and restores the default chip lists.'''

        # Evaluates whether any chips are currently excluded
        if state['excluded_chips']:

            # Clears the excluded chips tracking set
            state['excluded_chips'].clear()

            # Resets the available chips search input
            chip_search.value = ''

            # Resets the excluded chips search input
            excluded_search.value = ''

            # Refreshes the user interface to display default lists
            refresh_ui()

            # Triggers a redrawing of the plots to include all data
            plot_data()

    # Binds the exclusion function to the single exclude button click event
    exclude_btn.on_click(exclude_selected)

    # Binds the batch exclusion function to the exclude all button click event
    exclude_all_btn.on_click(exclude_all_filtered)

    # Binds the restoration function to the re-add button click event
    readd_btn.on_click(readd_selected)

    # Binds the reset function to the reset exclusions button click event
    reset_btn.on_click(reset_exclusions)

    # Groups the left, middle, and right panels into the final horizontal exclusion control box
    exclusion_controls = widgets.HBox([left_panel, button_panel, right_panel], layout=widgets.Layout(border='1px solid #ddd', padding='10px', margin='10px 0px', width='100%'))
    
    # Initialises the output widget for the primary scatter plot
    out_scatter = widgets.Output()

    # Initialises the output widget for the secondary distribution plots
    out_dist = widgets.Output()

    def update_sources(*args):
        '''Updates available data sources depending on the selected channel.'''

        # Evaluates whether an update operation is already in progress
        if state['updating']: 

            # Returns control to prevent recursive updates
            return 

        # Flags the state to indicate an active update operation
        state['updating'] = True

        # Extracts the currently selected channel value
        ch = channel_dropdown.value

        # Determines the valid sources based on whether the secondary channel is explicitly selected
        valid_sources = [s for s in all_sources if s != 'Standard Curve'] if ch == 'Channel 2' else all_sources
        
        # Extracts the current horizontal axis source value
        curr_x = x_source_drop.value

        # Extracts the current vertical axis source value
        curr_y = y_source_drop.value

        # Updates the available options for the horizontal axis source dropdown
        x_source_drop.options = valid_sources

        # Updates the available options for the vertical axis source dropdown
        y_source_drop.options = valid_sources

        # Assigns the previous horizontal source if valid, otherwise falls back to the first available source
        x_source_drop.value = curr_x if curr_x in valid_sources else valid_sources[0]

        # Assigns the previous vertical source if valid, otherwise falls back to the first available source
        y_source_drop.value = curr_y if curr_y in valid_sources else valid_sources[0]

        # Flags the state to indicate the completion of the update operation
        state['updating'] = False

        # Triggers a cascading update of the dependent dropdown widgets
        update_dropdowns()

    def update_dropdowns(*args):
        '''Refreshes feature dropdown options after the selected sources change, applying the PrG filter if active.'''

        # Evaluates whether an update operation is already in progress
        if state['updating']: 

            # Returns control to prevent recursive updates
            return 

        # Flags the state to indicate an active update operation
        state['updating'] = True

        # Extracts the currently selected analysis mode
        mode = mode_toggle.value

        # Extracts the currently selected channel value
        ch = channel_dropdown.value

        # Assigns a fallback channel for extracting column names if both channels are selected
        col_ch = 'Channel 1' if ch == 'Both' else ch
        
        # Extracts the currently selected horizontal axis source
        x_src = x_source_drop.value

        # Extracts the currently selected vertical axis source
        y_src = y_source_drop.value

        # Extracts the current state of the Protein G filter checkbox
        prg_only = prg_checkbox.value

        # Evaluates whether both horizontal and vertical sources are selected
        if x_src and y_src:

            # Extracts the available column names for the selected horizontal source
            x_cols = list(data_catalog[mode][x_src][col_ch].columns)

            # Extracts the available column names for the selected vertical source
            y_cols = list(data_catalog[mode][y_src][col_ch].columns)

            # Evaluates whether the Protein G filter is currently active
            if prg_only:

                # Evaluates whether the horizontal source relates to immobilisation data
                if 'Immob' in x_src:

                    # Filters the horizontal column names to include only Protein G stages
                    x_cols = [col for col in x_cols if str(col).startswith('PrG')]

                # Evaluates whether the vertical source relates to immobilisation data
                if 'Immob' in y_src:

                    # Filters the vertical column names to include only Protein G stages
                    y_cols = [col for col in y_cols if str(col).startswith('PrG')]

            # Extracts the current horizontal axis metric value
            curr_x_col = x_col_drop.value

            # Extracts the current vertical axis metric value
            curr_y_col = y_col_drop.value

            # Updates the available options for the horizontal axis metric dropdown
            x_col_drop.options = x_cols

            # Updates the available options for the vertical axis metric dropdown
            y_col_drop.options = y_cols

            # Assigns the previous horizontal metric if valid, otherwise falls back to the first available metric
            x_col_drop.value = curr_x_col if curr_x_col in x_cols else (x_cols[0] if x_cols else None)

            # Assigns the previous vertical metric if valid, otherwise falls back to the first available metric
            y_col_drop.value = curr_y_col if curr_y_col in y_cols else (y_cols[0] if y_cols else None)

        # Flags the state to indicate the completion of the update operation
        state['updating'] = False

        # Triggers a cascading update of the dynamic filter bounds
        update_filter_bounds()

    def update_filter_bounds(*args):
        '''Dynamically sets the absolute min/max limits ONLY for the axes that were just modified.'''

        # Evaluates whether an update operation is already in progress
        if state['updating']: 

            # Returns control to prevent recursive updates
            return 

        # Flags the state to indicate an active update operation
        state['updating'] = True

        # Extracts the currently selected analysis mode and channel
        mode, ch = mode_toggle.value, channel_dropdown.value

        # Assigns a fallback channel for extracting limits if both channels are selected
        col_ch = 'Channel 1' if ch == 'Both' else ch
        
        # Extracts the currently selected sources for both axes
        x_src, y_src = x_source_drop.value, y_source_drop.value

        # Extracts the currently selected metrics for both axes
        x_col, y_col = x_col_drop.value, y_col_drop.value

        # Determines whether the horizontal axis parameters have changed since the last update
        needs_x_update = (mode != state.get('last_mode') or ch != state.get('last_ch') or x_src != state.get('last_x_src') or x_col != state.get('last_x_col'))

        # Determines whether the vertical axis parameters have changed since the last update
        needs_y_update = (mode != state.get('last_mode') or ch != state.get('last_ch') or y_src != state.get('last_y_src') or y_col != state.get('last_y_col'))

        # Evaluates whether any relevant changes occurred subsequent to the initial load
        if (needs_x_update or needs_y_update) and state.get('last_mode') is not None:

            # Clears the excluded chips tracking set
            state['excluded_chips'].clear()

            # Resets the available chips search input
            chip_search.value = ''

            # Resets the excluded chips search input
            excluded_search.value = ''

        # Defines a large numerical constant to temporarily remove boundary limits
        LARGE_NUM = 1e30 

        # Evaluates whether the horizontal axis filters require updating
        if needs_x_update:

            # Displays the horizontal axis filter controls
            x_filters.layout.display = 'flex'

            # Checks whether a valid horizontal metric is selected
            if x_col:

                # Retrieves the target dataframe from the data catalog
                df_x = data_catalog[mode][x_src][col_ch]

                # Evaluates whether the target dataframe and metric column are valid
                if not df_x.empty and x_col in df_x.columns:

                    # Extracts the target horizontal data and drops any missing values
                    x_data = df_x[x_col].dropna()

                    # Evaluates whether the extracted horizontal data is populated
                    if not x_data.empty:

                        # Calculates the absolute minimum and maximum boundaries for the horizontal data
                        x_min, x_max = float(x_data.min()), float(x_data.max())

                        # Adjusts the maximum boundary slightly if it equals the minimum to prevent widget errors
                        if x_min == x_max: 
                            x_max += 1e-9

                        # Temporarily expands the minimum input bounds to accept the new values
                        x_min_input.min, x_max_input.max = -LARGE_NUM, LARGE_NUM

                        # Temporarily expands the maximum input bounds to accept the new values
                        x_max_input.min, x_min_input.max = -LARGE_NUM, LARGE_NUM
                        
                        # Assigns the calculated boundaries as the current input values
                        x_min_input.value, x_max_input.value = x_min, x_max
                        
                        # Restricts the minimum input bounds to the newly calculated limits
                        x_min_input.min, x_min_input.max = x_min, x_max

                        # Restricts the maximum input bounds to the newly calculated limits
                        x_max_input.min, x_max_input.max = x_min, x_max
                        
            # Records the updated horizontal source and metric into the state tracking dictionary
            state['last_x_src'], state['last_x_col'] = x_src, x_col

        # Evaluates whether the vertical axis filters require updating
        if needs_y_update:

            # Displays the vertical axis filter controls
            y_filters.layout.display = 'flex'

            # Checks whether a valid vertical metric is selected
            if y_col:

                # Retrieves the target dataframe from the data catalog
                df_y = data_catalog[mode][y_src][col_ch]

                # Evaluates whether the target dataframe and metric column are valid
                if not df_y.empty and y_col in df_y.columns:

                    # Extracts the target vertical data and drops any missing values
                    y_data = df_y[y_col].dropna()

                    # Evaluates whether the extracted vertical data is populated
                    if not y_data.empty:

                        # Calculates the absolute minimum and maximum boundaries for the vertical data
                        y_min, y_max = float(y_data.min()), float(y_data.max())

                        # Adjusts the maximum boundary slightly if it equals the minimum to prevent widget errors
                        if y_min == y_max: 
                            y_max += 1e-9

                        # Temporarily expands the minimum input bounds to accept the new values
                        y_min_input.min, y_max_input.max = -LARGE_NUM, LARGE_NUM

                        # Temporarily expands the maximum input bounds to accept the new values
                        y_max_input.min, y_min_input.max = -LARGE_NUM, LARGE_NUM
                        
                        # Assigns the calculated boundaries as the current input values
                        y_min_input.value, y_max_input.value = y_min, y_max
                        
                        # Restricts the minimum input bounds to the newly calculated limits
                        y_min_input.min, y_min_input.max = y_min, y_max

                        # Restricts the maximum input bounds to the newly calculated limits
                        y_max_input.min, y_max_input.max = y_min, y_max
                        
            # Records the updated vertical source and metric into the state tracking dictionary
            state['last_y_src'], state['last_y_col'] = y_src, y_col

        # Records the updated mode and channel into the state tracking dictionary
        state['last_mode'], state['last_ch'] = mode, ch

        # Flags the state to indicate the completion of the update operation
        state['updating'] = False
        
        # Triggers a redrawing of the plots reflecting the updated boundaries
        plot_data()

    def get_joined_data(mode, x_src, y_src, x_col, y_col, channel_label):
        '''Joins selected dashboard fields and applies an optional channel filter.'''

        # Retrieves a copy of the horizontal dataframe from the data catalog
        df_x = data_catalog[mode][x_src][channel_label].copy()

        # Retrieves a copy of the vertical dataframe from the data catalog
        df_y = data_catalog[mode][y_src][channel_label].copy()

        # Evaluates whether either extracted dataframe is empty
        if df_x.empty or df_y.empty:

            # Returns an empty dataframe indicating a joining failure
            return pd.DataFrame()

        # Casts the horizontal dataframe column names to strings to ensure consistent merging
        df_x.columns = df_x.columns.astype(str)

        # Casts the vertical dataframe column names to strings to ensure consistent merging
        df_y.columns = df_y.columns.astype(str)

        # Casts the targeted horizontal and vertical metric column names to strings
        x_col_str, y_col_str = str(x_col), str(y_col)

        # Merges the targeted columns from both dataframes into a single structure using an inner join
        df_merged = pd.merge(df_x[[x_col_str]], df_y[[y_col_str]], left_index=True, right_index=True, how='inner', suffixes=('_x', '_y'))
        
        # Identifies the actual horizontal column name depending on potential suffixing during the merge
        actual_x_col = x_col_str + '_x' if x_col_str == y_col_str else x_col_str

        # Identifies the actual vertical column name depending on potential suffixing during the merge
        actual_y_col = y_col_str + '_y' if x_col_str == y_col_str else y_col_str
        
        # Renames the merged columns to standard generic identifiers
        df_merged = df_merged.rename(columns={actual_x_col: 'x_val', actual_y_col: 'y_val'})

        # Replaces infinite values with missing representations and drops any incomplete rows
        return df_merged.replace([np.inf, -np.inf], np.nan).dropna()

    def plot_data(*args):
        '''Calculates data for and constructs both the scatter plot and distribution subplots.'''
        
        # Evaluates whether an update operation is already in progress
        if state['updating']: 

            # Returns control to prevent recursive drawing
            return

        # Extracts the currently selected analysis mode and channel
        mode, channel = mode_toggle.value, channel_dropdown.value

        # Extracts the currently selected sources for both axes
        x_src, y_src = x_source_drop.value, y_source_drop.value

        # Extracts the currently selected metrics for both axes
        x_col, y_col = x_col_drop.value, y_col_drop.value

        # Evaluates whether valid metrics are selected for both axes
        if not x_col or not y_col:

            # Clears the valid chips tracking list
            state['valid_chips'] = []

            # Refreshes the user interface to reflect the empty chip list
            refresh_ui()

            # Enters the context of the primary scatter plot widget
            with out_scatter:

                # Clears the existing scatter plot output
                out_scatter.clear_output(wait=True)

                # Prints an instructional message prompting valid metric selection
                print('Please select valid metrics to plot.')

            # Enters the context of the secondary distribution plots widget
            with out_dist:

                # Clears the existing distribution plots output
                out_dist.clear_output(wait=True)

            # Returns control to halt further plotting execution
            return

        # Determines the specific channels to plot based on the dropdown selection
        channels_to_plot = ['Channel 1', 'Channel 2'] if channel == 'Both' else [channel]

        # Defines a dictionary mapping distinct colours to each channel
        colors = {'Channel 1': '#1f77b4', 'Channel 2': '#ff1e0e'}
        
        # Initialises an empty set to track valid chips for the current selection
        current_valid_chips = set()

        # Initialises an empty dictionary to temporarily store raw plotted dataframes
        raw_dfs = {}

        # Initialises an empty dictionary to store the final filtered dataframes
        filtered_dfs = {}

        # Loops through each specific channel targeted for plotting
        for ch in channels_to_plot:

            # Wraps the data extraction process in a try block to handle potential missing sources
            try:

                # Triggers the joining function to retrieve the combined data for the current channel
                df = get_joined_data(mode, x_src, y_src, x_col, y_col, ch)

                # Checks whether the extracted joined dataframe contains data
                if not df.empty:

                    # Maps the extracted dataframe to its corresponding channel in the raw data dictionary
                    raw_dfs[ch] = df

                    # Appends the chip identifiers present in the dataframe to the valid chips tracking set
                    current_valid_chips.update(df.index.astype(str).tolist())

            # Catches exceptions thrown during data retrieval without breaking the loop
            except Exception as e:

                # Proceeds to the next iteration
                continue
        
        # Updates the state tracking list with the sorted valid chips
        state['valid_chips'] = sorted(list(current_valid_chips))

        # Cleans the excluded chips tracking set to remove identifiers no longer relevant to the selection
        state['excluded_chips'] = {c for c in state['excluded_chips'] if c in state['valid_chips']}

        # Refreshes the user interface to update the chip selection lists
        refresh_ui()

        # Initialises a boolean flag to track if any data is successfully plotted
        plotted_any = False

        # Loops through each channel targeted for plotting to apply active filters
        for ch in channels_to_plot:

            # Evaluates whether raw data was successfully compiled for the current channel
            if ch not in raw_dfs: 
                
                # Skips to the next channel iteration
                continue
            
            # Extracts the compiled raw dataframe for the current channel
            plot_df = raw_dfs[ch]

            # Initialises a boolean mask populated with true values corresponding to the dataframe index
            mask = pd.Series(True, index=plot_df.index)

            # Updates the mask to filter horizontal values within the specified boundaries
            mask &= (plot_df['x_val'] >= x_min_input.value) & (plot_df['x_val'] <= x_max_input.value)

            # Updates the mask to filter vertical values within the specified boundaries
            mask &= (plot_df['y_val'] >= y_min_input.value) & (plot_df['y_val'] <= y_max_input.value)

            # Evaluates whether any chips are currently selected for exclusion
            if state['excluded_chips']:

                # Updates the mask to remove rows corresponding to the excluded chip identifiers
                mask &= ~plot_df.index.astype(str).isin(state['excluded_chips'])

            # Applies the aggregated boolean mask to filter the plot dataframe
            plot_df = plot_df[mask]

            # Evaluates whether the filtered dataframe retains sufficient data points for visualisation
            if len(plot_df) >= 2: 

                # Maps the filtered dataframe to its corresponding channel in the output dictionary
                filtered_dfs[ch] = plot_df

                # Sets the plotting flag to true indicating valid data is present
                plotted_any = True

        # Checks whether the active filters removed all available data records
        if not plotted_any:

            # Enters the context of the primary scatter plot widget
            with out_scatter:

                # Clears the existing scatter plot output
                out_scatter.clear_output(wait=True)

                # Prints an instructional message indicating insufficient data
                print('Not enough matching chip records to plot these selections within the specified filter bounds.')

            # Enters the context of the secondary distribution plots widget
            with out_dist:

                # Clears the existing distribution plots output
                out_dist.clear_output(wait=True)

            # Returns control to halt further plotting execution
            return

        # Initialises an empty Plotly figure object for the primary scatter plot
        fig = go.Figure()

        # Loops through each filtered dataframe prepared for plotting
        for ch, plot_df in filtered_dfs.items():

            # Extracts the targeted horizontal and vertical arrays from the filtered dataframe
            x_data, y_data = plot_df['x_val'], plot_df['y_val']

            # Adds a scatter trace plotting the extracted data points onto the figure
            fig.add_trace(go.Scatter(
                x=x_data, y=y_data, mode='markers', name=f'{ch}', 
                marker=dict(size=8, opacity=0.7, color=colors[ch], line=dict(width=1, color='DarkSlateGrey')), 
                text=plot_df.index, hovertemplate='Chip ID: %{text}<br>X: %{x:.4f}<br>Y: %{y:.4f}<extra></extra>'
            ))

            # Evaluates whether the horizontal data contains sufficient variance to calculate a linear fit
            if x_data.nunique() > 1:

                # Calculates the linear regression parameters spanning the scatter points
                slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)

                # Generates a dense array of linearly spaced horizontal coordinates bounding the data
                x_fit = np.linspace(x_data.min(), x_data.max(), 100)

                # Calculates the predicted vertical coordinates using the derived linear model
                y_fit = slope * x_fit + intercept

                # Adds a dashed line trace representing the linear fit onto the figure
                fig.add_trace(go.Scatter(
                    x=x_fit, y=y_fit, mode='lines', 
                    name=f'{ch} Fit (r={r_value:.3f}, R^2={r_value**2:.3f})', 
                    line=dict(color=colors[ch], dash='dash', width=2), hoverinfo='skip'
                ))

        # Constructs the title string incorporating the selected sources and metrics
        title = f'{y_src} [{y_col}] vs {x_src} [{x_col}]'

        # Updates the comprehensive layout and styling parameters of the scatter figure
        fig.update_layout(title=title, xaxis_title=f'{x_src} : {x_col}', yaxis_title=f'{y_src} : {y_col}', template='plotly_white', height=600, margin=dict(l=40, r=40, t=60, b=40), hovermode='closest')
        
        # Initialises a complex Plotly subplots figure for the secondary distributions
        dist_fig = make_subplots(
            rows=2, cols=2, 
            row_heights=[0.2, 0.8], 
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=(f'{x_col} Distribution', f'{y_col} Distribution', '', '')
        )

        # Extracts the histogram bin counts designated by the slider widgets
        x_bins = x_bins_slider.value
        y_bins = y_bins_slider.value

        # Loops through each filtered dataframe prepared for plotting
        for ch, df in filtered_dfs.items():

            # Adds a horizontal box plot for the primary axis distribution into the first subplot row
            dist_fig.add_trace(go.Box(x=df['x_val'], name=ch, marker_color=colors[ch], showlegend=False), row=1, col=1)

            # Adds a vertical box plot for the secondary axis distribution into the first subplot row
            dist_fig.add_trace(go.Box(x=df['y_val'], name=ch, marker_color=colors[ch], showlegend=False), row=1, col=2)
            
            # Adds a horizontal histogram representing the primary axis density into the second subplot row
            dist_fig.add_trace(go.Histogram(x=df['x_val'], nbinsx=x_bins, name=ch, marker_color=colors[ch], opacity=0.7, showlegend=False), row=2, col=1)

            # Adds a vertical histogram representing the secondary axis density into the second subplot row
            dist_fig.add_trace(go.Histogram(x=df['y_val'], nbinsx=y_bins, name=ch, marker_color=colors[ch], opacity=0.7, showlegend=False), row=2, col=2)

        # Updates the comprehensive layout and styling parameters of the distribution figure
        dist_fig.update_layout(barmode='overlay', template='plotly_white', height=400, margin=dict(l=40, r=40, t=40, b=40))

        # Enters the context of the primary scatter plot widget
        with out_scatter:

            # Clears the existing scatter plot output
            out_scatter.clear_output(wait=True)

            # Renders the newly generated scatter figure
            fig.show()

        # Enters the context of the secondary distribution plots widget
        with out_dist:

            # Clears the existing distribution plots output
            out_dist.clear_output(wait=True)

            # Renders the newly generated distribution figure
            dist_fig.show()

    # Binds the source update function to the channel dropdown value changes
    channel_dropdown.observe(update_sources, 'value')

    # Binds the dropdown update function to the analysis mode toggle changes
    mode_toggle.observe(update_dropdowns, 'value')

    # Binds the dropdown update function to the Protein G filter checkbox changes
    prg_checkbox.observe(update_dropdowns, 'value')

    # Binds the dropdown update function to the horizontal axis source dropdown changes
    x_source_drop.observe(update_dropdowns, 'value')

    # Binds the dropdown update function to the vertical axis source dropdown changes
    y_source_drop.observe(update_dropdowns, 'value')
    
    # Binds the boundary update function to the horizontal axis metric dropdown changes
    x_col_drop.observe(update_filter_bounds, 'value')

    # Binds the boundary update function to the vertical axis metric dropdown changes
    y_col_drop.observe(update_filter_bounds, 'value')
    
    # Binds the plot redrawing function to the horizontal minimum boundary changes
    x_min_input.observe(plot_data, 'value')

    # Binds the plot redrawing function to the horizontal maximum boundary changes
    x_max_input.observe(plot_data, 'value')

    # Binds the plot redrawing function to the vertical minimum boundary changes
    y_min_input.observe(plot_data, 'value')

    # Binds the plot redrawing function to the vertical maximum boundary changes
    y_max_input.observe(plot_data, 'value')
    
    # Binds the plot redrawing function to the horizontal and vertical histogram bins slider changes
    x_bins_slider.observe(plot_data, 'value')
    y_bins_slider.observe(plot_data, 'value')

    # Groups all upper interface control elements into a master vertical container
    controls = widgets.VBox([
        top_controls, 
        channel_dropdown, 
        widgets.HBox([x_source_drop, x_col_drop]), 
        x_filters, 
        widgets.HBox([y_source_drop, y_col_drop]), 
        y_filters, 
        exclusion_controls
    ])
    
    # Wraps the histogram bins sliders within an aligned horizontal box for layout styling
    bins_container = widgets.HBox([x_bins_slider, y_bins_slider], layout=widgets.Layout(padding='10px', justify_content='space-around'))

    # Displays the dashboard title header with defined styling
    display(widgets.HTML(f"<h3 style='margin-bottom:0px; color:#0000FF;'>Master Chip Comparison Dashboard</h3>"))

    # Renders the sequential layout of control containers and plot outputs
    display(controls, out_scatter, bins_container, out_dist)
    
    # Triggers the initial cascading source update to populate dropdowns and render the first plot
    update_sources()
    
def get_top_drivers(df, chip_id, outliers=None, top_n=3):
    '''Identifies the features that most strongly distinguish a chip from its peers.

    Args:
        df (pd.DataFrame): Feature table indexed by chip identifier.
        chip_id (str): Identifier of the chip to investigate.
        outliers (list | None, optional): List of known outlier chip identifiers. Defaults to None.
        top_n (int, optional): Number of highest-impact features to return. Defaults to 3.

    Returns:
        list[str]: Ranked absolute z-scores formatted as strings for the selected chip.
    '''

    # Skips evaluation if the chip ID is not found in the standard index
    if chip_id not in df.index: 

        # Returns a not-applicable string message within a list
        return ['N/A (Chip not in dataset)']
    
    # Evaluates whether an explicit list of outliers is provided to isolate a clean baseline
    if outliers is not None:
        
        # Identifies the healthy baseline chips by excluding the known outliers from the index
        normals = df.index.difference(outliers)
        
        # Evaluates whether there are enough normal chips to form a reliable baseline
        if len(normals) >= 2:

            # Calculates the baseline mean using only the healthy normal chips
            baseline_mean = df.loc[normals].mean()

            # Calculates the baseline standard deviation, replacing zeros to prevent division errors
            baseline_std = df.loc[normals].std().replace(0, 1e-9)
            
        # Handles cases where the normal baseline is too small for standard statistics
        else:

            # Calculates the robust baseline mean using the median of the entire dataframe
            baseline_mean = df.median()

            # Calculates the robust baseline standard deviation using the median absolute deviation
            baseline_std = ((df - baseline_mean).abs().median() * 1.4826).replace(0, 1e-9)

    # Handles cases where no outlier list is provided
    else:

        # Calculates the robust baseline mean using the median of the entire dataframe
        baseline_mean = df.median()

        # Calculates the robust baseline standard deviation using the median absolute deviation
        baseline_std = ((df - baseline_mean).abs().median() * 1.4826).replace(0, 1e-9)

    # Calculates the absolute z-scores for the selected chip against the derived baseline and drops missing values
    z_scores = ((df.loc[chip_id] - baseline_mean) / baseline_std).abs().dropna()

    # Evaluates whether the calculated z-scores series contains data
    if z_scores.empty: 

        # Returns a standard not-applicable string within a list
        return ['N/A']
    
    # Sorts the z-scores in descending order and extracts the highest impact stages based on the threshold
    top_stages = z_scores.sort_values(ascending=False).head(top_n)

    # Returns the formatted top drivers as a list of strings mapping the stage to its z-score
    return [f'{stage} (Z: {z:.1f})' for stage, z in top_stages.items()]

def prepare_ensemble_section(events_df, changes_df, intra_df, val_col, change_col):
    '''Pivots all three data types, detects anomalies, and calculates true ensemble failures.

    Args:
        events_df (pd.DataFrame): The dataframe containing event features.
        changes_df (pd.DataFrame): The dataframe containing transition features.
        intra_df (pd.DataFrame): The dataframe containing intra-stage features.
        val_col (str): The column name representing absolute values.
        change_col (str): The column name representing delta transitions.

    Returns:
        tuple: A dictionary of the three wide dataframes and the ensemble outlier IDs.
    '''

    # Filters out non-informative startup stages from the events dataframe
    filtered_events = events_df[~events_df['stage'].isin(['Start', 'Initial'])]

    # Filters out non-informative startup stages from the changes dataframe
    filtered_changes = changes_df[~changes_df['stage'].isin(['Start', 'Initial'])]

    # Filters out non-informative startup stages from the intra-stage dataframe
    filtered_intra = intra_df[~intra_df['stage'].isin(['Start', 'Initial'])]

    # Pivots the filtered events dataframe to prepare for absolute signal anomaly detection
    wide_abs = pivot_chip_data(filtered_events, 'stage', val_col)

    # Pivots the filtered changes dataframe to prepare for stage delta anomaly detection
    wide_delta = pivot_chip_data(filtered_changes, 'stage', change_col)

    # Pivots the filtered intra-stage dataframe to prepare for kinetic anomaly detection
    wide_intra = create_wide_intra(filtered_intra, val_col)

    # Detects absolute signal anomalies and extracts their corresponding flags
    _, flags_abs = detect_anomalies(wide_abs, wide_abs.columns.tolist())

    # Detects stage delta anomalies and extracts their corresponding flags
    _, flags_delta = detect_anomalies(wide_delta, wide_delta.columns.tolist())

    # Detects intra-stage kinetic anomalies and extracts their corresponding flags
    _, flags_intra = detect_anomalies(wide_intra, wide_intra.columns.tolist())

    # Calculates and extracts the ensemble failures based on strict stage-overlap logic
    ensemble_outliers = get_ensemble_failures(flags_abs, flags_delta, flags_intra)

    # Packs the wide dataframes into a dictionary for granular downstream reporting
    dfs = {
        'Absolute': wide_abs,
        'Stage Delta': wide_delta,
        'Intra-stage Kinetics': wide_intra
    }

    # Returns the dictionary of datasets alongside the list of ensemble outliers
    return dfs, ensemble_outliers

def generate_at_risk_summary(master_df, sc_metrics, sc_raw_df, all_at_risk_chips, section_config, incomplete_chips=None, title='Overall At-Risk Chips Summary'):
    '''Clusters chips based on combined stage data, cross-references against standard curve metrics, and generates a structured summary report across 5 distinct sections.

    Args:
        master_df (pd.DataFrame): The master dataframe containing all features.
        sc_metrics (pd.DataFrame): Standard curve metrics.
        sc_raw_df (pd.DataFrame): Raw standard curve data.
        all_at_risk_chips (list): List of all chips identified as at-risk.
        section_config (dict): Configuration mapping for different summary sections.
        incomplete_chips (set | list, optional): Chips identified as having incomplete datasets.
        title (str, optional): The title for the summary. Defaults to 'Overall At-Risk Chips Summary'.
    '''

    # Evaluates whether the incomplete chips tracking variable is unassigned
    if incomplete_chips is None:

        # Initialises the incomplete chips as an empty set
        incomplete_chips = set()

    else:

        # Casts the provided incomplete chips to a set to ensure uniqueness
        incomplete_chips = set(incomplete_chips)

    # Initialises an empty set to track unique PEL anomalies
    pel_unique = set()

    # Initialises an empty dictionary to store the PEL breakdown mapping
    pel_breakdown = {}
    
    # Initialises an empty set to track unique immobilisation anomalies
    immob_unique = set()

    # Initialises an empty dictionary to store the immobilisation breakdown mapping
    immob_breakdown = {}
    
    # Loops through each section configuration to aggregate the targeted anomalies
    for section_name, (_, outliers) in section_config.items():

        # Checks if the current section relates specifically to PEL analysis
        if 'PEL' in section_name:

            # Adds the extracted outliers to the unique PEL tracking set
            pel_unique.update(outliers)

            # Evaluates whether any outliers were found in the current PEL section
            if outliers: 

                # Maps the identified outliers to their respective PEL section name
                pel_breakdown[section_name] = outliers
                
        # Checks if the current section relates specifically to immobilisation analysis
        elif 'Immob' in section_name:

            # Adds the extracted outliers to the unique immobilisation tracking set
            immob_unique.update(outliers)

            # Evaluates whether any outliers were found in the current immobilisation section
            if outliers: 

                # Maps the identified outliers to their respective immobilisation section name
                immob_breakdown[section_name] = outliers

    # Initialises an empty set to track unique standard curve failures
    sc_fails = set()

    # Checks whether the standard curve metrics dataframe contains valid R-squared data
    if sc_metrics is not None and not sc_metrics.empty and 'r2' in sc_metrics.columns:

        # Identifies and extracts chip IDs that fall below the R-squared pass threshold
        sc_fails = set(sc_metrics[sc_metrics['r2'] < 0.95].index)
        
    # Combines and sorts all unique faulty chip IDs across all analysis stages
    combined_all_fails = sorted(list(set(all_at_risk_chips).union(pel_unique, immob_unique, sc_fails)))

    # Displays the overall summary results within a collapsible output section
    with collapsible_output(f'{title} - Overall Summary'):

        # Prints the total count of unique faulty chips identified
        print(f'Total Unique Faulty Chips: {len(combined_all_fails)}')

        # Checks whether any faulty chips were identified across the entire dataset
        if combined_all_fails:

            # Prints the compiled list of faulty chip IDs
            print(f"Chip IDs: {', '.join(map(str, combined_all_fails))}\n")

            # Prints the total count of unique PEL failures
            print(f'Total Unique PEL Fails: {len(pel_unique)}')

            # Prints the total count of unique immobilisation failures
            print(f'Total Unique Immobilisation Fails: {len(immob_unique)}')

            # Prints the total count of unique standard curve failures
            print(f'Total Unique Standard Curve Fails: {len(sc_fails)}')

            # Prints the total count of chips flagged with incomplete datasets
            print(f'Total Unique Chips with Incomplete Datasets: {len(incomplete_chips)}')
            
            # Checks whether there are any specific incomplete chips to list
            if incomplete_chips:

                # Prints the sorted list of incomplete chip IDs
                print(f"  - Incomplete Chip IDs: {', '.join(map(str, sorted(list(incomplete_chips))))}")

        else:

            # Prints a confirmation message indicating normal overall behaviour
            print('No faults detected across any stage. Everything looks normal!')

    # Displays the PEL breakdown results within a collapsible output section
    with collapsible_output(f'{title} - PEL Breakdown'):

        # Prints the total count of unique PEL faulty chips
        print(f'Total Unique PEL Faulty Chips: {len(pel_unique)}')

        # Prints the set of unique PEL anomalous chips
        print(pel_unique, '\n')

        # Evaluates whether any unique PEL anomalous chips exist
        if pel_unique:

            # Loops through the PEL breakdown dictionary to print specific section triggers
            for sec_name, out_list in pel_breakdown.items():

                # Prints the section name and the chips that triggered it
                print(f"  - {sec_name}: {len(out_list)} chips ({', '.join(map(str, out_list))})")

        else:

            # Prints a message confirming no anomalies were detected in the PEL stage
            print('No anomalies detected in the PEL stage.')

    # Displays the immobilisation breakdown results within a collapsible output section
    with collapsible_output(f'{title} - Immobilisation Breakdown'):

        # Prints the total count of unique immobilisation faulty chips
        print(f'Total Unique Immobilisation Faulty Chips: {len(immob_unique)}')

        # Prints the set of unique immobilisation anomalous chips
        print(immob_unique, '\n')

        # Evaluates whether any unique immobilisation anomalous chips exist
        if immob_unique:

            # Loops through the immobilisation breakdown dictionary to print specific section triggers
            for sec_name, out_list in immob_breakdown.items():

                # Prints the section name and the chips that triggered it
                print(f"  - {sec_name}: {len(out_list)} chips ({', '.join(map(str, out_list))})")

        else:

            # Prints a message confirming no anomalies were detected in the immobilisation stage
            print('No anomalies detected in the Immobilisation stage.')

    # Displays the standard curve breakdown results within a collapsible output section
    with collapsible_output(f'{title} - Standard Curve Breakdown'):

        # Prints the total count of unique standard curve faulty chips
        print(f'Total Unique Standard Curve Faulty Chips: {len(sc_fails)}')

        # Checks whether any standard curve failures were detected
        if sc_fails: 

            # Prints the sorted list of standard curve failure chip IDs
            print(f"Chip IDs (R^2 < 0.95): {', '.join(map(str, sorted(list(sc_fails))))}")

        else:

            # Prints a message confirming all standard curve scores met the required threshold
            print('No Standard Curve anomalies detected (All R² >= 0.95).')

    # Displays the most anomalous stages per dataset within a collapsible output section
    with collapsible_output(f'{title} - Top Anomalous Stages Overall'):

        # Prints a brief header explaining the ranking metric
        print('Top 5 Anomalous Stages (Ranked by Average Z-Score among Outliers):\n')
        
        # Loops through each configuration section to evaluate global anomalies
        for section_name, (dfs, outlier_list) in section_config.items():

            # Prints the specific section header
            print(f'--- {section_name} ---')
            
            # Evaluates whether any outliers were flagged in the current section
            if not outlier_list:

                # Prints a message confirming no anomalies exist for this section
                print('  No outliers detected in this section.\n')
                continue
            
            # Loops through each specific dataset (Absolute, Stage Delta, Kinetics)
            for ds_name, df_source in dfs.items():

                # Filters the outlier list to ensure the chips exist in the dataframe index
                valid_outliers = [chip for chip in outlier_list if chip in df_source.index]
                
                # Evaluates whether valid outlier data is available to process
                if not valid_outliers:

                    # Prints a message indicating missing data for the dataset
                    print(f'  [{ds_name}] No valid outlier data available.\n')

                    continue

                # Identifies the healthy baseline chips by excluding the valid outliers
                normals = df_source.index.difference(valid_outliers)
                
                # Ensures there is a sufficient baseline for comparison
                if len(normals) < 2:
                    
                    # Prints a message indicating insufficient baseline data
                    print(f'  [{ds_name}] Not enough normal chips to form a baseline.\n')

                    continue

                # Calculates the statistical mean exclusively on the healthy baseline
                normal_mean = df_source.loc[normals].mean()

                # Calculates the standard deviation exclusively on the healthy baseline, replacing zeros
                normal_std = df_source.loc[normals].std().replace(0, 1e-9)
                
                # Calculates Z-scores for the outliers against the isolated healthy baseline
                z_scores = ((df_source.loc[valid_outliers] - normal_mean) / normal_std).abs()

                # Calculates the average, minimum, and maximum z-score across the outliers for each stage
                avg_z_scores = z_scores.mean(axis=0).dropna()
                min_z_scores = z_scores.min(axis=0)
                max_z_scores = z_scores.max(axis=0)
                
                # Checks whether the calculated averages contain data
                if avg_z_scores.empty:

                    # Prints a standard not-applicable message
                    print(f'  [{ds_name}] N/A\n')
                    continue
                    
                # Sorts the average z-scores in descending order and extracts the top 5 highest impact stages
                top_5_stages = avg_z_scores.sort_values(ascending=False).head(5)
                
                # Prints the dataset sub-header
                print(f'  [{ds_name}] Top Drivers:')

                # Loops through the top stages to format and print their rankings
                for rank, (stage, z_avg) in enumerate(top_5_stages.items(), 1):
                    
                    # Extracts the minimum and maximum Z-scores for the current stage
                    z_min = min_z_scores[stage]
                    z_max = max_z_scores[stage]
                    
                    # Prints the formatted ranking including average, minimum, and maximum Z-scores
                    print(f'    {rank}. {stage} (Avg Z: {z_avg:.1f} | Min: {z_min:.1f} | Max: {z_max:.1f})')
                
                # Prints an empty line for visual spacing between datasets
                print('')

    # Displays the detailed chip profiles within a collapsible output section
    with collapsible_output(f'{title} - Chip Profiles'):

        # Checks whether the master dataframe is empty or if no combined failures exist
        if master_df.empty or not combined_all_fails:

            # Prints a message indicating detailed profiles cannot be generated
            print('No detailed profiles available.')
            
            # Returns control to the calling function
            return

        # Extracts the list of feature columns from the master dataframe
        features = [col for col in master_df.columns if col != 'Cluster']

        # Creates a boolean mask identifying rows with complete feature data
        valid_rows = master_df[features].notna().all(axis=1)

        # Initialises a new column to store cluster assignments with missing values
        master_df['Cluster'] = pd.NA

        # Checks if there are enough valid rows to perform clustering
        if valid_rows.sum() >= 2:

            # Standardises the valid feature data using a robust scaler
            X_master = RobustScaler().fit_transform(master_df.loc[valid_rows, features])
        
            # Initialises the KMeans clustering model with two target contextual clusters
            kmeans = KMeans(n_clusters=2, random_state=8030, n_init=10)

            # Fits the KMeans model and assigns the predicted cluster labels to the valid rows
            master_df.loc[valid_rows, 'Cluster'] = kmeans.fit_predict(X_master)

        # Attempts to dynamically identify the date column from the raw standard curve records
        date_col = next((c for c in sc_raw_df.columns if 'date' in c.lower()), None)

        # Attempts to dynamically identify the time column from the raw standard curve records
        time_col = next((c for c in sc_raw_df.columns if 'time' in c.lower()), None)

        # Initialises a list of columns required for merging raw records
        cols_to_merge = ['standard_curve_uuid']

        # Checks if a valid date column was found
        if date_col: 

            # Appends the identified date column to the merging list
            cols_to_merge.append(date_col)

        # Checks if a valid time column was found
        if time_col:

            # Appends the identified time column to the merging list
            cols_to_merge.append(time_col)

        # Initialises an empty dictionary to construct individual chip summaries
        chip_summaries = {}

        # Loops through each identified faulty chip to build its detailed profile
        for chip in combined_all_fails:
            
            # Evaluates if the current chip exists in the master dataframe index
            if chip in master_df.index.get_level_values(master_df.index.name or 'chip_id'):
                
                # Checks if the master dataframe utilises a multi-level index
                if isinstance(master_df.index, pd.MultiIndex):

                    # Extracts the cluster assignments mapped to the chip across multiple channels
                    chip_clusters = master_df.xs(chip, level='chip_id')['Cluster'].to_dict()

                    # Formats the multi-channel cluster assignments into a descriptive string handling missing values
                    cluster_profile = ' | '.join([f"{ch}: {'N/A' if pd.isna(cl) else f'Cluster {int(cl)}'}" for ch, cl in chip_clusters.items()])
                
                # Proceeds if the master dataframe uses a standard flat index
                else:

                    # Extracts the specific cluster assignment for the chip
                    cl = master_df.loc[chip, 'Cluster']

                    # Checks if the assigned cluster is missing due to incomplete data
                    if pd.isna(cl):

                        # Assigns an incomplete data label to the cluster profile
                        cluster_profile = 'N/A (Incomplete Stage Data)'

                    else:

                        # Formats the valid cluster assignment into a descriptive string
                        cluster_profile = f'Cluster {int(cl)}'
                    
            # Proceeds if the chip is entirely missing from the master dataframe index
            else:

                # Assigns an incomplete data label to the cluster profile
                cluster_profile = 'N/A (Incomplete Stage Data)'
                
            # Extracts a copy of the standard curve metrics corresponding to the current chip
            sc_rows = sc_metrics[sc_metrics.index == chip].copy()

            # Checks whether the standard curve data contains records and requires date merging
            if not sc_rows.empty and len(cols_to_merge) > 1:

                # Merges the raw date and time records into the standard curve metrics
                sc_rows = sc_rows.merge(sc_raw_df[cols_to_merge], on='standard_curve_uuid', how='left')
            
            # Initialises an empty list to track specific anomaly flags for the chip
            detailed_flags = []

            # Evaluates if any of the chip's standard curve scores fall below the pass threshold
            if not sc_rows.empty and (sc_rows['r2'] < 0.95).any():

                # Appends the standard curve failure flag to the tracking list
                detailed_flags.append('Standard Curve (R² < 0.95)')
                
            # Initialises an empty dictionary to collect the driving metrics for each triggered section
            drivers_info = {}

            # Loops through each section configuration to aggregate the driving metrics
            for section_name, (dfs, outlier_list) in section_config.items():

                # Checks if the current chip is flagged within the specific section
                if chip in outlier_list:

                    # Appends the triggered section name to the detailed flags list
                    detailed_flags.append(section_name)

                    # Initialises a nested dictionary for the dataset metrics
                    drivers_info[section_name] = {}

                    # Loops through each dataset to extract the top distinct drivers
                    for ds_name, df_source in dfs.items():
                        drivers_info[section_name][ds_name] = get_top_drivers(df_source, chip, outliers=outlier_list)
            
            # Initialises an empty list to assemble the formatted standard curve data
            sc_data = []

            # Checks whether the extracted standard curve rows contain data
            if not sc_rows.empty:

                # Loops through each standard curve record associated with the chip
                for _, sc_row in sc_rows.iterrows():

                    # Extracts the R-squared score from the current record
                    r2_val = sc_row['r2']

                    # Formats the date string if it exists and is valid
                    d_str = str(sc_row[date_col]).strip() if date_col and pd.notna(sc_row[date_col]) else ''

                    # Formats the time string by replacing hyphens with colons if it exists
                    t_str = str(sc_row[time_col]).strip().replace('-', ':') if time_col and pd.notna(sc_row[time_col]) else ''
                    
                    # Appends the formatted metrics dictionary to the standard curve summary list
                    sc_data.append({
                        'Date/Time': f'{d_str} {t_str}'.strip() or 'Unknown',
                        'Curve UUID': sc_row['standard_curve_uuid'],
                        'Fit Model': sc_row['fit_model'],
                        'R² Score': r2_val,
                        'Status': 'FAILED' if r2_val < 0.95 else 'PASSED'
                    })

            # Converts the assembled standard curve data list into a pandas dataframe
            sc_df = pd.DataFrame(sc_data)
            
            # Checks whether the generated dataframe contains records and a datetime column
            if not sc_df.empty and 'Date/Time' in sc_df.columns:

                # Parses the combined date and time strings into datetime objects for chronological sorting
                sc_df['Parsed_DateTime'] = pd.to_datetime(sc_df['Date/Time'], errors='coerce')

                # Sorts the dataframe chronologically, drops the temporary sorting column, and resets the index
                sc_df = sc_df.sort_values(by='Parsed_DateTime').drop(columns=['Parsed_DateTime']).reset_index(drop=True)
            
            # Maps the fully assembled profile dictionary to the chip ID in the master summary
            chip_summaries[chip] = {
                'Cluster': cluster_profile,
                'Flagged By': detailed_flags,
                'Drivers': drivers_info,
                'SC_DF': sc_df
            }

        def style_status(val):
            '''Returns a CSS style declaration for a standard-curve status value.

            Args:
                val (str): Status value to evaluate.

            Returns:
                str: CSS declaration appropriate for the status.
            '''
        
            # Checks if the evaluated status is marked as failed
            if val == 'FAILED': 

                # Returns a bold red CSS style declaration
                return 'color: red; font-weight: bold;'

            # Checks if the evaluated status is marked as passed
            if val == 'PASSED': 

                # Returns a green CSS style declaration
                return 'color: green;'

            # Returns an empty string for unrecognised status values
            return ''
        
        # Loops through each constructed chip profile to render the output
        for chip, info in chip_summaries.items():

            # Prints a visual separator to distinguish individual chip profiles
            print('=' * 80)

            # Prints the chip identifier header
            print(f'CHIP ID: {chip}')

            # Prints the mapped cluster profile
            print(f"Cluster Profile: {info['Cluster']}")
            
            # Formats the list of triggered flags into a string or returns a manual review label
            flag_str = '\n    - '.join(info['Flagged By']) if info['Flagged By'] else 'None (Manual Review)'

            # Prints the formatted sources flagging this chip
            print(f'Flagged By Anomaly In:\n    - {flag_str}')
            
            # Checks if any driving metrics were recorded for the chip
            if info['Drivers']:

                # Prints the header for divergent regions
                print('\nTop 3 Divergent Regions for Triggered Sections:')

                # Loops through each triggered section and its associated datasets
                for section, ds_drivers in info['Drivers'].items():

                    # Prints the specific section header
                    print(f'\n  - {section}:')

                    # Loops through each specific dataset (Absolute, Delta, Kinetics)
                    for ds_name, drivers in ds_drivers.items():
                        
                        # Prints the dataset sub-header
                        print(f'    [{ds_name}]')

                        # Loops through each specific driving metric
                        for d in drivers: 

                            # Prints the individual driving metric
                            print(f'      {d}')

            # Prints a trailing visual separator to complete the profile block
            print('-' * 80)
            
            # Checks whether the formatted standard curve dataframe contains data
            if not info['SC_DF'].empty:

                # Applies specific formatting rules and CSS styles to the standard curve dataframe
                styled_df = info['SC_DF'].style.format({'R² Score': '{:.4f}'}).set_properties(**{'text-align': 'left', 'white-space': 'nowrap'}).set_table_styles([dict(selector='th', props=[('text-align', 'left')])]).map(style_status, subset=['Status'])

                # Renders the styled dataframe in the notebook output
                display(styled_df)

            else:

                # Prints a missing data message for the standard curve section
                print('No Standard Curve data available for this chip.')
                
            # Prints a newline for visual spacing between chips
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

    # Calculates and returns the predicted association response based on the input parameters
    return R0 + RI + Req * (1 - np.exp(-kobs * (t - t[0])))

def dissoc_model(t, R0, kd, Rinf, RI):
    '''Calculates Dissociation with Bulk Shift (RI) dropping off.

    Args:
        t (array-like): Time values.
        R0, kd, Rinf, RI (float): Dissociation model parameters.

    Returns:
        array-like: Predicted dissociation response.
    '''

    # Calculates and returns the predicted dissociation response based on the input parameters
    return Rinf + (R0 - Rinf - RI) * np.exp(-kd * (t - t[0]))

def fit_association(t, y):
    '''Fits the association-phase model to a time-series response.

    Args:
        t (array-like): Association-phase time values.
        y (array-like): Measured response values.

    Returns:
        tuple | None: Fitted parameters and covariance, or None.
    '''

    # Estimates the initial response value from the first data point
    R0_guess = y[0]
    
    # Estimates the equilibrium response value by calculating the total shift
    Req_guess = y[-1] - y[0]

    # Attempts to fit the association model to the provided time-series data
    try:
        
        # Defines the lower bounds for the curve fit parameters
        lower_bounds = [-np.inf, 0.0, -np.inf, -np.inf]
        
        # Defines the upper bounds for the curve fit parameters
        upper_bounds = [np.inf, np.inf, np.inf, np.inf]

        # Executes the curve fitting algorithm with the initial guesses and bounds
        popt, _ = curve_fit(assoc_model, t, y, p0=[Req_guess, 0.01, R0_guess, 0.0], bounds=(lower_bounds, upper_bounds), maxfev=10000)

        # Returns the optimised parameters mapped from the fit function
        return popt

    # Catches runtime errors if optimal parameters cannot be found
    except RuntimeError:

        # Returns None indicating the curve fit failed
        return None

def fit_dissociation(t, y):
    '''Fits the dissociation-phase model to a time-series response.

    Args:
        t (array-like): Dissociation-phase time values.
        y (array-like): Measured response values.

    Returns:
        tuple | None: Fitted parameters and covariance, or None.
    '''

    # Estimates the initial response value for dissociation from the first data point
    R0_guess = y[0]
    
    # Estimates the infinite response value from the final data point
    Rinf_guess = y[-1]

    # Attempts to fit the dissociation model to the provided time-series data
    try:
        
        # Defines the lower bounds for the curve fit parameters
        lower_bounds = [-np.inf, 0.0, -np.inf, -np.inf]
        
        # Defines the upper bounds for the curve fit parameters
        upper_bounds = [np.inf, np.inf, np.inf, np.inf]

        # Executes the curve fitting algorithm with the initial guesses and bounds
        popt, _ = curve_fit(dissoc_model, t, y, p0=[R0_guess, 0.01, Rinf_guess, 0.0], bounds=(lower_bounds, upper_bounds), maxfev=10000)

        # Returns the optimised parameters mapped from the fit function
        return popt

    # Catches runtime errors if optimal parameters cannot be found
    except RuntimeError:

        # Returns None indicating the curve fit failed
        return None
    
def plot_kinetic_curves(data, title):
    '''Plots fitted association and dissociation curves for one experiment, grouped by measurement for interactive toggling.

    Args:
        data (dict): Kinetic results containing fits and measurement data.
        title (str): The title for the generated plots.
    '''

    # Initialises the Matplotlib figure and axes with specific dimensions
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Initialises the Plotly figure for interactive plotting
    fig_plotly = go.Figure()
    
    # Defines the colour map for the plotted curves
    cmap = plt.cm.tab20
    
    # Loops through each index and fit dictionary in the data
    for i, f in enumerate(data['fits']):

        # Assigns a specific colour from the colour map based on the current index
        colour = cmap(i % 20)

        # Converts the mapped colour to a hexadecimal string for Plotly rendering
        hex_colour = mcolors.to_hex(colour)
        
        # Extracts the association zone from the fit dictionary
        assoc_zone = f['assoc_zone']

        # Extracts the dissociation zone from the fit dictionary
        dissoc_zone = f['dissoc_zone']

        # Skips the current iteration if the association zone contains no data
        if assoc_zone.empty:
            continue

        # Identifies the start time of the association zone
        t0 = assoc_zone['time'].iloc[0]
        
        # Checks whether the dissociation zone contains data
        if not dissoc_zone.empty:

            # Concatenates and normalises the time arrays relative to the start time
            t_raw = pd.concat([assoc_zone['time'], dissoc_zone['time']]) - t0

            # Concatenates the response arrays for the complete raw trace
            y_raw = pd.concat([assoc_zone['response'], dissoc_zone['response']])

        else:

            # Normalises the association time array relative to the start time
            t_raw = assoc_zone['time'] - t0

            # Extracts the association response array
            y_raw = assoc_zone['response']
            
        # Defines a unique legend group for this specific measurement to bind the raw data and fits together
        group_id = str(f['meas_id'])
            
        # Adds the raw data trace to the Plotly figure
        fig_plotly.add_trace(go.Scatter(x=t_raw, y=y_raw, mode='lines', line=dict(color=hex_colour, width=1.5), name=f['meas_id'], legendgroup=group_id))
        
        # Plots the raw association data on the Matplotlib axes
        ax.plot(assoc_zone['time'] - t0, assoc_zone['response'], color=colour, lw=1.5)

        # Evaluates whether the dissociation zone contains data
        if not dissoc_zone.empty:

            # Plots the raw dissociation data on the Matplotlib axes
            ax.plot(dissoc_zone['time'] - t0, dissoc_zone['response'], color=colour, lw=1.5)

        # Checks if the fitted association model is successfully derived
        if f['assoc_fit'] is not None:

            # Extracts the association fit parameters
            Req, kobs, R0, RI_a = f['assoc_fit']

            # Extracts the time values for the fitted association curve
            t_fit = assoc_zone['time'].values

            # Calculates the predicted response values using the association model
            y_fit = assoc_model(t_fit, Req, kobs, R0, RI_a)

            # Adds the fit line to the Plotly figure, hiding the duplicate legend entry
            fig_plotly.add_trace(go.Scatter(x=t_fit - t0, y=y_fit, mode='lines', line=dict(color='black', dash='dash', width=1.5), showlegend=False, hoverinfo='skip', legendgroup=group_id))

            # Plots the fitted association line on the Matplotlib axes
            ax.plot(t_fit - t0, y_fit, '--', color='black', lw=1.2)

        # Checks if the fitted dissociation model is successfully derived
        if f['dissoc_fit'] is not None:

            # Extracts the dissociation fit parameters
            R0d, kd, Rinf, RI_d = f['dissoc_fit']

            # Extracts the time values for the fitted dissociation curve
            t_fit_d = dissoc_zone['time'].values

            # Calculates the predicted response values using the dissociation model
            y_fit_d = dissoc_model(t_fit_d, R0d, kd, Rinf, RI_d)

            # Adds the fit line to the Plotly figure, hiding the duplicate legend entry
            fig_plotly.add_trace(go.Scatter(x=t_fit_d - t0, y=y_fit_d, mode='lines', line=dict(color='black', dash='dash', width=1.5), showlegend=False, hoverinfo='skip', legendgroup=group_id))

            # Plots the fitted dissociation line on the Matplotlib axes
            ax.plot(t_fit_d - t0, y_fit_d, '--', color='black', lw=1.2)

        # Adds an empty plot element to generate the legend entry for the Matplotlib figure
        ax.plot([], [], color=colour, label=f['meas_id'])

    # Updates the layout configuration for the Plotly figure
    fig_plotly.update_layout(title=title, xaxis_title='Time from injection start (s)', yaxis_title='Response (RU, ref-subtracted)', template='plotly_white', legend_title_text='Measurement')
    
    # Renders the interactive Plotly figure
    fig_plotly.show()

    # Sets the x-axis label for the Matplotlib graph
    ax.set_xlabel('Time from injection start (s)')

    # Sets the y-axis label for the Matplotlib graph
    ax.set_ylabel('Response (RU, ref-subtracted)')

    # Sets the title for the Matplotlib graph
    ax.set_title(title)
    
    # Configures and adds the legend to the Matplotlib axes
    ax.legend(title='Measurement', fontsize=8, loc='upper right')

    # Adjusts the layout to prevent clipping of plot elements
    plt.tight_layout()

    # Renders the static Matplotlib graph
    plt.show()

def plot_rates_vs_conc(data, title_prefix):
    '''Plots k_obs, k_a, k_d, and K_D against concentration as sequential plots.
    
    Includes all data points, even physically impossible negative values, for diagnostic purposes.

    Args:
        data (dict): Kinetic results containing fits and global rates.
        title_prefix (str): Prefix for the plot titles.
    '''

    # Extracts the concentration values for all successful association fits
    concs_kobs = [f['conc'] for f in data['fits'] if f['assoc_fit'] is not None]

    # Extracts the observed association rates for all successful association fits
    kobs_vals = [f['assoc_fit'][1] for f in data['fits'] if f['assoc_fit'] is not None]

    # Evaluates whether any valid concentration values exist
    if concs_kobs:

        # Initialises the figure and axes for the k_obs scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plots the k_obs scatter points using the extracted values
        ax.scatter(concs_kobs, kobs_vals, color='tab:blue')

        # Checks whether sufficient data exists to plot a linear fit line
        if len(concs_kobs) >= 2 and 'ka' in data:
            
            # Generates evenly spaced concentration values for the fit line
            x_fit = np.linspace(0, max(concs_kobs), 50)

            # Calculates the corresponding y-values using the global ka and kd rates
            y_fit = data['ka'] * x_fit + data['kd']

            # Plots the linear fit line and adds its equation to the legend
            ax.plot(x_fit, y_fit, '--', color='black', label=f"ka={data['ka']:.2e}, kd={data['kd']:.2e}")

            # Renders the legend on the axes
            ax.legend()

        # Sets the x-axis label for the k_obs plot
        ax.set_xlabel('Concentration (ug/ml)')

        # Sets the y-axis label for the k_obs plot
        ax.set_ylabel('k_obs (s$^{-1}$)')

        # Sets the title for the k_obs plot
        ax.set_title(f'{title_prefix} - k_obs')

        # Adds a dashed background grid to the k_obs plot
        ax.grid(True, linestyle='--', alpha=0.5)

        # Renders the k_obs scatter plot
        plt.show()

    # Initialises an empty list to store concentration values for individual parameters
    concs_params = []

    # Initialises an empty list to store calculated individual association rates
    ka_vals = []

    # Initialises an empty list to store extracted individual dissociation rates
    kd_vals = []

    # Initialises an empty list to store calculated equilibrium constants
    KD_vals = []

    # Loops through each generated curve fit to extract and calculate individual rates
    for f in data['fits']:

        # Checks whether both association and dissociation fits are successful and concentration is valid
        if f['assoc_fit'] is not None and f['dissoc_fit'] is not None and f['conc'] > 0:

            # Extracts the observed association rate from the fit parameters
            kobs = f['assoc_fit'][1]

            # Extracts the dissociation rate from the fit parameters
            kd = f['dissoc_fit'][1]

            # Calculates the individual association rate for the current concentration
            ka_indiv = (kobs - kd) / f['conc']
            
            # Calculates the equilibrium constant, avoiding division by zero errors
            KD_indiv = kd / ka_indiv if ka_indiv != 0 else np.nan
            
            # Appends the concentration value to the tracking list
            concs_params.append(f['conc'])

            # Appends the calculated individual association rate to the tracking list
            ka_vals.append(ka_indiv)

            # Appends the extracted dissociation rate to the tracking list
            kd_vals.append(kd)

            # Appends the calculated equilibrium constant to the tracking list
            KD_vals.append(KD_indiv)

    # Evaluates whether the collected parameter tracking lists are empty
    if not concs_params:

        # Prints a warning message indicating a lack of valid data points
        print('No valid fits available for additional parameter plots.')

        return

    # Initialises the figure and axes for the individual association rate scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plots the individual association rate scatter points
    ax.scatter(concs_params, ka_vals, color='tab:green')
    
    # Sets the x-axis label for the association rate plot
    ax.set_xlabel('Concentration (ug/ml)')

    # Sets the y-axis label for the association rate plot
    ax.set_ylabel('k_a (ug/ml*s)$^{-1}$')

    # Sets the title for the association rate plot
    ax.set_title(f'{title_prefix} - Association Rate (k_a)')

    # Adds a horizontal reference line at zero
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    # Adds a dashed background grid to the association rate plot
    ax.grid(True, linestyle='--', alpha=0.5)

    # Renders the individual association rate scatter plot
    plt.show()

    # Initialises the figure and axes for the individual dissociation rate scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plots the individual dissociation rate scatter points
    ax.scatter(concs_params, kd_vals, color='tab:red')
    
    # Sets the x-axis label for the dissociation rate plot
    ax.set_xlabel('Concentration (ug/ml)')

    # Sets the y-axis label for the dissociation rate plot
    ax.set_ylabel('k_d (s$^{-1}$)')

    # Sets the title for the dissociation rate plot
    ax.set_title(f'{title_prefix} - Dissociation Rate (k_d)')

    # Adds a dashed background grid to the dissociation rate plot
    ax.grid(True, linestyle='--', alpha=0.5)

    # Renders the individual dissociation rate scatter plot
    plt.show()

    # Initialises the figure and axes for the equilibrium constant scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plots the equilibrium constant scatter points
    ax.scatter(concs_params, KD_vals, color='tab:purple')
    
    # Sets the x-axis label for the equilibrium constant plot
    ax.set_xlabel('Concentration (ug/ml)')

    # Sets the y-axis label for the equilibrium constant plot
    ax.set_ylabel('K_D (ug/ml)')

    # Sets the title for the equilibrium constant plot
    ax.set_title(f'{title_prefix} - Equilibrium Constant (K_D)')

    # Configures the y-axis to use a symmetrical logarithmic scale
    ax.set_yscale('symlog')

    # Adds a horizontal reference line at zero
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    # Adds a dashed background grid to the equilibrium constant plot
    ax.grid(True, linestyle='--', alpha=0.5)

    # Renders the equilibrium constant scatter plot
    plt.show()

def plot_binding_curves(data, title, pre_baseline_seconds=10, tail_avg_seconds=5):
    '''Plots baseline-aligned binding curves using configurable sampling windows.

    Args:
        data (dict): Sensorgram measurements to align and plot.
        title (str): Plot title.
        pre_baseline_seconds (float, optional): Duration of the pre-event baseline window. Defaults to 10.
        tail_avg_seconds (float, optional): Duration of the tail-averaging window. Defaults to 5.
    '''

    # Extracts the raw sensorgram data from the input dictionary
    sensor = data['sensor']
    
    # Initialises the Matplotlib figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Initialises the Plotly figure for interactive viewing
    fig_plotly = go.Figure()
    
    # Defines the colour map for the plotted curves
    cmap = plt.cm.viridis
    
    # Determines the total number of fits to calibrate the colour scaling
    n = len(data['fits'])

    # Initialises a tracking variable for the current curve identifier
    curve_id = 0

    # Loops through each index and fit dictionary in the data
    for i, f in enumerate(data['fits']):

        # Extracts the association zone from the current fit
        assoc_zone = f['assoc_zone']

        # Skips the current iteration if the association zone contains no data
        if assoc_zone.empty:
            continue

        # Assigns a dynamically scaled colour based on the total number of fits
        colour = cmap(i / max(n - 1, 1)) 

        # Converts the mapped colour to a hexadecimal string for Plotly rendering
        hex_colour = mcolors.to_hex(colour)
        
        # Identifies the start time of the association zone
        t0 = assoc_zone['time'].iloc[0]

        # Slices out a specific event window encompassing the baseline and association phases
        window = sensor[(sensor['time'] >= t0 - pre_baseline_seconds) & (sensor['time'] <= assoc_zone['time'].iloc[-1])]

        # Slices out the specific pre-event zone to calculate the standard baseline
        pre_zone = window[window['time'] < t0]

        # Calculates the baseline mean, falling back to the first window value if empty
        baseline = pre_zone['response'].mean() if not pre_zone.empty else window['response'].iloc[0]

        # Normalises the time array relative to the event start
        t_plot = window['time'] - t0

        # Aligns the response array relative to the calculated baseline
        y_plot = window['response'] - baseline

        # Extracts the measurement identifier to serve as the plot label
        label = f['meas_id']
        
        # Adds the aligned trace as a scatter plot to the Plotly figure
        fig_plotly.add_trace(go.Scatter(x=t_plot, y=y_plot, mode='markers', marker=dict(color=hex_colour, size=4), name=label))

        # Plots the aligned trace as a scatter plot on the Matplotlib axes
        ax.scatter(t_plot, y_plot, s=18, label=label, color=colour)

        # Slices the tail zone based on the configured tail-averaging duration
        tail_zone = y_plot[t_plot >= t_plot.max() - tail_avg_seconds]

        # Calculates the average plateau value from the tail zone
        plateau_val = tail_zone.mean()

        # Extracts the maximum time value within the plotted data
        t_max = t_plot.max()

        # Calculates the total time range within the plotted data
        t_range = t_max - t_plot.min()

        # Determines the starting x-coordinate for the plateau annotation line
        x0_pl = t_max - (0.15 * t_range) 
        
        # Adds a dashed plateau indicator line to the Plotly figure
        fig_plotly.add_trace(go.Scatter(x=[x0_pl, t_max], y=[plateau_val, plateau_val], mode='lines', line=dict(color=hex_colour, dash='dot', width=2), showlegend=False, hoverinfo='skip'))

        # Adds the calculated plateau value as a text annotation to the Plotly figure
        fig_plotly.add_annotation(x=t_max, y=plateau_val, text=f'{plateau_val:.3f}', showarrow=False, xanchor='left', xshift=5, font=dict(size=11, color=hex_colour))

        # Adds a dashed plateau indicator line to the Matplotlib axes
        ax.axhline(y=plateau_val, xmin=0.85, xmax=1.0, linestyle=':', color=colour, linewidth=1.2)

        # Adds the calculated plateau value as a text annotation to the Matplotlib axes
        ax.annotate(f'{plateau_val:.3f}', xy=(1.0, plateau_val), xycoords=('axes fraction', 'data'), xytext=(5, 0), textcoords='offset points', va='center', ha='left', fontsize=9)

        # Increments the curve identifier for the next iteration
        curve_id += 1

    # Updates the layout configuration for the Plotly figure
    fig_plotly.update_layout(title=title, xaxis_title='Time (sec)', yaxis_title='Signal (RU)', template='plotly_white', legend_title_text='Measurement')
    
    # Renders the interactive Plotly figure
    fig_plotly.show()

    # Sets the x-axis label for the Matplotlib graph
    ax.set_xlabel('Time(sec)')

    # Sets the y-axis label for the Matplotlib graph
    ax.set_ylabel('Signal (RU)')

    # Sets the title for the Matplotlib graph
    ax.set_title(title)

    # Adds a background grid to the Matplotlib graph
    ax.grid(True)

    # Configures and adds the legend to the Matplotlib axes
    ax.legend(loc='upper left', fontsize=8)

    # Adjusts the layout to prevent clipping of plot elements
    plt.tight_layout()

    # Renders the static Matplotlib graph
    plt.show()

def plot_overall_rates_vs_conc(sc_data, title_prefix):
    '''Plots boxplots and stripplots for individual kinetic rates (k_obs, k_a, k_d, K_D) grouped by concentration across all assay measurements.

    Args:
        sc_data (list): A list of data dictionaries containing 'fits'.
        title_prefix (str): Prefix for the plot titles.
    '''
    
    # Initialises an empty list to aggregate individual per-concentration rates
    records = []
    
    # Loops through each chip's data dictionary in the global dataset
    for data in sc_data:

        # Skips the current data dictionary if no fits are present
        if 'fits' not in data:
            continue
            
        # Loops through each generated curve fit for the current chip
        for f in data['fits']:

            # Checks whether both association and dissociation fits are successful and concentration is valid
            if f['assoc_fit'] is not None and f['dissoc_fit'] is not None and f['conc'] > 0:

                # Extracts the observed association rate from the fit parameters
                kobs = f['assoc_fit'][1]

                # Extracts the dissociation rate from the fit parameters
                kd = f['dissoc_fit'][1]

                # Calculates the individual association rate for the current concentration
                ka_indiv = (kobs - kd) / f['conc']

                # Calculates the equilibrium constant, avoiding division by zero errors
                KD_indiv = kd / ka_indiv if ka_indiv != 0 else np.nan
                
                # Appends the extracted and calculated metrics to the tracking list
                records.append({
                    'Concentration': f['conc'],
                    'k_obs': kobs,
                    'k_a': ka_indiv,
                    'k_d': kd,
                    'K_D': KD_indiv
                })
    
    # Evaluates whether the aggregation process successfully extracted any records
    if not records:

        # Prints a warning message indicating a lack of valid data points
        print('No individual per-concentration rates found across the dataset.')

        return

    # Converts the aggregated list of records into a pandas dataframe
    df = pd.DataFrame(records)

    # Sorts the dataframe by concentration to ensure ordered x-axis categories
    df = df.sort_values('Concentration')

    # Defines the metrics, axis labels, and formatting rules for the sequential plots
    metrics = [
        ('k_obs', 'k_obs (s$^{-1}$)', 'tab:blue', 'lightblue', False),
        ('k_a', 'k_a (ug/ml*s)$^{-1}$', 'tab:green', 'lightgreen', False),
        ('k_d', 'k_d (s$^{-1}$)', 'tab:red', 'lightcoral', False),
        ('K_D', 'K_D (ug/ml)', 'tab:purple', '#d8b4e2', True)
    ]

    # Loops through each defined metric to generate its grouped plot
    for col, ylabel, color, boxcolor, is_symlog in metrics:
        
        # Drops any missing values for the specific metric being plotted
        df_plot = df.dropna(subset=[col])

        # Skips the current metric if the filtered dataframe contains no data
        if df_plot.empty:
            continue

        # Initialises the figure and axes for the current metric's plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plots the boxplot, automatically grouping the discrete concentration values
        sns.boxplot(data=df_plot, x='Concentration', y=col, color=boxcolor, width=0.4, showfliers=False, ax=ax)

        # Overlays a stripplot to display individual data points within the groups
        sns.stripplot(data=df_plot, x='Concentration', y=col, color=color, size=6, jitter=True, alpha=0.8, ax=ax)
        
        # Checks if the current metric requires a symmetrical logarithmic scale
        if is_symlog:

            # Configures the y-axis to use a symmetrical logarithmic scale
            ax.set_yscale('symlog')

            # Adds a horizontal reference line at zero
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

        # Evaluates if the current metric is the association rate
        elif col == 'k_a':

            # Adds a horizontal reference line at zero for the association rate plot
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
            
        # Sets the y-axis label based on the current metric
        ax.set_ylabel(ylabel)

        # Sets the x-axis label for concentration
        ax.set_xlabel('Concentration (ug/ml)')

        # Sets the title for the grouped metric plot
        ax.set_title(f'{title_prefix} - {col} grouped by Concentration')

        # Adds a dashed background grid to the plot
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Adjusts the layout to prevent clipping of plot elements
        plt.tight_layout()

        # Renders the current metric's grouped plot
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

    # Initialises an empty dictionary to store the computed kinetics results
    kinetics_results = {}
    
    # Cleans the input baseline target to guarantee matching regardless of spacing
    target_baseline = str(baseline).replace(' ', '')

    # Loops through each folder within the provided files dictionary
    for folder in files.keys():

        # Skips the current iteration if the folder contains solely baseline data
        if 'baseline' in folder:
            continue

        # Initialises an empty dictionary to store kinetics results for the current folder
        kinetics_results[folder] = {}

        # Initialises a variable to track the previous file name for flag matching
        previous_file = ''

        # Loops through each file within the current folder
        for file in list(files[folder].keys()):

            # Evaluates whether the current file is a sensorgram data file
            if 'sensorgram' in file:
       
                # Sorts the sensorgram data chronologically and resets its index
                sensor = files[folder][file].sort_values('time').reset_index(drop=True)

                # Sorts the matched flag data chronologically and resets its index
                flags = files[folder][previous_file].sort_values('time').reset_index(drop=True)

                # Maps the primary response channel for analysis
                sensor['response'] = sensor['channel1']

                # Constructs the expected filename for the corresponding baseline flags
                file_name_flags = file.split('_')[1] + '_baseline_flags'

                # Retrieves the associated baseline flags from the folder dictionary
                baseline_flags = files[folder].get(file_name_flags)

                # Initialises an empty list to store parsed concentration events
                parsed = []

                # Loops through each row in the standard flag data
                for _, row in flags.iterrows():

                    # Parses the concentration string to extract the numeric value
                    conc_val = parse_conc_flag(row['conc'])

                    # Checks whether a valid numeric concentration was successfully extracted
                    if conc_val is not None:

                        # Appends the parsed event data to the tracking list
                        parsed.append({'time': row['time'], 'conc': conc_val, 'label': str(row['conc'])})

                # Initialises an empty list to store defined kinetic analysis segments
                segments = []

                # Evaluates whether baseline flags were successfully retrieved
                if baseline_flags is not None:

                    # Strips spaces from the baseline information column to match the cleaned target
                    info_str = baseline_flags['information'].astype(str).str.replace(' ', '')

                    # Filters the baseline flags to isolate the specific target events
                    target_flags = baseline_flags[info_str.str.contains(target_baseline, na=False)]

                    # Loops through each parsed concentration event to define its boundaries
                    for i, event in enumerate(parsed):

                        # Assigns the event start time
                        t_start = event['time']

                        # Determines the next event start time or defaults to the maximum sensor time
                        t_next = parsed[i+1]['time'] if i + 1 < len(parsed) else sensor['time'].max()

                        # Matches the event window to the filtered target baseline timings
                        valid_flags = target_flags[(target_flags['absolute_peak_time'] >= t_start) & (target_flags['absolute_peak_time'] <= t_next)]
                        
                        # Skips the current event if no valid baseline flags are found within its window
                        if valid_flags.empty:
                            continue
                            
                        # Extracts the first valid flag row corresponding to the event
                        flag_row = valid_flags.iloc[0]
                        
                        # Assigns the association phase start time
                        t_assoc_start = t_start

                        # Assigns the association phase end time based on the absolute peak
                        t_assoc_end = flag_row['absolute_peak_time']
                        
                        # Assigns the dissociation phase start time based on the pre-drop peak
                        t_dissoc_start = flag_row['peak_time']

                        # Assigns the dissociation phase end time based on the target baseline window
                        t_dissoc_end = flag_row['Window_Start']

                        # Skips the event if any critical timing parameters are missing
                        if pd.isna(t_assoc_end) or pd.isna(t_dissoc_start) or pd.isna(t_dissoc_end):
                            continue

                        # Appends the defined segment boundaries to the tracking list
                        segments.append({
                            'conc': event['conc'], 
                            'meas_id': event['label'], 
                            't_assoc_start': t_assoc_start,
                            't_assoc_end': t_assoc_end,
                            't_dissoc_start': t_dissoc_start,
                            't_dissoc_end': t_dissoc_end
                        })

                # Initialises an empty list to store the generated curve fits
                fits = []

                # Loops through each defined analysis segment
                for seg in segments:

                    # Slices the sensorgram data to isolate the association zone
                    assoc_zone = sensor[(sensor['time'] >= seg['t_assoc_start']) & (sensor['time'] <= seg['t_assoc_end'])]

                    # Slices the sensorgram data to isolate the dissociation zone
                    dissoc_zone = sensor[(sensor['time'] >= seg['t_dissoc_start']) & (sensor['time'] <= seg['t_dissoc_end'])]

                    # Executes the association curve fit if sufficient data points exist
                    assoc_fit = fit_association(assoc_zone['time'].values, assoc_zone['response'].values) if len(assoc_zone) > 5 else None

                    # Executes the dissociation curve fit if sufficient data points exist
                    dissoc_fit = fit_dissociation(dissoc_zone['time'].values, dissoc_zone['response'].values) if len(dissoc_zone) > 5 else None

                    # Appends the generated fits and their corresponding data zones to the tracking list
                    fits.append({
                        'conc': seg['conc'], 'meas_id': seg['meas_id'],
                        'assoc_fit': assoc_fit, 'dissoc_fit': dissoc_fit,
                        'assoc_zone': assoc_zone, 'dissoc_zone': dissoc_zone
                    })

                # Compiles the raw data, segments, and fits into a consolidated file dictionary
                file_data = {'sensor': sensor, 'segments': segments, 'fits': fits}

                # Extracts the concentration values for all successful association fits
                concs = [f['conc'] for f in fits if f['assoc_fit'] is not None]

                # Extracts the observed association rates for all successful association fits
                kobs_vals = [f['assoc_fit'][1] for f in fits if f['assoc_fit'] is not None]

                # Evaluates whether sufficient data exists to calculate global kinetics
                if len(concs) >= 2:

                    # Calculates the linear regression between concentrations and observed association rates
                    slope, intercept, r_value, _, _ = linregress(concs, kobs_vals)

                    # Assigns the slope as the global association rate
                    ka = slope

                    # Assigns the intercept as the global dissociation rate
                    kd_from_intercept = intercept

                    # Calculates the global equilibrium constant, avoiding division by zero errors
                    KD = kd_from_intercept / ka if ka != 0 else np.nan

                    # Appends the global association rate to the file dictionary
                    file_data['ka'] = ka

                    # Appends the global dissociation rate to the file dictionary
                    file_data['kd'] = kd_from_intercept

                    # Appends the global equilibrium constant to the file dictionary
                    file_data['KD'] = KD

                    # Appends the R-squared value of the linear fit to the file dictionary
                    file_data['kobs_r2'] = r_value ** 2

                    # Extracts the individual dissociation rates for all successful dissociation fits
                    dissoc_kds = [f['dissoc_fit'][1] for f in fits if f['dissoc_fit'] is not None]

                    # Checks whether any valid individual dissociation rates exist
                    if dissoc_kds:

                        # Calculates and appends the mean of the individual dissociation rates to the file dictionary
                        file_data['kd_dissoc_mean'] = np.mean(dissoc_kds)

                # Stores the compiled file dictionary within the overarching kinetics results dictionary
                kinetics_results[folder][file] = file_data

            # Updates the previous file tracker for the next iteration
            previous_file = file

    # Initialises an empty list to aggregate global kinetics data across all folders
    all_global_data = []

    # Loops through each folder and its associated files in the generated results
    for folder, folder_files in kinetics_results.items():

        # Loops through each file and its data within the current folder
        for file, data in folder_files.items():

            # Checks whether generated fits are present in the current file data
            if 'fits' in data:

                # Appends the file data to the global tracking list
                all_global_data.append(data)
                
    # Evaluates whether any global kinetics data was successfully aggregated
    if all_global_data:
        
        # Prints the header for overall kinetics distributions
        print(f"\n{'='*40}\nOverall Kinetics Distributions\n{'='*40}")
            
        # Displays the overall grouped distributions within a collapsible output section
        with collapsible_output('Overall Distributions by Concentration'):

            # Generates the grouped concentration plots using the aggregated data
            plot_overall_rates_vs_conc(all_global_data, 'Overall')
            
    # Loops through each folder and its associated files to generate specific summaries
    for folder, folder_files in kinetics_results.items():

        # Prints the analysis header for the current folder
        print(f"\n{'='*40}\nKinetics Analysis: {folder}\n{'='*40}")

        # Initialises an empty list to store summary rows for the current folder
        summary_rows = []

        # Loops through each file and its data within the current folder
        for file, data in folder_files.items():

            # Checks whether global association rates were successfully calculated
            if 'ka' in data:

                # Appends the formatted metrics to the summary tracking list
                summary_rows.append({
                    'File': file, 'k_a (ug/ml*s)^-1': data['ka'], 'k_d_intercept (s^-1)': data['kd'],
                    'k_d_dissoc_mean (s^-1)': data.get('kd_dissoc_mean', np.nan),
                    'K_D (ug/ml)': data['KD'], 'k_obs_fit_R2': data['kobs_r2']
                })

        # Evaluates whether any summary rows were collected for the current folder
        if summary_rows:

            # Displays the global kinetics summary table within a collapsible output section
            with collapsible_output(f'Global Kinetics Summary: {folder}'):

                # Converts the summary rows into a pandas dataframe and renders it
                display(pd.DataFrame(summary_rows))

        # Loops through each file and its data again to generate detailed visualisations
        for file, data in folder_files.items():

            # Skips the current file if no fits were generated
            if 'fits' not in data: 
                continue

            # Cleans the file name by stripping unneeded prefixes
            clean_filename = file.replace('Copy of ', '').strip()

            # Splits the cleaned file name to extract identifiers
            parts = clean_filename.split('_')

            # Extracts the chip identifier or assigns a default placeholder
            chip_id = parts[0] if len(parts) > 0 else 'Unknown'

            # Extracts the measurement identifier or assigns a default placeholder
            measurement_id = parts[1] if len(parts) > 1 else 'Unknown'
            
            # Prints a sub-header indicating the current chip and measurement
            print(f'\n--- Chip ID: {chip_id} | Measurement ID: {measurement_id} ---')

            # Initialises an empty list to store per-concentration detail rows
            per_conc_rows = []

            # Loops through each generated curve fit for the current file
            for f in data['fits']:

                # Initialises an empty string to track specific fit warnings
                warning = ''
                                
                # Extracts association parameters if the fit is valid, otherwise assigns NaNs
                Req, kobs, R0 = (f['assoc_fit'][0], f['assoc_fit'][1], f['assoc_fit'][2]) if f['assoc_fit'] is not None else (np.nan, np.nan, np.nan)

                # Checks whether the association fit failed
                if f['assoc_fit'] is None: 

                    # Appends a failure warning to the tracking string
                    warning += 'Assoc fit failed; '
                    
                # Extracts dissociation parameters if the fit is valid, otherwise assigns NaNs
                R0d, kd, Rinf = (f['dissoc_fit'][0], f['dissoc_fit'][1], f['dissoc_fit'][2]) if f['dissoc_fit'] is not None else (np.nan, np.nan, np.nan)

                # Checks whether the dissociation fit failed
                if f['dissoc_fit'] is None: 

                    # Appends a failure warning to the tracking string
                    warning += 'Dissoc fit failed; '
                    
                # Evaluates whether both observed rates exist and concentration is valid
                if not np.isnan(kobs) and not np.isnan(kd) and f['conc'] > 0:

                    # Calculates the individual association rate
                    ka_indiv = (kobs - kd) / f['conc']

                    # Calculates the individual equilibrium constant
                    KD_indiv = kd / ka_indiv if ka_indiv != 0 else np.nan

                    # Evaluates whether the dissociation rate is physically impossible compared to kobs
                    if kd >= kobs: 

                        # Appends a negative association rate warning to the tracking string
                        warning += 'kd >= kobs (Negative k_a); '

                else:

                    # Assigns NaNs to the calculated individual rates due to missing prerequisites
                    ka_indiv, KD_indiv = np.nan, np.nan
                    
                # Appends the detailed metrics and warnings to the per-concentration tracking list
                per_conc_rows.append({
                    'Measurement': f['meas_id'], 'Concentration (ug/ml)': f['conc'], 'k_obs (s^-1)': kobs,
                    'k_d (s^-1)': kd, 'k_a (calc)': ka_indiv, 'K_D': KD_indiv, 'R_eq': Req, 'Warning': warning.strip('; ')
                })

            # Evaluates whether any per-concentration details were collected
            if per_conc_rows:

                # Converts the detail rows into a sorted pandas dataframe
                df_per_conc = pd.DataFrame(per_conc_rows).sort_values(by=['Concentration (ug/ml)', 'Measurement']).reset_index(drop=True)

                # Displays the detailed metrics table within a collapsible output section
                with collapsible_output(f'Per-Concentration Details: {file}'):

                    # Renders the formatted per-concentration dataframe
                    display(df_per_conc)

            # Displays the kinetic fit curves within a collapsible output section
            with collapsible_output(f'Kinetic Fit Curves: {file}'):

                # Generates and renders the kinetic fit curves for the current chip
                plot_kinetic_curves(data, f'Kinetic Curves - {chip_id}')

            # Checks whether global association rates exist to generate additional scatter plots
            if 'ka' in data:

                # Displays the rates versus concentration plots within a collapsible output section
                with collapsible_output(f'Rates vs Concentration: {file}'):

                    # Generates and renders the scatter plots for the current chip
                    plot_rates_vs_conc(data, f'Rates vs Concentration - {chip_id}')

            # Displays the baseline-aligned binding curves within a collapsible output section
            with collapsible_output(f'Binding Curves: {file}'):

                # Generates and renders the overlaid binding curves for the current chip
                plot_binding_curves(data, f'Binding Curves Overlay - {chip_id}')
     
    # Returns the completed kinetics results dictionary
    return kinetics_results
