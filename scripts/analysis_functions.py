# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import linregress

from scripts.setup_functions import collapsible_output

def detect_anomalies(df, feature_cols, contamination='auto'):
    '''
    Applies Isolation Forest to detect multivariate outliers.
    Returns the dataframe with 'anomaly' (-1 for outlier, 1 for inlier) 
    and 'anomaly_score' columns.
    '''

    df_clean = df.dropna(subset=feature_cols).copy()
    
    if df_clean.empty:
        return df_clean
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[feature_cols])
    
    iso = IsolationForest(contamination=contamination, random_state=8030)

    df_clean['anomaly'] = iso.fit_predict(X_scaled)
    df_clean['anomaly_score'] = iso.decision_function(X_scaled)
    
    return df_clean

def pivot_chip_data(df, stage_col, val_col):
    '''
    Pivots data for anomaly detection. 
    If a 'Channel' column exists, it uses BOTH chip_id and Channel as the row index, 
    effectively treating each channel as an independent observation (doubling the dataset size).
    '''

    if 'Channel' in df.columns:
        return df.pivot_table(index=['chip_id', 'Channel'], columns=stage_col, values=val_col, aggfunc='mean', observed=False)
    else:
        return df.pivot_table(index='chip_id', columns=stage_col, values=val_col, aggfunc='mean', observed=False)
    
def summarise_correlations(corr_matrix, label, num=3):
    '''Summarise strongest positive, strongest negative, and weakest correlations.'''

    print(f'\n {label} Correlation Insights')

    corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).unstack().dropna()

    if corr.empty:
        print('Not enough data to compute correlations.')
        return

    positive = corr[corr > 0].sort_values(ascending=False)
    negative = corr[corr < 0].sort_values()
    weakest = corr.iloc[corr.abs().argsort()]

    print('Top Positive Correlations:')

    if len(positive):

        for (s1, s2), val in positive.head(num).items():
            print(f'  - {s1} & {s2} (r = {val:.3f})')
    else:
        print('  None')

    print('\nTop Negative (Inverse) Correlations:')

    if len(negative):

        for (s1, s2), val in negative.head(num).items():
            print(f'  - {s1} & {s2} (r = {val:.3f})')
    else:
        print('  None')

    print('\nWeakest Relationships (Closest to zero):')

    for (s1, s2), val in weakest.head(num).items():
        print(f'  - {s1} & {s2} (r = {val:.3f})')

def analyse_anomaly_drivers(df, anomalies_df, label, num=3):
    '''Compares anomalous chips to normal chips to find which stages drive the anomaly.'''

    outliers = anomalies_df[anomalies_df['anomaly'] == -1].index
    normals = anomalies_df[anomalies_df['anomaly'] == 1].index

    print(f'\n {label} Anomaly Stage Breakdown')

    if len(outliers) == 0:

        print('No anomalies detected. All chips are behaving within normal bounds.')
        return

    if len(normals) < 2:

        print('Not enough normal chips to form a baseline for comparison.')
        return

    normal_data = df.loc[normals]
    anomaly_data = df.loc[outliers]

    normal_mean = normal_data.mean()
    normal_std = normal_data.std().replace(0, 1e-9) 

    anomaly_mean = anomaly_data.mean()
    z_scores = ((anomaly_mean - normal_mean) / normal_std).abs().sort_values(ascending=False).dropna()

    print(f'Top {num} stages driving these anomalies (Largest deviation from normal):')

    for stage, z in z_scores.head(num).items():

        norm_val = normal_mean[stage]
        anom_val = anomaly_mean[stage]
        print(f'  - {stage}: Anomaly Avg = {anom_val:.3f} | Normal Avg = {norm_val:.3f} (Z-Score: {z:.2f})')

def create_wide_intra(intra_df, val_col):
    '''Pivots multiple intra-stage metrics and suffixes columns to prevent overlap.'''

    intra_metrics = [f'{val_col}_std', f'{val_col}_spike_to_noise_ratio', f'{val_col}_max_residual_zscore']

    wide_list = []
    
    for metric in intra_metrics:

        if metric in intra_df.columns:

            wide_metric = pivot_chip_data(intra_df, 'stage', metric)
            metric_suffix = metric.replace(f'{val_col}_', '')
            wide_metric.columns = [f'{col}_{metric_suffix}' for col in wide_metric.columns]
            wide_list.append(wide_metric)
            
    return pd.concat(wide_list, axis=1) if wide_list else pd.DataFrame()

def get_outlier_chips(anomalies_df):
    '''Extracts unique chip IDs from the anomaly dataframe, handling both standard and MultiIndex.'''

    if anomalies_df.empty:
        return []
    
    outliers = anomalies_df[anomalies_df['anomaly'] == -1]
    
    if isinstance(outliers.index, pd.MultiIndex):
        return outliers.index.get_level_values('chip_id').unique().tolist()
    
    return outliers.index.tolist()

def analyse_pel(events_df, changes_df, intra_df, val_col='quad_ch1', change_col='quad_ch1_change', intra_val_col=None, title='PEL Analysis'):

    intra_target = intra_val_col if intra_val_col else val_col

    events_df = events_df[~events_df['stage'].isin(['Start', 'Initial'])]
    
    wide_events = pivot_chip_data(events_df, 'stage', val_col)
    wide_changes = pivot_chip_data(changes_df, 'stage', change_col)
    wide_intra = create_wide_intra(intra_df, intra_target)

    anomalies_abs = detect_anomalies(wide_events, wide_events.columns.tolist())
    outliers_abs = get_outlier_chips(anomalies_abs)

    anomalies_delta = detect_anomalies(wide_changes, wide_changes.columns.tolist())
    outliers_delta = get_outlier_chips(anomalies_delta)

    anomalies_intra = detect_anomalies(wide_intra, wide_intra.columns.tolist())
    outliers_intra = get_outlier_chips(anomalies_intra)

    with collapsible_output(f'{title} - Absolute Signal'):

        plt.figure(figsize=(10, 6))
        sns.heatmap(wide_events.corr(), cmap='plasma', center=0, annot=True)
        plt.title(f'PEL [{val_col}]: Absolute Signal Correlation')
        plt.tight_layout()
        plt.show()

        summarise_correlations(wide_events.corr(), f'PEL {val_col} Absolute Signal')
        print(f'\nPotential Anomalous Chips (Absolute): {len(outliers_abs)}')
        analyse_anomaly_drivers(wide_events, anomalies_abs, f'PEL {val_col} Absolute Signal')

    with collapsible_output(f'{title} - Stage Delta'):

        plt.figure(figsize=(10, 6))
        sns.heatmap(wide_changes.corr(), cmap='plasma', center=0, annot=True)
        plt.title(f'PEL [{change_col}]: Stage Delta Correlation')
        plt.tight_layout()
        plt.show()

        summarise_correlations(wide_changes.corr(), f'PEL {change_col} Stage Delta')
        print(f'\nPotential Anomalous Chips (Delta): {len(outliers_delta)}')
        analyse_anomaly_drivers(wide_changes, anomalies_delta, f'PEL {change_col} Stage Delta')
    
    with collapsible_output(f'{title} - Intra-stage Kinetics'):

        plt.figure(figsize=(10, 6))
        sns.heatmap(wide_intra.corr(), cmap='plasma', center=0, annot=True)
        plt.title(f'PEL [{intra_target}]: Intra-stage Kinetics Correlation')
        plt.tight_layout()
        plt.show()

        summarise_correlations(wide_intra.corr(), f'PEL {intra_target} Intra-stage Kinetics')
        print(f'\nPotential Anomalous Chips (Intra-stage Kinetics): {len(outliers_intra)}')
        analyse_anomaly_drivers(wide_intra, anomalies_intra, f'PEL {intra_target} Intra-stage Kinetics')

    return wide_events, wide_changes, wide_intra, outliers_abs, outliers_delta, outliers_intra

def analyse_immob_split(events_df, changes_df, intra_df, split_name, val_col='channel1', change_col='channel1_change', intra_val_col=None, title='Immobilisation Analysis'):

    intra_target = intra_val_col if intra_val_col else val_col

    events_df = events_df[~events_df['stage'].isin(['Start', 'Initial'])]

    wide_events = pivot_chip_data(events_df, 'stage', val_col)
    wide_changes = pivot_chip_data(changes_df, 'stage', change_col)
    wide_intra = create_wide_intra(intra_df, intra_target)

    anomalies_abs = detect_anomalies(wide_events, wide_events.columns.tolist())
    outliers_abs = get_outlier_chips(anomalies_abs)

    anomalies_delta = detect_anomalies(wide_changes, wide_changes.columns.tolist())
    outliers_delta = get_outlier_chips(anomalies_delta)

    anomalies_intra = detect_anomalies(wide_intra, wide_intra.columns.tolist())
    outliers_intra = get_outlier_chips(anomalies_intra)

    with collapsible_output(f'{title} - Absolute Signal'):

        plt.figure(figsize=(20, 20))
        sns.heatmap(wide_events.corr(), cmap='plasma', annot=True, annot_kws={'size': 12})
        plt.title(f'Immob [{split_name}] - {val_col}: Absolute Signal Correlation')
        plt.tight_layout()
        plt.show()

        summarise_correlations(wide_events.corr(), f'Immob [{split_name}] {val_col} Absolute Signal')
        print(f'\nPotential Anomalous Chips (Absolute): {len(outliers_abs)}')
        analyse_anomaly_drivers(wide_events, anomalies_abs, f'Immob [{split_name}] {val_col} Absolute Signal')

    with collapsible_output(f'{title} - Stage Delta'):

        plt.figure(figsize=(20, 20))
        sns.heatmap(wide_changes.corr(), cmap='plasma', center=0, annot=True)
        plt.title(f'Immob [{split_name}] - {change_col}: Stage Delta Correlation')
        plt.show()

        summarise_correlations(wide_changes.corr(), f'Immob [{split_name}] {change_col} Stage Delta')
        print(f'\nPotential Anomalous Chips (Delta): {len(outliers_delta)}')
        analyse_anomaly_drivers(wide_changes, anomalies_delta, f'Immob [{split_name}] {change_col} Stage Delta')
    
    with collapsible_output(f'{title} - Intra-stage Kinetics'):

        plt.figure(figsize=(20, 20))
        sns.heatmap(wide_intra.corr(), cmap='plasma', center=0, annot=True)
        plt.title(f'Immob [{split_name}] - {intra_target}: Intra-stage Kinetics Correlation')
        plt.show()

        summarise_correlations(wide_intra.corr(), f'Immob [{split_name}] {intra_target} Intra-stage Kinetics')
        print(f'\nPotential Anomalous Chips (Intra-stage Kinetics): {len(outliers_intra)}')
        analyse_anomaly_drivers(wide_intra, anomalies_intra, f'Immob [{split_name}] {intra_target} Intra-stage Kinetics')
    
    return wide_events, wide_changes, wide_intra, outliers_abs, outliers_delta, outliers_intra

def plot_pel_layer_shifts(pel_changes, channels_dict=None):

    if channels_dict is None:
        channels_dict = {
            'Channel 1': 'quad_ch1_change',
            'Channel 2': 'quad_ch2_change'
        }
    
    df_plot = pel_changes[pel_changes['stage'] != 'Total_Shift_Initial_to_Final'].copy()

    for ch_name, ch_col in channels_dict.items():

        with collapsible_output(f'PEL Layer Shifts: {ch_name}'):
            
            fig = go.Figure()
            
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

            plt.figure(figsize=(10, 6))
            
            for chip_id, df_chip in df_plot.groupby('chip_id'):
                plt.plot(df_chip['stage'], df_chip[ch_col], marker='o', alpha=0.5, label=chip_id)

            plt.title(f'PEL Layer Build-up: Shift per Transition ({ch_name})')
            plt.xlabel('Layer Transition')
            plt.ylabel('Signal Shift (Δ)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

            if df_plot['chip_id'].nunique() <= 25:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            plt.show()

def combine_anomalies(*outliers_lists, title='Combined Anomalies Summary'):
    '''Combines an arbitrary number of outlier lists into a single set.'''

    all_outlier_chips = set()

    for outliers in outliers_lists:
        all_outlier_chips.update(outliers)
        
    all_outlier_chips = list(all_outlier_chips)

    with collapsible_output(title):

        print(f'Total Unique Anomalous Chips: {len(all_outlier_chips)}')

        if all_outlier_chips:
            print(f'Chip IDs: {all_outlier_chips}')
        else:
            print('No anomalies detected in this combination.')

    return all_outlier_chips

def analyse_cross_stage_split(pel_df, immob_df, split_name, pel_outliers=None, immob_outliers=None, sc_metrics_df=None, title='Cross-Stage Validation'):
    '''
    Merges PEL and a specific Immobilisation split, runs PCA & Clustering,
    plots the results with anomaly highlights for Channel 1, Channel 2, and Overall.
    Validates against Standard Curves if provided.
    '''
    
    pel_copy = pel_df.copy()
    immob_copy = immob_df.copy()
    
    pel_copy.columns = pel_copy.columns.astype(str)
    immob_copy.columns = immob_copy.columns.astype(str)

    channel_mapping = {
        'quad_ch1': 'Ch1', 'channel1': 'Ch1',
        'quad_ch2': 'Ch2', 'channel2': 'Ch2',
        'quad_ch1_change': 'Ch1', 'channel1_change': 'Ch1',
        'quad_ch2_change': 'Ch2', 'channel2_change': 'Ch2'
    }

    if isinstance(pel_copy.index, pd.MultiIndex) and 'Channel' in pel_copy.index.names:
        pel_copy = pel_copy.rename(index=channel_mapping, level='Channel')
        
    if isinstance(immob_copy.index, pd.MultiIndex) and 'Channel' in immob_copy.index.names:
        immob_copy = immob_copy.rename(index=channel_mapping, level='Channel')

    merged = pd.merge(pel_copy, immob_copy, left_index=True, right_index=True, how='inner', suffixes=('_PEL', '_Immob')).dropna()
    
    if merged.empty:
        print(f'[{split_name}] Not enough overlapping chips between PEL and Immobilisation for cross-analysis.')
        return
        
    both_anomalies = set(pel_outliers if pel_outliers else []).union(set(immob_outliers if immob_outliers else []))

    def analyse_subset(subset_df, label):

        if subset_df.empty:
            return
            
        with collapsible_output(f'{title} - {label}'):

            print(f'[{split_name} - {label}] Overlapping chips analysed: {len(subset_df)}')
            
            features = subset_df.columns.tolist()
            X_subset = StandardScaler().fit_transform(subset_df[features])
            
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X_subset)
            
            df_to_plot = subset_df.copy()
            df_to_plot['PCA1'] = pca_result[:, 0]
            df_to_plot['PCA2'] = pca_result[:, 1]
            
            print(f'[{split_name} - {label}] Explained Variance by first 2 components: {pca.explained_variance_ratio_.sum()*100:.2f}%')
            
            kmeans = KMeans(n_clusters=2, random_state=8030, n_init=10)
            df_to_plot['Cluster'] = kmeans.fit_predict(X_subset)
            
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df_to_plot, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', s=100, alpha=0.7)
            
            if isinstance(df_to_plot.index, pd.MultiIndex):
                subset_chip_ids = df_to_plot.index.get_level_values('chip_id')
            else:
                subset_chip_ids = df_to_plot.index
                
            anomaly_mask = subset_chip_ids.isin(both_anomalies)
            
            if anomaly_mask.any():
                plt.scatter(df_to_plot.loc[anomaly_mask, 'PCA1'], df_to_plot.loc[anomaly_mask, 'PCA2'], edgecolor='black', facecolor='none', s=200, label='Flagged Anomaly', linewidth=2)
                            
            plt.title(f'Cross-Stage PCA: {split_name} ({label})')
            plt.xlabel('Principal Component 1 (General Signal Variance)')
            plt.ylabel('Principal Component 2 (Stage-to-Stage Dynamic Variance)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
            print(f'\n Cluster Profiling [{split_name} - {label}]')
            cluster_means = df_to_plot[features].groupby(df_to_plot['Cluster']).mean()
            overall_mean = df_to_plot[features].mean()
            overall_std = df_to_plot[features].std().replace(0, 1e-9)
            
            cluster_zscores = (cluster_means - overall_mean) / overall_std
            
            for cluster_id in sorted(df_to_plot['Cluster'].unique()):

                print(f'\nCluster {cluster_id} Profile (n={sum(df_to_plot['Cluster'] == cluster_id)} observations):')
                
                top_pos = cluster_zscores.loc[cluster_id].sort_values(ascending=False).head(3)
                top_neg = cluster_zscores.loc[cluster_id].sort_values(ascending=True).head(3)
                
                print('  Defining High Features (Above Average):')
                pos_found = False

                for feat, z in top_pos.items():

                    if z > 0.5:

                        print(f'    - {feat}: +{z:.2f} standard deviations')
                        pos_found = True
                        
                if not pos_found: 
                    print('    - None significantly above average')
                    
                print('  Defining Low Features (Below Average):')
                neg_found = False

                for feat, z in top_neg.items():

                    if z < -0.5:

                        print(f'    - {feat}: {z:.2f} standard deviations')
                        neg_found = True

                if not neg_found: 
                    print('    - None significantly below average')
            
            if sc_metrics_df is not None:

                print(f'\n--- Cluster Functional Yield (R² Comparison) [{split_name} - {label}] ---')
                
                cluster_mapping = df_to_plot[['Cluster']].reset_index()

                if 'index' in cluster_mapping.columns:
                    cluster_mapping = cluster_mapping.rename(columns={'index': 'chip_id'})
                    
                sc_cluster_df = pd.merge(sc_metrics_df.reset_index(), cluster_mapping, on='chip_id', how='inner')
                
                if sc_cluster_df.empty:
                    print('No Standard Curve data available to map to clusters.')
                else:

                    cluster_stats = sc_cluster_df.groupby('Cluster')['r2'].agg(['count', 'mean', 'median', 'std']).fillna(0)
                    print('Standard Curve R² Performance by Cluster:')
                    display(cluster_stats)
                    
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
                    
                subset_unique_chips = set(subset_chip_ids)
                anomaly_ids = list(both_anomalies.intersection(subset_unique_chips))
                
                if not anomaly_ids:
                    print(f'[{split_name} - {label}] No anomalous chips from the cross-stage analysis to validate against SC.')
                else:

                    print(f'\n[{split_name} - {label}] Validating functional performance for anomalous chip(s): {anomaly_ids}\n')
                    
                    anomaly_sc_data = sc_metrics_df[sc_metrics_df.index.isin(anomaly_ids)]
                    
                    if anomaly_sc_data.empty:
                        print('No Standard Curve data found for the flagged anomalous chip(s).')
                    else:

                        print(f'{'Chip ID':<15} | {'Curve UUID':<40} | {'R² Score':<10} | {'Status'}')
                        print('-' * 80)

                        for chip_id, row in anomaly_sc_data.iterrows():

                            r2 = row['r2']
                            curve_id = row['standard_curve_uuid']
                            status = 'PASSED' if r2 >= 0.95 else 'FAILED'
                            print(f'{chip_id:<15} | {curve_id:<40} | {r2:<10.4f} | {status}')
                
                normal_sc_data = sc_metrics_df[~sc_metrics_df.index.isin(anomaly_ids)]

                if not normal_sc_data.empty:
                    print(f'\n>>> Average R² for NORMAL chips: {normal_sc_data['r2'].mean():.4f}')
            else:

                subset_unique_chips = set(subset_chip_ids)
                anomaly_ids = list(both_anomalies.intersection(subset_unique_chips))
                print(f'\n--- Anomaly Identifications [{split_name} - {label}] ---')

                if anomaly_ids:
                    print(f'Flagged Anomalous Chips in cluster: {anomaly_ids}')
                else:
                    print('No anomalous chips detected in this cross-stage split.')

    if isinstance(merged.index, pd.MultiIndex) and 'Ch1' in merged.index.get_level_values('Channel'):
        analyse_subset(merged.xs('Ch1', level='Channel', drop_level=False), 'Channel 1')
        
    if isinstance(merged.index, pd.MultiIndex) and 'Ch2' in merged.index.get_level_values('Channel'):
        analyse_subset(merged.xs('Ch2', level='Channel', drop_level=False), 'Channel 2')
        
    analyse_subset(merged, 'Overall (Pooled)')

def print_correlation_insights(corr_matrix, split_name, channel_name, top_n=3):
    '''Extracts and prints top positive, negative, and weakest correlations.'''

    print(f'{split_name} ({channel_name}) Signal Correlation Insights')
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_pairs = upper_tri.stack().dropna()
    
    if corr_pairs.empty:
        print('Not enough variance to calculate correlation insights.\n')
        return

    print('Top Positive Correlations:')
    top_pos = corr_pairs[corr_pairs > 0].sort_values(ascending=False).head(top_n)

    if top_pos.empty: 
        print('  None')
    else:
        for (feat1, feat2), val in top_pos.items(): 
            print(f'  - {feat1} & {feat2} (r = {val:.3f})')

    print('\nTop Negative (Inverse) Correlations:')
    top_neg = corr_pairs[corr_pairs < 0].sort_values(ascending=True).head(top_n)

    if top_neg.empty: 
        print('  None')
    else:
        for (feat1, feat2), val in top_neg.items(): 
            print(f'  - {feat1} & {feat2} (r = {val:.3f})')

    print('\nWeakest Relationships (Closest to zero):')
    weakest = corr_pairs.reindex(corr_pairs.abs().sort_values().index).head(top_n)

    for (feat1, feat2), val in weakest.items():
        print(f'  - {feat1} & {feat2} (r = {val:.3f})')
    print('\n')

def cross_stage_correlations(pel_df, immob_df, split_name, sc_metrics_df=None, title='Cross-Stage Correlations'):

    pel_copy = pel_df.copy()
    immob_copy = immob_df.copy()

    pel_copy.columns = pel_copy.columns.astype(str)
    immob_copy.columns = immob_copy.columns.astype(str)

    channel_mapping = {
        'quad_ch1': 'Ch1', 'channel1': 'Ch1',
        'quad_ch2': 'Ch2', 'channel2': 'Ch2',
        'quad_ch1_change': 'Ch1', 'channel1_change': 'Ch1',
        'quad_ch2_change': 'Ch2', 'channel2_change': 'Ch2'
    }
    
    if isinstance(pel_copy.index, pd.MultiIndex) and 'Channel' in pel_copy.index.names:
        pel_copy = pel_copy.rename(index=channel_mapping, level='Channel')
        
    if isinstance(immob_copy.index, pd.MultiIndex) and 'Channel' in immob_copy.index.names:
        immob_copy = immob_copy.rename(index=channel_mapping, level='Channel')

    merged = pel_copy.merge(immob_copy, left_index=True, right_index=True, how='inner', suffixes=('_PEL', '_Immob'))

    if sc_metrics_df is not None:

        sc_copy = sc_metrics_df.copy()
        sc_copy.columns = sc_copy.columns.astype(str)
        numeric_cols = sc_copy.select_dtypes(include='number').columns.tolist()
        
        if 'datetime' in sc_copy.columns: 
            sc_copy = sc_copy[['datetime'] + numeric_cols]
        else: 
            sc_copy = sc_copy[numeric_cols]
            
        sc_first = sc_copy.sort_values('datetime').groupby(sc_copy.index, sort=False).first().drop(columns='datetime')
        if isinstance(merged.index, pd.MultiIndex):
            merged = merged.merge(sc_first, left_on='chip_id', right_index=True, how='left')
        else:
            merged = merged.merge(sc_first, left_index=True, right_index=True, how='left')

    if merged.empty:

        print(f'[{split_name}] Not enough overlapping data points for cross-analysis.')

        return
    
    def analyse_subset(subset_df, label):

        if subset_df.empty:
            return
            
        with collapsible_output(f'{title} - {label}'):

            print(f'[{split_name} - {label}] Overlapping data points analysed: {len(subset_df)}\n')

            corr_matrix = subset_df.corr()
            print_correlation_insights(corr_matrix, split_name, label, top_n=3)

            plt.figure(figsize=(20, 20))
            sns.heatmap(corr_matrix, cmap='plasma', center=0, annot=True)
            plt.title(f'{split_name} ({label}) - Overall Correlation')
            plt.tight_layout()
            plt.show()

    if isinstance(merged.index, pd.MultiIndex) and 'Ch1' in merged.index.get_level_values('Channel'):
        analyse_subset(merged.xs('Ch1', level='Channel', drop_level=False), 'Channel 1')
        
    if isinstance(merged.index, pd.MultiIndex) and 'Ch2' in merged.index.get_level_values('Channel'):
        analyse_subset(merged.xs('Ch2', level='Channel', drop_level=False), 'Channel 2')
        
    analyse_subset(merged, 'Overall (Pooled)')

def create_interactive_dashboard(data_catalog):

    mode_toggle = widgets.ToggleButtons(options=['Absolute', 'Stage Delta', 'Intra-stage Kinetics'], style={'description_width': 'initial'})
    channel_dropdown = widgets.Dropdown(options=['Channel 1', 'Channel 2', 'Both'], value='Channel 1', description='Channel:')
    sources = list(data_catalog['Absolute'].keys())

    x_source_drop = widgets.Dropdown(options=sources, value=sources[0], description='X Source:')
    y_source_drop = widgets.Dropdown(options=sources, value=sources[0], description='Y Source:')

    x_col_drop = widgets.Dropdown(description='X Metric:')
    y_col_drop = widgets.Dropdown(description='Y Metric:')
    
    out = widgets.Output()
    
    def update_dropdowns(*args):

        mode = mode_toggle.value
        x_src, y_src = x_source_drop.value, y_source_drop.value
        
        x_cols = list(data_catalog[mode][x_src].columns)
        y_cols = list(data_catalog[mode][y_src].columns)
        
        x_col_drop.options = x_cols

        if x_col_drop.value not in x_cols: 

            x_col_drop.value = x_cols[0] if x_cols else None
            
        y_col_drop.options = y_cols
        if y_col_drop.value not in y_cols: 

            y_col_drop.value = y_cols[0] if y_cols else None
            
    mode_toggle.observe(update_dropdowns, 'value')
    x_source_drop.observe(update_dropdowns, 'value')
    y_source_drop.observe(update_dropdowns, 'value')
    update_dropdowns()
    
    def get_joined_data(mode, x_src, y_src, x_col, y_col, channel_label):

        df_x = data_catalog[mode][x_src].copy()
        df_y = data_catalog[mode][y_src].copy()
        
        def slice_channel(df, ch_label):

            target_ch = 'Ch1' if ch_label == 'Channel 1' else 'Ch2'

            if isinstance(df.index, pd.MultiIndex) and 'Channel' in df.index.names:
                
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

                df = df.rename(index=ch_map, level='Channel')
                
                if target_ch in df.index.get_level_values('Channel'):
                    return df.xs(target_ch, level='Channel')
                else:
                    return pd.DataFrame()
                
            return df

        df_x = slice_channel(df_x, channel_label)
        df_y = slice_channel(df_y, channel_label)
        
        if df_x.empty or df_y.empty:
            return pd.DataFrame()

        df_x.columns = df_x.columns.astype(str)
        df_y.columns = df_y.columns.astype(str)
        x_col_str, y_col_str = str(x_col), str(y_col)
        
        df_merged = pd.merge(df_x[[x_col_str]], df_y[[y_col_str]], left_index=True, right_index=True, how='inner', suffixes=('_x', '_y'))
        
        actual_x_col = x_col_str + '_x' if x_col_str == y_col_str else x_col_str
        actual_y_col = y_col_str + '_y' if x_col_str == y_col_str else y_col_str
        df_merged = df_merged.rename(columns={actual_x_col: 'x_val', actual_y_col: 'y_val'})

        return df_merged.replace([np.inf, -np.inf], np.nan).dropna()

    def plot_data(*args):

        with out:

            out.clear_output(wait=True)

            mode, channel = mode_toggle.value, channel_dropdown.value
            x_src, y_src = x_source_drop.value, y_source_drop.value
            x_col, y_col = x_col_drop.value, y_col_drop.value
            
            if not x_col or not y_col:

                print('Please select valid metrics to plot.')

                return
                
            fig = go.Figure()
            channels_to_plot = ['Channel 1', 'Channel 2'] if channel == 'Both' else [channel]
            colors = {'Channel 1': '#1f77b4', 'Channel 2': '#ff1e0e'} 
            plotted_any = False
            
            for ch in channels_to_plot:

                try:
                    plot_df = get_joined_data(mode, x_src, y_src, x_col, y_col, ch)
                except Exception as e:
                    continue
                    
                if len(plot_df) < 2: 
                    continue

                plotted_any = True
                x_data, y_data = plot_df['x_val'], plot_df['y_val']
                
                fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name=f'{ch}', marker=dict(size=8, opacity=0.7, color=colors[ch], line=dict(width=1, color='DarkSlateGrey')), text=plot_df.index, hovertemplate='Chip ID: %{text}<br>X: %{x:.4f}<br>Y: %{y:.4f}<extra></extra>'))
                
                if x_data.nunique() > 1:

                    slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
                    x_fit = np.linspace(x_data.min(), x_data.max(), 100)
                    y_fit = slope * x_fit + intercept
                    fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name=f'{ch} Fit (r={r_value:.3f}, R^2={r_value**2:.3f})', line=dict(color=colors[ch], dash='dash', width=2), hoverinfo='skip'))            
            if not plotted_any:

                print('Not enough matching chip records to plot these selections.')

                return
                
            title = f'{y_src} [{y_col}] vs {x_src} [{x_col}]'
            fig.update_layout(title=title, xaxis_title=f'{x_src} : {x_col}', yaxis_title=f'{y_src} : {y_col}', template='plotly_white', height=600, margin=dict(l=40, r=40, t=60, b=40), hovermode='closest')
            fig.show()

    mode_toggle.observe(plot_data, 'value')
    channel_dropdown.observe(plot_data, 'value')
    x_source_drop.observe(plot_data, 'value')
    y_source_drop.observe(plot_data, 'value')
    x_col_drop.observe(plot_data, 'value')
    y_col_drop.observe(plot_data, 'value')
    
    controls = widgets.VBox([mode_toggle, channel_dropdown, widgets.HBox([x_source_drop, x_col_drop]), widgets.HBox([y_source_drop, y_col_drop])])
    
    display(widgets.HTML(f"<h3 style='margin-bottom:0px; color:#0000FF;'>Master Chip Comparison Dashboard</h3>"))
    display(controls, out)

    plot_data()

def get_top_drivers(df, chip_id, top_n=3):

    mean = df.mean()
    std = df.std().replace(0, 1e-9)
    
    if isinstance(df.index, pd.MultiIndex):

        if chip_id not in df.index.get_level_values('chip_id'): 
            return ['N/A (Chip not in dataset)']
        
        chip_rows = df.xs(chip_id, level='chip_id')
        drivers = []

        for ch, row in chip_rows.iterrows():

            z_scores = ((row - mean) / std).abs().dropna()

            if not z_scores.empty:

                top_stages = z_scores.sort_values(ascending=False).head(top_n)
                top_str = ' | '.join([f'{stage} (Z: {z:.1f})' for stage, z in top_stages.items()])
                drivers.append(f'[{ch}] {top_str}')

        return drivers
    else:

        if chip_id not in df.index: 
            return ['N/A (Chip not in dataset)']
        
        z_scores = ((df.loc[chip_id] - mean) / std).abs().dropna()

        if z_scores.empty: 
            return ['N/A']
        
        top_stages = z_scores.sort_values(ascending=False).head(top_n)

        return [f'{stage} (Z: {z:.1f})' for stage, z in top_stages.items()]

def generate_at_risk_summary(master_df, sc_metrics, sc_raw_df, all_at_risk_chips, section_config, title='Overall At-Risk Chips Summary'):
    '''
    Clusters chips based on combined stage data, cross-references against standard 
    curve metrics, and generates a structured summary report across 5 distinct sections.
    '''

    pel_unique = set()
    pel_breakdown = {}
    
    immob_unique = set()
    immob_breakdown = {}
    
    for section_name, (_, outliers) in section_config.items():

        if 'PEL' in section_name:

            pel_unique.update(outliers)

            if outliers: 
                pel_breakdown[section_name] = outliers
        elif 'Immob' in section_name:

            immob_unique.update(outliers)

            if outliers: 
                immob_breakdown[section_name] = outliers

    sc_fails = set()

    if sc_metrics is not None and not sc_metrics.empty and 'r2' in sc_metrics.columns:
        sc_fails = set(sc_metrics[sc_metrics['r2'] < 0.95].index)
        
    combined_all_fails = sorted(list(set(all_at_risk_chips).union(pel_unique, immob_unique, sc_fails)))

    with collapsible_output(f'{title} - Overall Summary'):

        print(f'Total Unique Faulty Chips: {len(combined_all_fails)}')

        if combined_all_fails:

            print(f'Chip IDs: {', '.join(map(str, combined_all_fails))}\n')
            print(f'Total Unique PEL Fails: {len(pel_unique)}')
            print(f'Total Unique Immobilisation Fails: {len(immob_unique)}')
            print(f'Total Unique Standard Curve Fails: {len(sc_fails)}')
        else:
            print('No faults detected across any stage. Everything looks normal!')

    with collapsible_output(f'{title} - PEL Breakdown'):

        print(f'Total Unique PEL Faulty Chips: {len(pel_unique)}')

        if pel_unique:

            for sec_name, out_list in pel_breakdown.items():
                print(f'  - {sec_name}: {len(out_list)} chips ({', '.join(map(str, out_list))})')

    with collapsible_output(f'{title} - Immobilisation Breakdown'):

        print(f'Total Unique Immobilisation Faulty Chips: {len(immob_unique)}')

        if immob_unique:

            for sec_name, out_list in immob_breakdown.items():
                print(f'  - {sec_name}: {len(out_list)} chips ({', '.join(map(str, out_list))})')

    with collapsible_output(f'{title} - Standard Curve Breakdown'):

        print(f'Total Unique Standard Curve Faulty Chips: {len(sc_fails)}')

        if sc_fails: 
            print(f'Chip IDs (R^2 < 0.95): {', '.join(map(str, sorted(list(sc_fails))))}')

    with collapsible_output(f'{title} - Chip Profiles'):

        if master_df.empty or not combined_all_fails:

            print('No detailed profiles available.')
            return

        features = master_df.columns.tolist()
        X_master = StandardScaler().fit_transform(master_df[features])
        kmeans = KMeans(n_clusters=2, random_state=8030, n_init=10)
        master_df['Cluster'] = kmeans.fit_predict(X_master)

        date_col = next((c for c in sc_raw_df.columns if 'date' in c.lower()), None)
        time_col = next((c for c in sc_raw_df.columns if 'time' in c.lower()), None)

        cols_to_merge = ['standard_curve_uuid']

        if date_col: 
            cols_to_merge.append(date_col)

        if time_col:
            cols_to_merge.append(time_col)

        chip_summaries = {}

        for chip in combined_all_fails:
            
            if chip in master_df.index.get_level_values('chip_id'):

                chip_clusters = master_df.xs(chip, level='chip_id')['Cluster'].to_dict()
                cluster_profile = ' | '.join([f'{ch}: Cluster {cl}' for ch, cl in chip_clusters.items()])
            else:
                cluster_profile = 'N/A (Incomplete Stage Data)'
                
            sc_rows = sc_metrics[sc_metrics.index == chip].copy()

            if not sc_rows.empty and len(cols_to_merge) > 1:
                sc_rows = sc_rows.merge(sc_raw_df[cols_to_merge], on='standard_curve_uuid', how='left')
            
            detailed_flags = []

            if not sc_rows.empty and (sc_rows['r2'] < 0.95).any():
                detailed_flags.append('Standard Curve (R² < 0.95)')
                
            drivers_info = {}

            for section_name, (df_source, outlier_list) in section_config.items():

                if chip in outlier_list:
                    detailed_flags.append(section_name)
                    drivers_info[section_name] = get_top_drivers(df_source, chip)
            
            sc_data = []

            if not sc_rows.empty:

                for _, sc_row in sc_rows.iterrows():

                    r2_val = sc_row['r2']
                    d_str = str(sc_row[date_col]).strip() if date_col and pd.notna(sc_row[date_col]) else ''
                    t_str = str(sc_row[time_col]).strip().replace('-', ':') if time_col and pd.notna(sc_row[time_col]) else ''
                    sc_data.append({
                        'Date/Time': f'{d_str} {t_str}'.strip() or 'Unknown',
                        'Curve UUID': sc_row['standard_curve_uuid'],
                        'Fit Model': sc_row['fit_model'],
                        'R² Score': r2_val,
                        'Status': 'FAILED' if r2_val < 0.95 else 'PASSED'
                    })

            sc_df = pd.DataFrame(sc_data)
            
            if not sc_df.empty and 'Date/Time' in sc_df.columns:

                sc_df['Parsed_DateTime'] = pd.to_datetime(sc_df['Date/Time'], errors='coerce')
                sc_df = sc_df.sort_values(by='Parsed_DateTime').drop(columns=['Parsed_DateTime']).reset_index(drop=True)
            
            chip_summaries[chip] = {
                'Cluster': cluster_profile,
                'Flagged By': detailed_flags,
                'Drivers': drivers_info,
                'SC_DF': sc_df
            }

        def style_status(val):
        
            if val == 'FAILED': 
                return 'color: red; font-weight: bold;'

            if val == 'PASSED': 
                return 'color: green;'

            return ''
        
        for chip, info in chip_summaries.items():

            print('=' * 80)
            print(f'CHIP ID: {chip}')
            print(f'Cluster Profile: {info['Cluster']}')
            
            flag_str = '\n    - '.join(info['Flagged By']) if info['Flagged By'] else 'None (Manual Review)'
            print(f'Flagged By Anomaly In:\n    - {flag_str}')
            
            if info['Drivers']:

                print('\nTop 3 Divergent Regions for Triggered Sections:')

                for section, drivers in info['Drivers'].items():

                    print(f'\n  - {section}:')

                    for d in drivers: 
                        print(f'      {d}')

            print('-' * 80)
            
            if not info['SC_DF'].empty:

                styled_df = info['SC_DF'].style.format({'R² Score': '{:.4f}'}).set_properties(**{'text-align': 'left', 'white-space': 'nowrap'}).set_table_styles([dict(selector='th', props=[('text-align', 'left')])]).map(style_status, subset=['Status'])

                display(styled_df)
            else:
                print('No Standard Curve data available for this chip.')
                
            print('\n')
