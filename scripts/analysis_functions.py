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

def detect_anomalies(df, feature_cols, contamination=0.05):
    """
    Applies Isolation Forest to detect multivariate outliers.
    Returns the dataframe with 'anomaly' (-1 for outlier, 1 for inlier) 
    and 'anomaly_score' columns.
    """

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
    return df.pivot_table(index='chip_id', columns=stage_col, values=val_col, aggfunc='mean', observed=False)

def summarise_correlations(corr_matrix, label, num=3):
    """Summarise strongest positive, strongest negative, and weakest correlations."""

    print(f"\n {label} Correlation Insights")

    corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).unstack().dropna()

    if corr.empty:
        print("Not enough data to compute correlations.")
        return

    positive = corr[corr > 0].sort_values(ascending=False)
    negative = corr[corr < 0].sort_values()
    weakest = corr.iloc[corr.abs().argsort()]

    print("Top Positive Correlations:")

    if len(positive):

        for (s1, s2), val in positive.head(num).items():
            print(f"  • {s1} & {s2} (r = {val:.3f})")
    else:
        print("  None")

    print("\nTop Negative (Inverse) Correlations:")

    if len(negative):

        for (s1, s2), val in negative.head(num).items():
            print(f"  • {s1} & {s2} (r = {val:.3f})")
    else:
        print("  None")

    print("\nWeakest Relationships (Closest to zero):")

    for (s1, s2), val in weakest.head(num).items():
        print(f"  • {s1} & {s2} (r = {val:.3f})")

def analyse_anomaly_drivers(df, anomalies_df, label, num=3):
    """Compares anomalous chips to normal chips to find which stages drive the anomaly."""

    outliers = anomalies_df[anomalies_df['anomaly'] == -1].index
    normals = anomalies_df[anomalies_df['anomaly'] == 1].index

    print(f"\n {label} Anomaly Stage Breakdown")

    if len(outliers) == 0:

        print("No anomalies detected. All chips are behaving within normal bounds.")
        return

    if len(normals) < 2:

        print("Not enough normal chips to form a baseline for comparison.")
        return

    normal_data = df.loc[normals]
    anomaly_data = df.loc[outliers]

    normal_mean = normal_data.mean()
    normal_std = normal_data.std().replace(0, 1e-9) 

    anomaly_mean = anomaly_data.mean()
    z_scores = ((anomaly_mean - normal_mean) / normal_std).abs().sort_values(ascending=False).dropna()

    print(f"Top {num} stages driving these anomalies (Largest deviation from normal):")

    for stage, z in z_scores.head(num).items():

        norm_val = normal_mean[stage]
        anom_val = anomaly_mean[stage]
        print(f"  • {stage}: Anomaly Avg = {anom_val:.3f} | Normal Avg = {norm_val:.3f} (Z-Score: {z:.2f})")

def create_wide_intra(intra_df, val_col):
    """Pivots multiple intra-stage metrics and suffixes columns to prevent overlap."""

    intra_metrics = [f'{val_col}_std', f'{val_col}_auc', f'{val_col}_stability', f'{val_col}_init_slope']
    wide_list = []
    
    for metric in intra_metrics:

        if metric in intra_df.columns:

            wide_metric = pivot_chip_data(intra_df, 'stage', metric)
            metric_suffix = metric.replace(f'{val_col}_', '')
            wide_metric.columns = [f"{col}_{metric_suffix}" for col in wide_metric.columns]
            wide_list.append(wide_metric)
            
    return pd.concat(wide_list, axis=1) if wide_list else pd.DataFrame()

def analyse_pel(events_df, changes_df, intra_df, val_col='quad_ch1', change_col='quad_ch1_change', title="PEL Analysis"):

    events_df = events_df[~events_df['stage'].isin(['Start', 'Initial'])]
    
    wide_events = pivot_chip_data(events_df, 'stage', val_col)
    wide_changes = pivot_chip_data(changes_df, 'stage', change_col)
    wide_intra = create_wide_intra(intra_df, val_col)

    anomalies_abs = detect_anomalies(wide_events, wide_events.columns.tolist(), contamination=0.05)
    outliers_abs = anomalies_abs[anomalies_abs['anomaly'] == -1].index.tolist() if not anomalies_abs.empty else []

    anomalies_delta = detect_anomalies(wide_changes, wide_changes.columns.tolist(), contamination=0.05)
    outliers_delta = anomalies_delta[anomalies_delta['anomaly'] == -1].index.tolist() if not anomalies_delta.empty else []

    outliers_intra = []
    if not wide_intra.empty:
        anomalies_intra = detect_anomalies(wide_intra, wide_intra.columns.tolist(), contamination=0.05)
        outliers_intra = anomalies_intra[anomalies_intra['anomaly'] == -1].index.tolist() if not anomalies_intra.empty else []

    with collapsible_output(f"{title} - Absolute Signal"):

        plt.figure(figsize=(10, 6))
        sns.heatmap(wide_events.corr(), cmap='plasma', center=0, annot=True)
        plt.title(f"PEL [{val_col}]: Absolute Signal Correlation")
        plt.tight_layout()
        plt.show()

        summarise_correlations(wide_events.corr(), f"PEL {val_col} Absolute Signal")
        print(f"\nPotential Anomalous Chips (Absolute): {len(outliers_abs)}")
        analyse_anomaly_drivers(wide_events, anomalies_abs, f"PEL {val_col} Absolute Signal")

    with collapsible_output(f"{title} - Stage Delta"):

        plt.figure(figsize=(10, 6))
        sns.heatmap(wide_changes.corr(), cmap='plasma', center=0, annot=True)
        plt.title(f"PEL [{change_col}]: Stage Delta Correlation")
        plt.tight_layout()
        plt.show()

        summarise_correlations(wide_changes.corr(), f"PEL {change_col} Stage Delta")
        print(f"\nPotential Anomalous Chips (Delta): {len(outliers_delta)}")
        analyse_anomaly_drivers(wide_changes, anomalies_delta, f"PEL {change_col} Stage Delta")
    
    if not wide_intra.empty:

        with collapsible_output(f"{title} - Intra-stage Kinetics"):

            plt.figure(figsize=(10, 6))
            sns.heatmap(wide_intra.corr(), cmap='plasma', center=0, annot=True)
            plt.title(f"PEL [{val_col}]: Intra-stage Kinetics Correlation")
            plt.tight_layout()
            plt.show()

            summarise_correlations(wide_intra.corr(), f"PEL {val_col} Intra-stage Kinetics")
            print(f"\nPotential Anomalous Chips (Intra-stage Kinetics): {len(outliers_intra)}")
            analyse_anomaly_drivers(wide_intra, anomalies_intra, f"PEL {val_col} Intra-stage Kinetics")
        
    return wide_events, wide_changes, wide_intra, outliers_abs, outliers_delta, outliers_intra

def analyse_immob_split(events_df, changes_df, intra_df, split_name, val_col='channel1', change_col='channel1_change', title="Immobilisation Analysis"):
    events_df = events_df[~events_df['stage'].isin(['Start', 'Initial'])]

    wide_events = pivot_chip_data(events_df, 'stage', val_col)
    wide_changes = pivot_chip_data(changes_df, 'stage', change_col)
    wide_intra = create_wide_intra(intra_df, val_col)

    anomalies_abs = detect_anomalies(wide_events, wide_events.columns.tolist(), contamination=0.05)
    outliers_abs = anomalies_abs[anomalies_abs['anomaly'] == -1].index.tolist() if not anomalies_abs.empty else []

    anomalies_delta = detect_anomalies(wide_changes, wide_changes.columns.tolist(), contamination=0.05)
    outliers_delta = anomalies_delta[anomalies_delta['anomaly'] == -1].index.tolist() if not anomalies_delta.empty else []

    outliers_intra = []

    if not wide_intra.empty:

        anomalies_intra = detect_anomalies(wide_intra, wide_intra.columns.tolist(), contamination=0.05)
        outliers_intra = anomalies_intra[anomalies_intra['anomaly'] == -1].index.tolist() if not anomalies_intra.empty else []

    with collapsible_output(f"{title} - Absolute Signal"):

        plt.figure(figsize=(20, 20))
        sns.heatmap(wide_events.corr(), cmap='plasma', annot=True, annot_kws={"size": 12})
        plt.title(f"Immob [{split_name}] - {val_col}: Absolute Signal Correlation")
        plt.tight_layout()
        plt.show()

        summarise_correlations(wide_events.corr(), f"Immob [{split_name}] {val_col} Absolute Signal")
        print(f"\nPotential Anomalous Chips (Absolute): {len(outliers_abs)}")
        analyse_anomaly_drivers(wide_events, anomalies_abs, f"Immob [{split_name}] {val_col} Absolute Signal")

    with collapsible_output(f"{title} - Stage Delta"):

        plt.figure(figsize=(20, 20))
        sns.heatmap(wide_changes.corr(), cmap='plasma', center=0, annot=True)
        plt.title(f"Immob [{split_name}] - {change_col}: Stage Delta Correlation")
        plt.show()

        summarise_correlations(wide_changes.corr(), f"Immob [{split_name}] {change_col} Stage Delta")
        print(f"\nPotential Anomalous Chips (Delta): {len(outliers_delta)}")
        analyse_anomaly_drivers(wide_changes, anomalies_delta, f"Immob [{split_name}] {change_col} Stage Delta")
    
    if not wide_intra.empty:

        with collapsible_output(f"{title} - Intra-stage Kinetics"):

            plt.figure(figsize=(20, 20))
            sns.heatmap(wide_intra.corr(), cmap='plasma', center=0, annot=True)
            plt.title(f"Immob [{split_name}] - {val_col}: Intra-stage Kinetics Correlation")
            plt.show()

            summarise_correlations(wide_intra.corr(), f"Immob [{split_name}] {val_col} Intra-stage Kinetics")
            print(f"\nPotential Anomalous Chips (Intra-stage Kinetics): {len(outliers_intra)}")
            analyse_anomaly_drivers(wide_intra, anomalies_intra, f"Immob [{split_name}] {val_col} Intra-stage Kinetics")
        
    return wide_events, wide_changes, wide_intra, outliers_abs, outliers_delta, outliers_intra

def plot_pel_layer_shifts(df, channel_name="Channel 1", title="PEL Layer Shifts"):
    
    with collapsible_output(title):

        layer_cols = [col for col in df.columns if col != 'Start -> End']
        fig = go.Figure()
        
        for chip_id, row in df.iterrows():

            fig.add_trace(go.Scatter(x=layer_cols, y=row[layer_cols], mode='lines+markers', name=str(chip_id), opacity=0.6, marker=dict(size=6), visible='legendonly'))

        fig.update_layout(title=f"PEL Layer Build-up: Shift per Transition ({channel_name})", xaxis_title="Layer Transition", yaxis_title="Signal Shift (Δ)", template="plotly_white", hovermode="x unified", height=600)
        fig.update_xaxes(tickangle=45)
        fig.show()

        plt.figure(figsize=(10, 6))

        for chip_id, row in df.iterrows():
            plt.plot(layer_cols, row[layer_cols], marker='o', alpha=0.5, label=chip_id)

        plt.title(f"PEL Layer Build-up: Shift per Transition ({channel_name})")
        plt.xlabel("Layer Transition")
        plt.ylabel("Signal Shift (Δ)")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        if len(df) <= 25:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()

def combine_anomalies(*outliers_lists, title="Combined Anomalies Summary"):
    """Combines an arbitrary number of outlier lists into a single set."""

    all_outlier_chips = set()

    for outliers in outliers_lists:
        all_outlier_chips.update(outliers)
        
    all_outlier_chips = list(all_outlier_chips)

    with collapsible_output(title):

        print(f"Total Unique Anomalous Chips: {len(all_outlier_chips)}")

        if all_outlier_chips:
            print(f"Chip IDs: {all_outlier_chips}")
        else:
            print("No anomalies detected in this combination.")

    return all_outlier_chips

def analyse_cross_stage_split(pel_df, immob_df, split_name, channel_name, pel_outliers, immob_outliers, sc_metrics_df=None, title="Cross-Stage Validation"):
    """
    Merges PEL and a specific Immobilisation split, runs PCA & Clustering,
    plots the results with anomaly highlights. Validates against Standard Curves if provided (Channel 1).
    """
    
    pel_copy = pel_df.copy()
    immob_copy = immob_df.copy()
    
    pel_copy.columns = pel_copy.columns.astype(str)
    immob_copy.columns = immob_copy.columns.astype(str)
    
    merged = pd.merge(pel_copy, immob_copy, left_index=True, right_index=True, how='inner', suffixes=('_PEL', '_Immob')).dropna()
    
    if merged.empty:
        print(f"[{split_name} - {channel_name}] Not enough overlapping chips between PEL and Immobilisation for cross-analysis.")
        return
        
    with collapsible_output(title):

        print(f"[{split_name} - {channel_name}] Overlapping chips analysed: {len(merged)}")
        
        features = merged.columns.tolist()
        X_merged = StandardScaler().fit_transform(merged[features])
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_merged)
        
        merged['PCA1'] = pca_result[:, 0]
        merged['PCA2'] = pca_result[:, 1]
        
        print(f"[{split_name} - {channel_name}] Explained Variance by first 2 components: {pca.explained_variance_ratio_.sum()*100:.2f}%")
        
        kmeans = KMeans(n_clusters=2, random_state=8030, n_init=10)
        merged['Cluster'] = kmeans.fit_predict(X_merged)
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=merged, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', s=100, alpha=0.7)
        
        both_anomalies = set(pel_outliers).union(set(immob_outliers))
        anomaly_mask = merged.index.isin(both_anomalies)
        
        if anomaly_mask.any():
            plt.scatter(merged.loc[anomaly_mask, 'PCA1'], merged.loc[anomaly_mask, 'PCA2'], edgecolor='black', facecolor='none', s=200, label='Flagged Anomaly', linewidth=2)
                        
        plt.title(f"Cross-Stage PCA: {split_name} ({channel_name})")
        plt.xlabel("Principal Component 1 (General Signal Variance)")
        plt.ylabel("Principal Component 2 (Stage-to-Stage Dynamic Variance)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print(f"\n Cluster Profiling [{split_name} - {channel_name}]")
        cluster_means = merged[features].groupby(merged['Cluster']).mean()
        overall_mean = merged[features].mean()
        overall_std = merged[features].std().replace(0, 1e-9)
        
        cluster_zscores = (cluster_means - overall_mean) / overall_std
        
        for cluster_id in sorted(merged['Cluster'].unique()):

            print(f"\nCluster {cluster_id} Profile (n={sum(merged['Cluster'] == cluster_id)} chips):")
            
            top_pos = cluster_zscores.loc[cluster_id].sort_values(ascending=False).head(3)
            top_neg = cluster_zscores.loc[cluster_id].sort_values(ascending=True).head(3)
            
            print("  Defining High Features (Above Average):")

            pos_found = False

            for feat, z in top_pos.items():

                if z > 0.5:

                    print(f"    - {feat}: +{z:.2f} standard deviations")
                    pos_found = True

            if not pos_found: 
                print("    - None significantly above average")
                
            print("  Defining Low Features (Below Average):")
            neg_found = False

            for feat, z in top_neg.items():

                if z < -0.5:
                    print(f"    - {feat}: {z:.2f} standard deviations")
                    neg_found = True

            if not neg_found: 
                print("    - None significantly below average")
        
        if sc_metrics_df is not None:

            print(f"\n--- Cluster Functional Yield (R² Comparison) [{split_name} - {channel_name}] ---")
            
            cluster_mapping = merged[['Cluster']].reset_index().rename(columns={'index': 'chip_id'})
            sc_cluster_df = pd.merge(sc_metrics_df, cluster_mapping, on='chip_id', how='inner')
            
            if sc_cluster_df.empty:
                print("No Standard Curve data available to map to clusters.")
            else:

                cluster_stats = sc_cluster_df.groupby('Cluster')['r2'].agg(['count', 'mean', 'median', 'std']).fillna(0)
                print("Standard Curve R² Performance by Cluster:")
                display(cluster_stats)
                
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=sc_cluster_df, x='Cluster', y='r2', showmeans=True,  meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
                sns.stripplot(data=sc_cluster_df, x='Cluster', y='r2', color='black', alpha=0.5, jitter=True)
                plt.axhline(0.95, color='red', linestyle='--', label='Pass Threshold (0.95)')
                
                plt.title(f"Standard Curve R² Distribution by Manufacturing Cluster\n{split_name} ({channel_name})")
                plt.xlabel("Manufacturing Cluster")
                plt.ylabel("Standard Curve R² Score")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
                
            anomaly_ids = list(both_anomalies.intersection(set(merged.index)))
            
            if not anomaly_ids:
                print(f"[{split_name} - {channel_name}] No anomalous chips from the cross-stage analysis to validate against SC.")
            else:
                print(f"\n[{split_name} - {channel_name}] Validating functional performance for anomalous chip(s): {anomaly_ids}\n")
                
                anomaly_sc_data = sc_metrics_df[sc_metrics_df.index.isin(anomaly_ids)]
                
                if anomaly_sc_data.empty:
                    print("No Standard Curve data found for the flagged anomalous chip(s).")
                else:

                    print(f"{'Chip ID':<15} | {'Curve UUID':<40} | {'R² Score':<10} | {'Status'}")
                    print("-" * 80)
                    
                    for chip_id, row in anomaly_sc_data.iterrows():

                        r2 = row['r2']
                        curve_id = row['standard_curve_uuid']
                        
                        status = "PASSED" if r2 >= 0.95 else "FAILED"
                        print(f"{chip_id:<15} | {curve_id:<40} | {r2:<10.4f} | {status}")
            
            normal_sc_data = sc_metrics_df[~sc_metrics_df.index.isin(anomaly_ids)]

            if not normal_sc_data.empty:
                print(f"\n>>> Average R² for NORMAL chips: {normal_sc_data['r2'].mean():.4f}")
        else:

            anomaly_ids = list(both_anomalies.intersection(set(merged.index)))
            print(f"\n--- Anomaly Identifications [{split_name} - {channel_name}] ---")
            if anomaly_ids:
                print(f"Flagged Anomalous Chips in cluster: {anomaly_ids}")
            else:
                print("No anomalous chips detected in this cross-stage split.")

def print_correlation_insights(corr_matrix, split_name, channel_name, top_n=3):
    """Extracts and prints top positive, negative, and weakest correlations."""

    print(f"{split_name} ({channel_name}) Signal Correlation Insights")
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_pairs = upper_tri.stack().dropna()
    
    if corr_pairs.empty:
        print("Not enough variance to calculate correlation insights.\n")
        return

    print("Top Positive Correlations:")
    top_pos = corr_pairs[corr_pairs > 0].sort_values(ascending=False).head(top_n)

    if top_pos.empty: print("  None")
    else:
        for (feat1, feat2), val in top_pos.items(): print(f"  - {feat1} & {feat2} (r = {val:.3f})")

    print("\nTop Negative (Inverse) Correlations:")
    top_neg = corr_pairs[corr_pairs < 0].sort_values(ascending=True).head(top_n)

    if top_neg.empty: print("  None")
    else:
        for (feat1, feat2), val in top_neg.items(): print(f"  - {feat1} & {feat2} (r = {val:.3f})")

    print("\nWeakest Relationships (Closest to zero):")
    weakest = corr_pairs.reindex(corr_pairs.abs().sort_values().index).head(top_n)

    for (feat1, feat2), val in weakest.items():
        print(f"  - {feat1} & {feat2} (r = {val:.3f})")
    print("\n")

def cross_stage_correlations(pel_df, immob_df, split_name, channel_name, sc_metrics_df=None, title="Cross-Stage Correlations"):

    pel_copy = pel_df.copy()
    immob_copy = immob_df.copy()

    pel_copy.columns = pel_copy.columns.astype(str)
    immob_copy.columns = immob_copy.columns.astype(str)

    merged = pel_copy.merge(immob_copy, left_index=True, right_index=True, how="inner", suffixes=("_PEL", "_Immob"))

    if sc_metrics_df is not None:

        sc_copy = sc_metrics_df.copy()
        sc_copy.columns = sc_copy.columns.astype(str)
        numeric_cols = sc_copy.select_dtypes(include="number").columns.tolist()
        
        if "datetime" in sc_copy.columns: 
            sc_copy = sc_copy[["datetime"] + numeric_cols]
        else: 
            sc_copy = sc_copy[numeric_cols]
            
        sc_first = sc_copy.sort_values("datetime").groupby(sc_copy.index, sort=False).first().drop(columns="datetime")
        merged = merged.merge(sc_first, left_index=True, right_index=True, how="left")

    with collapsible_output(title):

        if merged.empty:
            print(f"[{split_name} - {channel_name}] Not enough overlapping chips for cross-analysis.")
            return

        print(f"[{split_name} - {channel_name}] Overlapping chips analysed: {len(merged)}\n")

        corr_matrix = merged.corr()
        print_correlation_insights(corr_matrix, split_name, channel_name, top_n=3)

        plt.figure(figsize=(20, 20))
        sns.heatmap(corr_matrix, cmap='plasma', center=0, annot=True)
        plt.title(f"{split_name} ({channel_name}) - Overall Correlation")
        plt.tight_layout()
        plt.show()

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
        x_cols = list(data_catalog[mode][x_src]['Channel 1'].columns)
        y_cols = list(data_catalog[mode][y_src]['Channel 1'].columns)
        
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
    
    def get_joined_data(mode, x_src, y_src, x_col, y_col, channel):

        x_ch = channel if channel in data_catalog[mode][x_src] else 'Channel 1'
        y_ch = channel if channel in data_catalog[mode][y_src] else 'Channel 1'
        
        df_x = data_catalog[mode][x_src][x_ch].copy()
        df_y = data_catalog[mode][y_src][y_ch].copy()
        
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

                print("Please select valid metrics to plot.")
                return
                
            fig = go.Figure()
            channels_to_plot = ['Channel 1', 'Channel 2'] if channel == 'Both' else [channel]
            colors = {'Channel 1': '#1f77b4', 'Channel 2': "#ff1e0e"} 
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
                
                fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name=f'{ch}', marker=dict(size=8, opacity=0.7, color=colors[ch], line=dict(width=1, color='DarkSlateGrey')), text=plot_df.index, hovertemplate="Chip ID: %{text}<br>X: %{x:.4f}<br>Y: %{y:.4f}<extra></extra>"))
                
                if x_data.nunique() > 1:

                    slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
                    x_fit = np.linspace(x_data.min(), x_data.max(), 100)
                    y_fit = slope * x_fit + intercept
                    fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name=f'{ch} Fit (R²={r_value**2:.3f})', line=dict(color=colors[ch], dash='dash', width=2), hoverinfo='skip'))
            
            if not plotted_any:

                print("Not enough matching chip records to plot these selections.")
                return
                
            title = f"{y_src} [{y_col}] vs {x_src} [{x_col}]"
            fig.update_layout(title=title, xaxis_title=f"{x_src} : {x_col}", yaxis_title=f"{y_src} : {y_col}", template='plotly_white', height=600, margin=dict(l=40, r=40, t=60, b=40), hovermode="closest")
            fig.show()

    mode_toggle.observe(plot_data, 'value')
    channel_dropdown.observe(plot_data, 'value')
    x_source_drop.observe(plot_data, 'value')
    y_source_drop.observe(plot_data, 'value')
    x_col_drop.observe(plot_data, 'value')
    y_col_drop.observe(plot_data, 'value')
    
    plot_data()
    controls = widgets.VBox([mode_toggle, channel_dropdown, widgets.HBox([x_source_drop, x_col_drop]), widgets.HBox([y_source_drop, y_col_drop])])
    
    display(widgets.HTML(f"<h3 style='margin-bottom:0px; color:#0000FF;'>Master Chip Comparison Dashboard</h3>"))
    display(controls, out)

def get_top_drivers(df, chip_id, top_n=3):

    if chip_id not in df.index: 
        return ["N/A (Chip not in dataset)"]
    
    mean = df.mean()
    std = df.std().replace(0, 1e-9)
    z_scores = ((df.loc[chip_id] - mean) / std).abs().dropna()

    if z_scores.empty: 
        return ["N/A"]
    
    top_stages = z_scores.sort_values(ascending=False).head(top_n)

    return [f"{stage} (Z: {z:.1f})" for stage, z in top_stages.items()]

def prepare_section(df, stage_col, val_col):

    pivoted = pivot_chip_data(df, stage_col, val_col)
    anomalies = detect_anomalies(pivoted, pivoted.columns.tolist(), contamination=0.05)
    outliers = anomalies[anomalies['anomaly'] == -1].index.tolist() if not anomalies.empty else []

    return pivoted, outliers

def generate_at_risk_summary(master_df, sc_metrics, sc_raw_df, all_at_risk_chips, section_config, title="Overall At-Risk Chips Summary"):
    """
    Clusters chips based on combined stage data, cross-references against standard 
    curve metrics, and generates a structured summary report across 5 distinct sections.
    """

    pel_unique = set()
    pel_breakdown = {}
    
    immob_unique = set()
    immob_breakdown = {}
    
    for section_name, (df_source, outliers) in section_config.items():

        if "PEL" in section_name:

            pel_unique.update(outliers)

            if outliers: 
                pel_breakdown[section_name] = outliers
        elif "Immob" in section_name:

            immob_unique.update(outliers)

            if outliers: 
                immob_breakdown[section_name] = outliers

    sc_fails = set()

    if sc_metrics is not None and not sc_metrics.empty and 'r2' in sc_metrics.columns:
        sc_fails = set(sc_metrics[sc_metrics['r2'] < 0.95].index)
        
    combined_all_fails = set(all_at_risk_chips).union(pel_unique).union(immob_unique).union(sc_fails)
    combined_all_fails = sorted(list(combined_all_fails))

    with collapsible_output(f"{title} - Overall Summary"):

        print(f"Total Unique Faulty Chips: {len(combined_all_fails)}")

        if combined_all_fails:

            print(f"Chip IDs: {', '.join(map(str, combined_all_fails))}\n")
            print(f"Total Unique PEL Fails: {len(pel_unique)}")
            print(f"Total Unique Immobilisation Fails: {len(immob_unique)}")
            print(f"Total Unique Standard Curve Fails: {len(sc_fails)}")
        else:
            print("No faults detected across any stage. Everything looks normal!")

    with collapsible_output(f"{title} - PEL Breakdown"):

        print(f"Total Unique PEL Faulty Chips: {len(pel_unique)}")

        if pel_unique:

            print(f"Chip IDs: {', '.join(map(str, sorted(list(pel_unique))))}\n")
            print("Where they came from (Channel & Metric)")

            for sec_name, out_list in pel_breakdown.items():
                print(f"  • {sec_name}: {len(out_list)} chips ({', '.join(map(str, out_list))})")
        else:
            print("No anomalies detected in the PEL stage.")

    with collapsible_output(f"{title} - Immobilisation Breakdown"):

        print(f"Total Unique Immobilisation Faulty Chips: {len(immob_unique)}")

        if immob_unique:

            print(f"Chip IDs: {', '.join(map(str, sorted(list(immob_unique))))}\n")
            print("Where they came from (Split, Channel & Metric)")

            for sec_name, out_list in immob_breakdown.items():
                print(f"  • {sec_name}: {len(out_list)} chips ({', '.join(map(str, out_list))})")
        else:
            print("No anomalies detected in the Immobilisation stage.")

    with collapsible_output(f"{title} - Standard Curve Breakdown"):

        print(f"Total Unique Standard Curve Faulty Chips: {len(sc_fails)}")

        if sc_fails:
            print(f"Chip IDs (R² < 0.95): {', '.join(map(str, sorted(list(sc_fails))))}")
        else:
            print("No Standard Curve anomalies detected (All R² >= 0.95).")

    with collapsible_output(f"{title} - Chip Profiles"):

        if master_df.empty:
            print("Cannot generate detailed summary: Master dataframe is empty (no overlapping chips).")
            return
            
        if not combined_all_fails:
            print("No faulty chips to display profiles for.")
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

            cluster_profile = master_df.loc[chip, 'Cluster'] if chip in master_df.index else "N/A (Incomplete Stage Data)"
                
            sc_rows = sc_metrics[sc_metrics.index == chip].copy()

            if not sc_rows.empty and len(cols_to_merge) > 1:
                sc_rows = sc_rows.merge(sc_raw_df[cols_to_merge], on='standard_curve_uuid', how='left')
            
            detailed_flags = []
            if not sc_rows.empty and (sc_rows['r2'] < 0.95).any():
                detailed_flags.append("Standard Curve (R² < 0.95) [Channel 1 Only]")
                
            drivers_info = {}
            for section_name, (df_source, outlier_list) in section_config.items():

                if chip in outlier_list:
                    detailed_flags.append(section_name)
                    drivers_info[section_name] = get_top_drivers(df_source, chip)
            
            sc_data = []
            if not sc_rows.empty:

                for _, sc_row in sc_rows.iterrows():

                    r2_val = sc_row['r2']
                    d_str = str(sc_row[date_col]).strip() if date_col and pd.notna(sc_row[date_col]) else ""
                    t_str = str(sc_row[time_col]).strip().replace('-', ':') if time_col and pd.notna(sc_row[time_col]) else ""
                    datetime_str = f"{d_str} {t_str}".strip()
                    
                    sc_data.append({
                        "Date/Time": datetime_str if datetime_str else "Unknown",
                        "Curve UUID": sc_row['standard_curve_uuid'],
                        "Fit Model": sc_row['fit_model'],
                        "R² Score": r2_val,
                        "Status": "FAILED" if r2_val < 0.95 else "PASSED"
                    })
                    
            sc_df = pd.DataFrame(sc_data)

            if not sc_df.empty and "Date/Time" in sc_df.columns:

                sc_df["Parsed_DateTime"] = pd.to_datetime(sc_df["Date/Time"], errors='coerce')
                sc_df = sc_df.sort_values(by="Parsed_DateTime").drop(columns=["Parsed_DateTime"]).reset_index(drop=True)
            
            chip_summaries[chip] = {
                "Cluster": cluster_profile,
                "Flagged By": detailed_flags,
                "Drivers": drivers_info,
                "SC_DF": sc_df
            }

        if chip_summaries:

            def style_status(val):

                if val == 'FAILED': 
                    return 'color: red; font-weight: bold;'

                if val == 'PASSED': 
                    return 'color: green;'

                return ''
            
            for chip, info in chip_summaries.items():

                print("=" * 80)
                print(f"CHIP ID: {chip}")
                print(f"Cluster Profile: {info['Cluster']}")
                
                flag_str = "\n    - ".join(info['Flagged By']) if info['Flagged By'] else "None (Manual Review)"
                print(f"Flagged By Anomaly In:\n    - {flag_str}")
                
                if info['Drivers']:

                    print("\nTop 3 Divergent Regions for Triggered Sections:")

                    for section, drivers in info['Drivers'].items():
                        print(f"  • {section}: {' | '.join(drivers)}")

                print("-" * 80)
                
                if not info['SC_DF'].empty:

                    styled_df = info['SC_DF'].style.format({"R² Score": "{:.4f}"}).set_properties(**{'text-align': 'left', 'white-space': 'nowrap'}).set_table_styles([dict(selector='th', props=[('text-align', 'left')])]).map(style_status, subset=['Status'])

                    display(styled_df)
                else:
                    print("No Standard Curve data available for this chip.")
                    
                print("\n")

