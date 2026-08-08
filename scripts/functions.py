# Imports
import os
import re
import builtins
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.integrate import trapezoid
from IPython.display import display, HTML
from IPython.utils.io import capture_output

# Data Quality & Visualisation
def check_data_quality(dfs_dict, stage_name):
    """Prints missing values, duplicates, and summary stats for generic sensorgrams."""

    for name, df in dfs_dict.items():
        
        print(f'Metrics for {stage_name} - {name}:\n')
        print(f'\tMissing values: {df.isnull().sum().sum()}')
        print(f'\tDuplicated rows: {df.duplicated().sum()}\n')

        channels = df.columns.tolist()[1:]

        for chip in channels:

            print(f'\t{chip.capitalize()} Metrics:')
            display(df[chip].describe())
            display(df[chip].head())
            display(df[chip].tail())

def plot_sensorgrams(dfs_dict, stage_name):
    """Plots raw sensorgrams dynamically finding time and channel columns."""

    for name, df in dfs_dict.items():

        time_col = df.columns[0]
        channels = df.columns.tolist()[1:]

        plt.figure(figsize=(10, 6))
        
        for ch in channels:
            plt.plot(df[time_col], df[ch], label=ch)

        plt.title(f'{stage_name} Data for {name}')
        plt.xlabel('Time')
        plt.ylabel('RU')
        plt.legend()
        plt.show()

def plot_flags_on_sensorgrams(df_sens, times, labels, title, colours=None):
    """Overlays vertical flag lines and text on a generic sensorgram."""

    time_col = df_sens.columns[0]
    channels = df_sens.columns.tolist()[1:]
    colours = colours or ['k'] * len(times)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    for t, label, c, i in zip(times, labels, colours, range(len(times))):
        
        ax.axvline(x=t, linestyle='--', color=c)
        y_pos = 0.95 if i % 2 != 0 else 0.45
        ax.text(t, y_pos, label, rotation=45, transform=ax.get_xaxis_transform(), verticalalignment='top', color=c)

    for ch in channels:
        ax.plot(df_sens[time_col], df_sens[ch], label=ch)

    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('RU')
    ax.legend()
    plt.show()

def plot_5s_averaged_windows(sens, avg_flags, title_id):
    """Helper specific to Immob to plot avg windows."""

    time_col = sens.columns[0]
    channels = sens.columns.tolist()[1:]
    
    plt.figure(figsize=(10, 6))

    for ch in channels:
        plt.plot(sens[time_col], sens[ch], label=ch)
        
    for _, row in avg_flags.iterrows():

        w_start, w_end = row['Window_Start'], row['Window_End']
        plt.axvspan(w_start, w_end, color='black', alpha=0.2, linestyle='--')
        
        t_flag = (float(w_start) + float(w_end)) / 2
        avg_val = row.get(f'{channels[0]}_avg', 0)
        plt.text(t_flag, avg_val, f" {row['information']} ({row['Window_Label']})", fontsize=8, verticalalignment='bottom', rotation=15, color='darkred')
                 
    plt.title(f"Averaging Windows: {title_id}")
    plt.xlabel("Time (s)")
    plt.ylabel("Signal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def plot_all_statistics(df, x_col, y_col, ylabel='Signal', title_prefix=''):
    """
    Generates Box, Violin, Density, Correlation, and Heatmap plots 
    sequentially for a sensorgram stage metrics or changes dataframe.
    """
    
    title_prefix = f"{title_prefix} - " if title_prefix else ""
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=x_col, y=y_col)
    sns.stripplot(data=df, x=x_col, y=y_col, color='black', alpha=0.5, jitter=False)
    plt.xticks(rotation=90)
    plt.ylabel(ylabel)
    plt.title(f"{title_prefix}Stage Boxplot & Stripplot")
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x=x_col, y=y_col, palette='muted')
    plt.xticks(rotation=45)
    plt.ylabel(ylabel)
    plt.title(f"{title_prefix}Stage Violin Plot")
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x=y_col, hue=x_col, fill=True, common_norm=False, alpha=0.3)
    plt.xlabel(ylabel)
    plt.title(f"{title_prefix}Signal Density Profiles Per Stage")
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    corr_matrix = df.corr(numeric_only=True)

    if not corr_matrix.empty:

        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
        plt.title(f"{title_prefix}Channel & Time Correlation Matrix")
        plt.tight_layout()
        plt.show()

    plt.close()

    if 'chip_id' in df.columns:

        plt.figure(figsize=(10, 6))

        pivot_df = df.pivot_table(index='chip_id', columns=x_col, values=y_col, aggfunc='mean')
        
        sns.heatmap(pivot_df, annot=False, cmap='viridis', fmt='.2f', cbar_kws={'label': ylabel})
        plt.xticks(rotation=45)
        plt.ylabel('Chip ID')
        plt.xlabel('Experimental Stage')
        plt.title(f"{title_prefix}Overview Heatmap (Mean Value per Chip per Stage)")
        plt.tight_layout()
        plt.show()
        plt.close()

# Data Processing & Feature Extraction
def build_stage_summary(df, group_col, val_cols):
    """Builds statistical summaries mapping over given channel columns."""

    summary = df.groupby(group_col, observed=True).agg(
        n=('chip_id', 'count'),
        **{
            f'{col}_{stat}': (col, func)
            for col in val_cols
            for stat, func in [
                ('mean', 'mean'),
                ('median', 'median'),
                ('std', 'std'),
                ('min', 'min'),
                ('max', 'max'),
                ('iqr', lambda x: x.quantile(0.75) - x.quantile(0.25)),
            ]
        }
    ).reset_index()

    for col in val_cols:
        summary[f'{col}_cv_pct'] = 100 * summary[f'{col}_std'] / summary[f'{col}_mean'].abs()

    return summary

def add_stage_changes(event_table, channels):
    '''Calculates differences dynamically for provided channel columns and labels stage transitions'''

    for ch in channels:
        event_table[f'{ch}_change'] = event_table[ch].diff()

    event_table['stage'] = event_table['stage'].shift(1) + " -> " + event_table['stage']

    event_table = event_table.iloc[1:].reset_index(drop=True)

    return event_table

def extract_flags_from_pushes(flag_df):
    """
    Parses a flag dataframe to extract Buffers, Reagents, and their Plateaus 
    based on the syringe push event sequences.
    """

    df = flag_df.copy().sort_values(by='time').reset_index(drop=True)
    
    is_push = df['information'].str.contains("No 'Concentration'", na=False)
    new_records = []

    reagent_indices = df.index[~is_push].tolist()
    
    if not reagent_indices:
        first_reagent_time = float('inf')
        last_reagent_plateau_time = float('inf')
    else:
        first_reagent_time = df.loc[reagent_indices[0], 'time']
        last_reagent_plateau_time = df.loc[reagent_indices[-1], 'time']

    for idx in reagent_indices:

        reagent_name = df.loc[idx, 'information']
        
        if idx + 1 < len(df):

            plateau_time = df.loc[idx + 1, 'time']
            new_records.append({'time': plateau_time, 'information': f"{reagent_name} - Plateau"})
            last_reagent_plateau_time = max(last_reagent_plateau_time, plateau_time)

    push_df = df[is_push].copy()

    if push_df.empty:
        return df[~is_push].copy().reset_index(drop=True)

    push_df['time_diff'] = push_df['time'].diff().fillna(0)
    push_df['group'] = (push_df['time_diff'] > 150).cumsum()

    valid_bunches = [group for _, group in push_df.groupby('group') if len(group) >= 3]

    bunches_before = [b for b in valid_bunches if b['time'].max() < first_reagent_time]
    bunches_after = [b for b in valid_bunches if b['time'].min() > last_reagent_plateau_time]

    if bunches_before:

        bunch = max(bunches_before, key=len)
        last_idx = bunch.index[-1]

        new_records.append({'time': df.loc[last_idx, 'time'], 'information': 'Buffer 1'})

        if last_idx + 1 < len(df):
            new_records.append({'time': df.loc[last_idx + 1, 'time'], 'information': 'Buffer 1 - Plateau'})

    for i, bunch in enumerate(bunches_after[:2]):
        
        buffer_num = i + 2
        last_idx = bunch.index[-1]

        new_records.append({'time': df.loc[last_idx, 'time'], 'information': f'Buffer {buffer_num}'})

        if last_idx + 1 < len(df):
            new_records.append({'time': df.loc[last_idx + 1, 'time'], 'information': f'Buffer {buffer_num} - Plateau'})

    final_df = df[~is_push].copy()
    final_df = pd.concat([final_df, pd.DataFrame(new_records)], ignore_index=True)
    
    return final_df.sort_values(by='time').reset_index(drop=True)

def parse_stage_label(info):
    
    stage_raw = re.split(r'\s*-\s*Plateau', str(info))[0].strip()
    stage = stage_raw.lower()
    reagent_type = stage.split()[0] if stage.split() else ''
    is_buffer = reagent_type == 'buffer'

    chip_match = re.search(r'chip\s*(\d+)', stage)
    fc_match = re.search(r'fc\s*(\d+)', stage)
    all_numbers = re.findall(r'\d+', stage)

    if chip_match:
        chip_num = int(chip_match.group(1))
    elif fc_match:
        chip_num = int(fc_match.group(1))
    elif all_numbers:
        chip_num = int(all_numbers[-1])
    else:
        chip_num = None

    return stage_raw, stage, reagent_type, chip_num, is_buffer

def calculate_baseline_peaks(time, reagent_flags):
    
    baseline_records = []
    valid_flags = reagent_flags[reagent_flags['information'].str.contains('chip|fc|buffer', case=False, na=False) & ~reagent_flags['information'].str.contains('baseline', case=False, na=False)].copy()

    for i in range(len(valid_flags) - 1):

        row1 = valid_flags.iloc[i]
        row2 = valid_flags.iloc[i + 1]
        
        info1 = str(row1['information'])
        info2 = str(row2['information'])
        
        t_start = row1['time']
        t_end = row2['time']
        
        stage1_raw, stage1, type1, num1, is_buffer1 = parse_stage_label(info1)
        stage2_raw, stage2, type2, num2, is_buffer2 = parse_stage_label(info2)
        
        is_valid_transition = False
        
        if is_buffer1 != is_buffer2:
            is_valid_transition = True
        elif is_buffer1 and is_buffer2 and (stage1 != stage2):
            is_valid_transition = True
        elif not is_buffer1 and not is_buffer2 and (type1 != type2):
            
            is_dependent = (type1 in stage2) or (type2 in stage1)
            
            time_diff = t_end - t_start
            is_too_fast = time_diff < 10.0
            
            if is_dependent or is_too_fast:
                is_valid_transition = False
            else:
                if num1 is not None and num2 is not None:
                    is_valid_transition = num1 > num2
                else:
                    is_valid_transition = True
                    
        if is_valid_transition:
            
            time_diff = t_end - t_start
            
            start_after_gap = t_start + (time_diff * 0.1)
            target_time = start_after_gap + (t_end - start_after_gap) * 0.8
            
            closest_idx = np.argmin(np.abs(time - target_time))
            
            baseline_records.append({"time": time[closest_idx], "information": f"Baseline ({stage1_raw})"})
                    
    return baseline_records

def update_immob_flags(immob_df, immobilisation_dfs, flags_dfs, flags_dir='data/buckets/AMF_FLAGS'):
    """Calculates and writes Buffer, Peak, Baseline, and Flat flags."""

    os.makedirs(flags_dir, exist_ok=True)
    
    for _, immob_data in immob_df.iterrows():

        filename = immob_data['amfFlags']

        if filename not in flags_dfs: 
            continue
            
        file_path = os.path.join(flags_dir, filename)
        flag_data = flags_dfs[filename].copy()
        
        if flag_data['information'].str.contains('Buffer 1', na=False).any():
            continue

        print(f"Processing: {filename}")

        plot_data = immobilisation_dfs[immob_data['refPoly4']]
        time_array = plot_data['time'].sort_values().values
        signal = plot_data['channel1'].values
        
        start_time = time_array[0]
        finish_time = time_array.max()

        flag_data = flag_data[flag_data['time'] <= finish_time].copy()
        parsed_flags_df = extract_flags_from_pushes(flag_data)
        has_buffer_3_plateau = 'Buffer 3 - Plateau' in parsed_flags_df['information'].values

        time_delta = time_array[:-1]
        delta = np.diff(signal)
        initial_records = []
        final_records = []

        b1_flags = parsed_flags_df[parsed_flags_df['information'] == 'Buffer 1']

        if not b1_flags.empty:

            b1_time = b1_flags['time'].iloc[0]
            b1_idx = np.abs(time_array - b1_time).argmin()
            
            initial_offset = (time_array[b1_idx] - start_time) * 0.75
            initial_idx = np.where((time_delta > (start_time + initial_offset)) & (time_delta < time_array[b1_idx]))[0]

            if initial_idx.size > 0:

                flat_idx = initial_idx[np.argmin(np.abs(delta[initial_idx]))]
                initial_records.append({'time': time_array[flat_idx], 'information': 'Initial'})
        
        if has_buffer_3_plateau:

            b3p_flags = parsed_flags_df[parsed_flags_df['information'] == 'Buffer 3 - Plateau']

            if not b3p_flags.empty:

                b3p_time = b3p_flags['time'].iloc[0]
                b3p_idx = np.abs(time_array - b3p_time).argmin()
                
                final_offset = (finish_time - time_array[b3p_idx]) * 0.75
                final_idx = np.where((time_delta > (time_array[b3p_idx] + final_offset)) & (time_delta < finish_time))[0]

                if final_idx.size > 0:

                    flat_idx = final_idx[np.argmin(np.abs(delta[final_idx]))]
                    final_records.append({'time': time_array[flat_idx], 'information': 'Final'})
                else:
                    final_records.append({'time': time_array[max(0, len(time_array) - 40)], 'information': 'Final'})
                    
        boundary_list = [{"time": start_time, "information": "Start"}]

        if has_buffer_3_plateau:
            boundary_list.append({"time": finish_time, "information": "Finish"})
            
        boundary_records = pd.DataFrame(boundary_list)
        all_new_records = pd.concat([boundary_records, parsed_flags_df, pd.DataFrame(initial_records + final_records)], ignore_index=True)
        updated_df = all_new_records.sort_values('time').reset_index(drop=True)

        baseline_flags = updated_df[~updated_df['information'].str.contains('Concentration|Flat|Start|Finish', na=False)]
        baseline_list = calculate_baseline_peaks(time_array, baseline_flags)
        
        updated_df = pd.concat([updated_df, pd.DataFrame(baseline_list)], ignore_index=True)
        updated_df = updated_df.drop_duplicates(subset=["time", "information"], keep="first").sort_values("time").reset_index(drop=True)
        
        updated_df.to_csv(file_path, index=False)
        flags_dfs[filename] = updated_df.copy()
        
    return flags_dfs

def calculate_5s_averages(immob_df, immobilisation_dfs, flags_dfs, averaged_flags_dfs, avg_dir='data/buckets/averaged_flags'):
    """Calculates 5-second averages for flags dynamically across all channels."""

    os.makedirs(avg_dir, exist_ok=True)
    
    if averaged_flags_dfs is None: 
        averaged_flags_dfs = {}

    for _, immob_data in immob_df.iterrows():

        filename = immob_data['amfFlags']

        if filename in averaged_flags_dfs:
            continue
            
        file_path = os.path.join(avg_dir, filename)

        if filename in flags_dfs.keys():

            flag_data = flags_dfs[filename][~flags_dfs[filename]['information'].str.contains('Concentration', case=False, na=False)].reset_index(drop=True).copy()
        else:
            continue
        
        plot_data = immobilisation_dfs[immob_data['refPoly4']]
        channels = plot_data.columns[1:]
        sensor_time = plot_data['time'].values
        
        avg_data = {f'{ch}_avg': [] for ch in channels}
        window_starts = []
        window_ends = []
        window_labels = []

        for idx, row in flag_data.iterrows():

            t_flag = row['time']
            info = str(row['information']).lower()
            
            if 'start' in info:

                for ch in channels: 
                    avg_data[f'{ch}_avg'].append(plot_data[ch].values[0])

                window_starts.append(t_flag)
                window_ends.append(t_flag)
                window_labels.append("Start")
                continue

            elif 'finish' in info:

                for ch in channels: 
                    avg_data[f'{ch}_avg'].append(plot_data[ch].values[-1])
                
                window_starts.append(sensor_time[-1])
                window_ends.append(sensor_time[-1])
                window_labels.append("Finish")
                continue
                
            t_prev = flag_data.iloc[idx - 1]['time'] if idx > 0 else -np.inf
            t_next = flag_data.iloc[idx + 1]['time'] if idx < len(flag_data) - 1 else np.inf
            
            is_plateau_or_flat = any(w in info for w in ['plateau', 'flat'])
            is_baseline = 'baseline' in info

            t_start = None
            t_end = None
            w_label = ''
            
            if is_plateau_or_flat:

                if t_flag - 5 >= t_prev + 2:
                    t_start, t_end = t_flag - 5, t_flag
                    w_label = "Prev 5s"
                else:
                    t_start, t_end = t_prev + 2, t_flag
                    w_label = "Fallback"

            elif is_baseline:

                if t_flag + 5 <= t_next:
                    t_start, t_end = t_flag, t_flag + 5
                    w_label = "Next 5s"
            
                elif (t_flag - 2.5 >= t_prev + 2) and (t_flag + 2.5 <= t_next):
                    t_start, t_end = t_flag - 2.5, t_flag + 2.5
                    w_label = "Split 2.5s"

                elif t_flag - 5 >= t_prev + 2:
                    t_start, t_end = t_flag - 5, t_flag
                    w_label = "Prev 5s"

                else:
                    t_start, t_end = t_prev + 2, t_next - 0.1
                    w_label = "Gap Fallback"
            
            if t_start is not None and t_end > t_start:

                interp_grid = np.linspace(t_start, t_end, num=51)

                for ch in channels:
                    avg_data[f'{ch}_avg'].append(np.mean(np.interp(interp_grid, sensor_time, plot_data[ch].values)))

                window_starts.append(t_start)
                window_ends.append(t_end)
                window_labels.append(w_label)
            else:

                for ch in channels: 
                    avg_data[f'{ch}_avg'].append(np.nan)

                window_starts.append(np.nan)
                window_ends.append(np.nan)
                window_labels.append("")

        for col, data_list in avg_data.items(): 
            flag_data[col] = data_list

        flag_data['Window_Start'] = window_starts
        flag_data['Window_End'] = window_ends
        flag_data['Window_Label'] = window_labels
        
        flag_data = flag_data.dropna(subset=[f'{ch}_avg' for ch in channels]).reset_index(drop=True).drop('time', axis=1)
        flag_data.to_csv(file_path, index=False)
        averaged_flags_dfs[filename] = flag_data.copy()
        
    return averaged_flags_dfs

def collapse_combined_reagents(event_df):
    """Collapses consecutive identical stages (ignoring buffers) and normalizes names while preserving original case."""

    df = event_df.copy()

    df['stage'] = df['stage'].str.replace(r'(Baseline\s*\(\s*)(?!buffer)([\w/-]+)[^)]*(\))', r'\1\2\3', regex=True, flags=re.IGNORECASE)
    
    is_not_buffer = ~df['stage'].str.contains('buffer|baseline', case=False, na=False)
    
    extracted_stages = df['stage'].str.strip().str.extract(r'^([\w/-]+)', expand=False)
    
    df['base_stage'] = np.where(is_not_buffer, extracted_stages, df['stage'].str.strip())
    
    is_prg = df['stage'].str.contains('prg', case=False, na=False)
    is_chip1 = df['stage'].str.contains('chip 1', case=False, na=False)
    is_plateau = df['stage'].str.contains('plateau', case=False, na=False)
    
    mask_prg_chip1 = is_prg & is_chip1 & ~is_plateau
    
    mask_all_prg_plateaus = is_prg & is_plateau
    prg_plateau_indices = df[mask_all_prg_plateaus].index
    
    mask_last_prg_plateau = pd.Series(False, index=df.index)

    if not prg_plateau_indices.empty:
        mask_last_prg_plateau[prg_plateau_indices[-1]] = True
        
    mask_prg_inter = is_prg & ~mask_prg_chip1 & ~mask_last_prg_plateau
    
    df.loc[mask_prg_chip1, 'base_stage'] = 'PrG'
    df.loc[mask_last_prg_plateau, 'base_stage'] = 'PrG - Plateau'
    df.loc[mask_prg_inter, 'base_stage'] = 'PrG_Intermediate'
    
    df['compare_stage'] = df['base_stage'].str.lower()
    df['block_id'] = (df['compare_stage'] != df['compare_stage'].shift()).cumsum()
    
    collapsed = df.groupby('block_id').last().reset_index(drop=True)
    
    collapsed = collapsed[collapsed['base_stage'] != 'PrG_Intermediate'].reset_index(drop=True)
    
    counts = {}
    unique_stages = []

    for orig_name, comp_name in zip(collapsed['base_stage'], collapsed['compare_stage']):

        counts[comp_name] = counts.get(comp_name, 0) + 1
        unique_stages.append(orig_name if counts[comp_name] == 1 else f"{orig_name}_{counts[comp_name]}")
        
    collapsed['stage'] = unique_stages
    
    collapsed = collapsed.drop(columns=['base_stage', 'compare_stage'])
    
    return collapsed

def calculate_custom_immob_changes(event_df, channels):
    """Calculates standard stage-to-stage diffs and clean custom baseline metrics."""

    df = add_stage_changes(event_df.copy(), channels)
    
    stage_lower = event_df['stage'].str.lower()
    mask = (stage_lower.str.startswith('baseline', na=False) | stage_lower.str.startswith(('initial', 'final'), na=False))

    baselines = event_df[mask].copy()
    
    chip_id = event_df['chip_id'].iloc[0]
    
    custom_metrics = []
    
    if len(baselines) > 1:

        for i in range(1, len(baselines)):
            
            prev_b, curr_b = baselines.iloc[i-1], baselines.iloc[i]

            stage_name = f"Net_Shift_{prev_b['stage']}_to_{curr_b['stage']}"
            
            metric = {'stage': stage_name, 'time': curr_b['time'], 'chip_id': chip_id, 'base_stage': prev_b['stage']}

            metric.update({ch: curr_b[ch] for ch in channels})
            metric.update({f'{ch}_change': curr_b[ch] - prev_b[ch] for ch in channels})
            custom_metrics.append(metric)

        final_b = baselines.iloc[-1]
        initial_b = baselines.iloc[0]
        
        metric = {'stage': 'Total_Shift_Initial_to_Final', 'time': final_b['time'], 'chip_id': chip_id, 'base_stage': initial_b['stage']}

        metric.update({ch: final_b[ch] for ch in channels})
        metric.update({f'{ch}_change': final_b[ch] - initial_b[ch] for ch in channels})
        custom_metrics.append(metric)

    return pd.concat([df, pd.DataFrame(custom_metrics)], ignore_index=True)

def extract_intrastage_features(sens_df, event_df, channels, chip_id):
    """
    Slices raw sensorgram data between consecutive stage flags and extracts 
    kinetic and statistical intra-stage features for anomaly detection.
    """

    time_col = sens_df.columns[0]
    sens_time = sens_df[time_col].values
    features_list = []
    events = event_df.sort_values('time').reset_index(drop=True)
    
    for i in range(len(events) - 1):

        start_time = events.loc[i, 'time']
        end_time = events.loc[i + 1, 'time']
        
        stage_transition = f"{events.loc[i, 'stage']} -> {events.loc[i+1, 'stage']}"
        
        mask = (sens_time >= start_time) & (sens_time < end_time)
        stage_data = sens_df[mask]
        
        if stage_data.empty:
            continue
            
        stage_time = stage_data[time_col].values
        duration = stage_time[-1] - stage_time[0] if len(stage_time) > 1 else 0
        
        row_features = {
            'chip_id': chip_id,
            'stage': stage_transition,
            'duration': duration
        }
        
        for ch in channels:

            signal = stage_data[ch].values
            
            if len(signal) == 0:
                continue
                
            row_features[f"{ch}_min"] = np.min(signal)
            row_features[f"{ch}_max"] = np.max(signal)
            
            row_features[f"{ch}_std"] = np.std(signal)
            
            ch_mean = np.mean(signal)
            ch_median = np.median(signal)
            row_features[f"{ch}_stability"] = abs(ch_mean - ch_median)
            
            if len(signal) > 1:
                row_features[f"{ch}_auc"] = trapezoid(signal, stage_time)
            else:
                row_features[f"{ch}_auc"] = 0.0
                
            n_pts = max(2, int(len(signal) * 0.20))

            if len(signal) >= 2:
                
                slope, _, _, _, _ = linregress(stage_time[:n_pts], signal[:n_pts])
                row_features[f"{ch}_init_slope"] = slope
            else:
                row_features[f"{ch}_init_slope"] = 0.0
                
        features_list.append(row_features)
        
    return pd.DataFrame(features_list)

# Parameter Logistic Equations & Curve Fits
def five_pl(x, bottom, top, ec50, hill, asym):
    return top + (bottom - top) / (1 + (x / ec50)**(hill))**asym

def four_pl(x, bottom, top, ec50, hill):
    return top + (bottom - top) / (1 + (x / ec50)**(hill))

def fit_curve(data):

    x, y = data['x'], data['y']
    xfit = np.logspace(np.log10(min(x)), np.log10(max(x)), 1000)

    if data['fit_model'] == '5PL':

        bounds = ([-np.inf, -np.inf, min(x)/100, -10, 0.01], [np.inf, np.inf, max(x)*100, 10, 10])

        p0 = [min(y), max(y), np.median(x), 1.0, 1.0]
        popt, _ = curve_fit(five_pl, x, y, p0=p0, bounds=bounds, maxfev=50000)
        yfit = five_pl(xfit, *popt)

    else:

        bounds = ([-np.inf, -np.inf, min(x)/100, -10], [np.inf, np.inf, max(x)*100, 10])

        p0 = [min(y), max(y), np.median(x), 1.0]
        popt, _ = curve_fit(four_pl, x, y, p0=p0, bounds=bounds, maxfev=50000)
        yfit = four_pl(xfit, *popt)
        
    return xfit, yfit, popt

# UI Display Helpers
class collapsible_output:
    """
    PURE HTML CONTEXT MANAGER (Chronological + Solid Black Text Fix):
    Bypasses ipywidgets entirely, preserves order, and explicitly forces
    text to render in solid black instead of theme-inherited gray.
    """

    def __init__(self, title):

        self.title = title
        self.capture_ctx = capture_output()
        self.old_print = None

    def __enter__(self):

        self.old_print = builtins.print
        
        def custom_print(*args, **kwargs):

            sep = kwargs.get('sep', ' ')
            message = sep.join(str(arg) for arg in args)
            safe_message = message.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            html_snippet = f'<pre style="white-space: pre-wrap; font-family: monospace; margin: 0 0 10px 0; background: none; border: none; padding: 0; color: #000000; font-size: 13px;">{safe_message}</pre>'
            
            display({'text/html': html_snippet}, raw=True)

        builtins.print = custom_print
        self.captured_data = self.capture_ctx.__enter__()

        return self

    def __exit__(self, exc_type, exc_value, traceback):

        self.capture_ctx.__exit__(exc_type, exc_value, traceback)
        builtins.print = self.old_print
        
        if exc_type is not None:
            return False 
            
        html = f'''
        <details style="background-color: #f9f9f9; border: 1px solid #aaa; border-radius: 4px; margin-bottom: 10px;">
            <summary style="font-weight: bold; padding: 0.5em; cursor: pointer; font-family: sans-serif; color: #000000;">
                {self.title}
            </summary>
            <div style="padding: 10px; border-top: 1px solid #aaa; background-color: #fff; overflow-x: auto; color: #000000;">
        '''
        
        for out in self.captured_data.outputs:

            if 'text/html' in out.data:
                html += out.data['text/html'] + '<br>'
            elif 'image/png' in out.data:
                img_data = out.data['image/png']
                html += f'<img src="data:image/png;base64,{img_data}" style="max-width: 100%; height: auto;" /><br>'
            elif 'text/plain' in out.data:
                safe_text = out.data['text/plain'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                html += f'<pre style="white-space: pre-wrap; font-family: monospace; color: #000000;">{safe_text}</pre><br>'

        if self.captured_data.stderr:
            
            safe_stderr = self.captured_data.stderr.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            html += f'<pre style="white-space: pre-wrap; font-family: monospace; color: #a94442; background-color: #f2dede; padding: 10px; border-radius: 4px; margin-top: 10px;">{safe_stderr}</pre>'

        html += '</div></details>'
        
        display(HTML(html))

def generate_and_display_summary(data_list, val_cols, title, drop_na_col=None):

    if not data_list: 
        return None
    
    df = pd.concat(data_list, ignore_index=True)

    if drop_na_col: 
        df = df.dropna(subset=[drop_na_col]).copy()
        
    df['stage'] = pd.Categorical(df['stage'], categories=pd.unique(df['stage']), ordered=True)

    summary = build_stage_summary(df, 'stage', val_cols)
    
    with collapsible_output(title): 
        display(summary)
        
    return df

# Core Method Runners
def run_pel_analysis(pel_upload_df, sensorgram_dfs, flags_PEL_df):

    qc_output = collapsible_output("Quality Checks")

    with qc_output:
        check_data_quality(sensorgram_dfs, 'PEL')

    plot_output = collapsible_output("Sensorgram Plots")

    with plot_output:
        plot_sensorgrams(sensorgram_dfs, 'PEL')

    time_cols = pd.unique(flags_PEL_df.columns)[1:]
    all_events_PEL = []
    all_changes_PEL = []
    all_intrastage_PEL = []

    flag_output = collapsible_output("Sensorgram Flag Plots")

    with flag_output:

        for _, upload_data in pel_upload_df.iterrows():

            chip_id = upload_data['chip_id']

            if upload_data['sensorgram'] in sensorgram_dfs.keys():
                sens = sensorgram_dfs[upload_data['sensorgram']]
            else:
                continue

            flag_data = flags_PEL_df[flags_PEL_df['flags_id'] == upload_data['flags_id']].iloc[0]
            times = flag_data[time_cols].values
            labels = time_cols

            plot_flags_on_sensorgrams(sens, times, labels, title=f'PEL Data for {chip_id}')

            channels = sens.columns.tolist()[1:]
            event_dict = {
                'stage': labels,
                'time': times,
                'chip_id': chip_id
            }
            
            for ch in channels:
                event_dict[ch] = np.interp(times, sens.iloc[:,0], sens[ch])

            event_df = pd.DataFrame(event_dict)
            all_events_PEL.append(event_df)
            display(event_df)

            change_df = event_df.copy().drop(columns=['chip_id'])
            change_df = add_stage_changes(change_df, channels)

            initial_row = event_df[event_df['stage'] == 'Initial']
            final_row = event_df[event_df['stage'] == 'Final']

            summary_time = final_row['time'].iloc[0] if not final_row.empty else event_df['time'].iloc[-1]
            summary_row = {
                'stage': 'Total_Shift_Initial_to_Final',
                'time': summary_time
            }

            for ch in channels:

                val_initial = initial_row[ch].iloc[0] if not initial_row.empty else event_df[ch].iloc[0]
                val_final = final_row[ch].iloc[0] if not final_row.empty else event_df[ch].iloc[-1]
                summary_row[ch] = [val_initial, val_final]
                summary_row[f'{ch}_change'] = val_final - val_initial

            change_df = pd.concat([change_df, pd.DataFrame([summary_row])], ignore_index=True)
            change_df['chip_id'] = chip_id
            
            all_changes_PEL.append(change_df)
            display(change_df)

            intra_df = extract_intrastage_features(sens, event_df, channels, chip_id)
            all_intrastage_PEL.append(intra_df)

    all_events_PEL_df = pd.concat(all_events_PEL, ignore_index=True)
    all_events_PEL_df['stage'] = pd.Categorical(all_events_PEL_df['stage'], categories=time_cols, ordered=True)
    summary_PEL = build_stage_summary(all_events_PEL_df, 'stage', channels)

    metrics_output = collapsible_output("PEL Stage Metrics Summary")

    with metrics_output:
        display(summary_PEL)

    all_changes_PEL_df = pd.concat(all_changes_PEL, ignore_index=True)
    change_PEL_df = all_changes_PEL_df.dropna(subset=[f'{ch}_change' for ch in channels]).copy()

    stage_cols_pel = pd.unique(change_PEL_df['stage'])
    change_PEL_df['stage'] = pd.Categorical(change_PEL_df['stage'], categories=stage_cols_pel, ordered=True)

    change_summary_PEL = build_stage_summary(change_PEL_df, 'stage', [f'{ch}_change' for ch in channels])

    change_output = collapsible_output("PEL Stage Change Metrics Summary")

    with change_output:
        display(change_summary_PEL)

    for ch in channels:

        stats_output = collapsible_output(f"Statistical Plots ({ch})")

        with stats_output:
            plot_all_statistics(all_events_PEL_df, 'stage', ch, ylabel=f'{ch} Signal', title_prefix=f'Absolute - {ch}')
            plot_all_statistics(change_PEL_df, 'stage', f'{ch}_change', ylabel=f'Delta {ch} Signal', title_prefix=f'Delta - {ch}')

    return all_events_PEL_df, change_PEL_df, pd.concat(all_intrastage_PEL, ignore_index=True)

def run_immob_analysis(immob_df, immobilisation_dfs, flags_dfs, averaged_flags_dfs):

    qc_output = collapsible_output("Quality Checks")

    with qc_output:
        check_data_quality(immobilisation_dfs, 'Immobilisation')

    sensor_output = collapsible_output("Sensorgram Plots")

    with sensor_output:
        plot_sensorgrams(immobilisation_dfs, 'Immobilisation')

    calc_output = collapsible_output("Flag Calculations")

    with calc_output:

        print("Calculating and updating buffer, plateau, and baseline flags...")
        flags_dfs = update_immob_flags(immob_df, immobilisation_dfs, flags_dfs)

        print("Calculating 5-second averages...")

        averaged_flags_dfs = calculate_5s_averages(immob_df, immobilisation_dfs, flags_dfs, averaged_flags_dfs)

    all_events_imp, all_changes_imp, all_intra_imp = [], [], []
    all_events_comb, all_changes_comb, all_intra_comb = [], [], []
    all_events_avg_imp, all_changes_avg_imp, all_intra_avg_imp = [], [], []
    all_events_avg_comb, all_changes_avg_comb, all_intra_avg_comb = [], [], []

    for _, immob_data in immob_df.iterrows():

        chip_id = immob_data['chip_id']

        if immob_data['refPoly4'] in immobilisation_dfs.keys():
            sens = immobilisation_dfs[immob_data['refPoly4']]
        else:
            continue
        
        channels = sens.columns.tolist()[1:]
        
        if immob_data['amfFlags'] not in flags_dfs: 
            continue

        flag_data = flags_dfs[immob_data['amfFlags']]
        flag_data = flag_data[~flag_data.iloc[:, 1].str.contains('Concentration', case=False, na=False)]

        times, labels = flag_data['time'].tolist(), flag_data['information'].tolist()
        event_dict = {'stage': labels, 'time': times, 'chip_id': chip_id}

        for ch in channels: 
            event_dict[ch] = np.interp(times, sens.iloc[:, 0], sens[ch])
        
        event_df = pd.DataFrame(event_dict)
        all_events_imp.append(event_df)
        all_changes_imp.append(calculate_custom_immob_changes(event_df, channels))
        intra_imp_df = extract_intrastage_features(sens, event_df, channels, chip_id)
        all_intra_imp.append(intra_imp_df)

        comb_df = collapse_combined_reagents(event_df)
        all_events_comb.append(comb_df)
        all_changes_comb.append(calculate_custom_immob_changes(comb_df, channels))

        intra_comb_df = extract_intrastage_features(sens, comb_df, channels, chip_id)
        all_intra_comb.append(intra_comb_df)

        if averaged_flags_dfs and immob_data['amfFlags'] in averaged_flags_dfs:

            avg_flags = averaged_flags_dfs[immob_data['amfFlags']].copy()
            
            avg_dict = {
                'stage': avg_flags['information'].tolist(),
                'time': ((avg_flags['Window_Start'].astype(float) + avg_flags['Window_End'].astype(float)) / 2).tolist(),
                'chip_id': chip_id
            }

            for ch in channels: 
                avg_dict[ch] = avg_flags[f"{ch}_avg"].tolist()
                
            avg_df = pd.DataFrame(avg_dict)
            all_events_avg_imp.append(avg_df)
            all_changes_avg_imp.append(calculate_custom_immob_changes(avg_df, channels))

            intra_avg_imp_df = extract_intrastage_features(sens, avg_df, channels, chip_id)
            all_intra_avg_imp.append(intra_avg_imp_df)

            avg_comb_df = collapse_combined_reagents(avg_df)
            all_events_avg_comb.append(avg_comb_df)
            all_changes_avg_comb.append(calculate_custom_immob_changes(avg_comb_df, channels))

            intra_avg_comb_df = extract_intrastage_features(sens, avg_comb_df, channels, chip_id)
            all_intra_avg_comb.append(intra_avg_comb_df)

    flag_output = collapsible_output("Sensorgram Flag Plots")

    with flag_output:

        i = 0

        for _, immob_data in immob_df.iterrows():

            if immob_data['amfFlags'] not in flags_dfs.keys(): 
                continue

            sens = immobilisation_dfs[immob_data['refPoly4']]
            flag_data = flags_dfs[immob_data['amfFlags']]
            flag_data = flag_data[~flag_data.iloc[:, 1].str.contains('Concentration', case=False, na=False)]
            
            plot_flags_on_sensorgrams(sens, flag_data['time'].tolist(), flag_data['information'].tolist(), title=f'Immob Data: {immob_data["chip_id"]}', colours=['r' if 'Buffer' in str(lbl) else 'k' for lbl in flag_data['information']])

            display(all_events_imp[i])
            display(all_changes_imp[i])
            display(all_intra_imp[i])

            display(all_events_comb[i])
            display(all_changes_comb[i])
            display(all_intra_comb[i])
            i += 1

    if averaged_flags_dfs:

        avg_output = collapsible_output("Averaged Window Plots")

        with avg_output:

            i = 0

            for _, immob_data in immob_df.iterrows():

                if immob_data['amfFlags'] not in averaged_flags_dfs:
                    continue

                sens = immobilisation_dfs[immob_data['refPoly4']]
                avg_flags = averaged_flags_dfs[immob_data['amfFlags']]
                
                plot_5s_averaged_windows(sens, avg_flags, immob_data['chip_id'])

                display(all_events_avg_imp[i])
                display(all_changes_avg_imp[i])

                display(all_events_avg_comb[i])
                display(all_changes_avg_comb[i])
                i += 1

    change_cols = [f'{ch}_change' for ch in channels]

    imp_df = generate_and_display_summary(all_events_imp, channels, "Immediate: Metrics by Stage")
    imp_change_df = generate_and_display_summary(all_changes_imp, change_cols, "Immediate: Stage Changes", drop_na_col=change_cols[0])
    comb_df = generate_and_display_summary(all_events_comb, channels, "Combined Reagents: Metrics by Stage")
    comb_change_df = generate_and_display_summary(all_changes_comb, change_cols, "Combined Reagents: Stage Changes", drop_na_col=change_cols[0])

    imp_intra_df = pd.concat(all_intra_imp, ignore_index=True) if all_intra_imp else pd.DataFrame()
    comb_intra_df = pd.concat(all_intra_comb, ignore_index=True) if all_intra_comb else pd.DataFrame()

    avg_imp_intra_df = pd.DataFrame()
    avg_comb_intra_df = pd.DataFrame()

    if averaged_flags_dfs:

        avg_imp_df = generate_and_display_summary(all_events_avg_imp, channels, "5s Avg: Metrics by Stage")
        avg_imp_change_df = generate_and_display_summary(all_changes_avg_imp, change_cols, "5s Avg: Stage Changes", drop_na_col=change_cols[0])
        avg_comb_df = generate_and_display_summary(all_events_avg_comb, channels, "5s Avg Combined Reagents: Metrics by Stage")
        avg_comb_change_df = generate_and_display_summary(all_changes_avg_comb, change_cols, "5s Avg Combined Reagents: Stage Changes", drop_na_col=change_cols[0])

        avg_imp_intra_df = pd.concat(all_intra_avg_imp, ignore_index=True) if all_intra_avg_imp else pd.DataFrame()
        avg_comb_intra_df = pd.concat(all_intra_avg_comb, ignore_index=True) if all_intra_avg_comb else pd.DataFrame()
        
    for ch in channels:

        stats_output = collapsible_output(f"Statistical Plots ({ch})")

        with stats_output:

            plot_all_statistics(imp_df, 'stage', ch, ylabel=f'{ch} Signal (Immediate)', title_prefix=f'Immediate - {ch}')        
            plot_all_statistics(imp_change_df, 'stage', f'{ch}_change', ylabel=f'Delta {ch} (Immediate)', title_prefix=f'Immediate Delta - {ch}')        
            
            plot_all_statistics(comb_df, 'stage', ch, ylabel=f'{ch} Signal (Combined)', title_prefix=f'Combined - {ch}')        
            plot_all_statistics(comb_change_df, 'stage', f'{ch}_change', ylabel=f'Delta {ch} (Combined)', title_prefix=f'Combined Delta - {ch}')        
            
            if averaged_flags_dfs:
                plot_all_statistics(avg_imp_df, 'stage', ch, ylabel=f'{ch} Signal (5s Avg)', title_prefix=f'5s Avg - {ch}')
                plot_all_statistics(avg_imp_change_df, 'stage', f'{ch}_change', ylabel=f'Delta {ch} (5s Avg)', title_prefix=f'5s Avg Delta - {ch}')

                plot_all_statistics(avg_comb_df, 'stage', ch, ylabel=f'{ch} Signal (5s Avg Combined)', title_prefix=f'5s Avg Combined - {ch}')
                plot_all_statistics(avg_comb_change_df, 'stage', f'{ch}_change', ylabel=f'Delta {ch} (5s Avg Combined)', title_prefix=f'5s Avg Delta Combined - {ch}')

    return imp_df, imp_change_df, imp_intra_df, comb_df, comb_change_df, comb_intra_df, avg_imp_df, avg_imp_change_df, avg_imp_intra_df, avg_comb_df, avg_comb_change_df, avg_comb_intra_df

def run_standard_curve_analysis(standard_curves_df):

    qc_output = collapsible_output("Standard Curve Quality Checks")

    with qc_output:

        duplicate_count = standard_curves_df['standard_curve_uuid'].duplicated().sum()
        print('Duplicated standard curve UUIDs:', duplicate_count)

        def parse_delimited(s):

            sep = '|' if '|' in str(s) else ','
            return [float(v) for v in str(s).split(sep)]

        data_points_SC = {}

        for row in standard_curves_df.itertuples():

            x = parse_delimited(row.x_csv)
            y = parse_delimited(row.y_csv)

            if len(x) != len(y):
                print(f'Row {row.Index}: mismatched x/y lengths ({len(x)} vs {len(y)})')

            time_str = str(row.time).strip()

            if "-" in time_str and ":" not in time_str:
                time_str = time_str.replace("-", ":")

            dt = pd.to_datetime(f"{row.date} {time_str}", format="%Y-%m-%d %H:%M:%S", errors="coerce")

            data_points_SC[row.standard_curve_uuid] = {
                'x': x,
                'y': y,
                'chip_id': row.chip_id,
                'fit_model': row.fit_model,
                'a': row.a,
                'b': row.b,
                'c': row.c,
                'd': row.d,
                'e': row.e,
                'datetime': dt
            }

        clean_data_points_SC = {}
        print(f'Number of curves before removal: {len(data_points_SC)}')

        for k, v in data_points_SC.items():

            if pd.isnull(v['x']).any() or pd.isnull(v['y']).any():

                print(f"Removing curve: {k} due to NaN values.")
                print(f"x:\n{v['x']}\ny:\n{v['y']}\n")
            else:
                clean_data_points_SC[k] = v
                
        print(f'Remaining valid curves: {len(clean_data_points_SC)}')

    stored_plot_output = collapsible_output("Standard Curve Plots (Stored Parameters)")

    with stored_plot_output:

        for key, data in clean_data_points_SC.items():

            x_pts = np.logspace(np.log10(min(data['x'])), np.log10(max(data['x'])), 1000)

            if data['fit_model'] == '5PL':
                y_pts = five_pl(x_pts, data['a'], data['d'], data['c'], data['b'], data['e']) 
            else:
                y_pts = four_pl(x_pts, data['a'], data['d'], data['c'], data['b'])

            plt.figure(figsize=(10, 6))
            plt.scatter(data['x'], data['y'], label='Data', color='blue')
            plt.plot(x_pts, y_pts, color='red', label='Stored Fit')

            plt.xscale('log')
            plt.title(f'Standard Curve {key} (Chip: {data["chip_id"]}) - Stored Values')
            plt.xlabel('Concentration')
            plt.ylabel('Signal')
            plt.legend()
            plt.show()

    calc_plot_output = collapsible_output("Standard Curve Plots (Calculated Parameters)")
    unique_chips_SC = list({v['chip_id'] for v in clean_data_points_SC.values()})
    
    with calc_plot_output:

        print(f"Total Unique Chips: {len(unique_chips_SC)}")

        for chip_id in unique_chips_SC:

            for key, data in clean_data_points_SC.items():

                if data['chip_id'] != chip_id:
                    continue

                xfit, yfit, _ = fit_curve(data)

                plt.figure(figsize=(10, 6))
                plt.scatter(data['x'], data['y'], label='Data', color='blue')
                plt.plot(xfit, yfit, color='green', label='Calculated Fit')

                plt.xscale('log')
                plt.title(f'Standard Curve {key} (Chip: {chip_id}) - Calculated Fit')
                plt.xlabel('Concentration')
                plt.ylabel('Signal')
                plt.legend()
                plt.show()

    sc_collected_data = []

    for key, data in clean_data_points_SC.items():

        xfit, yfit, popt = fit_curve(data)

        if data['fit_model'] == '5PL':
            y_pred = five_pl(np.asarray(data['x']), *popt)
        else:
            y_pred = four_pl(np.asarray(data['x']), *popt)

        ss_res = np.sum((np.asarray(data['y']) - y_pred) ** 2)
        ss_tot = np.sum((np.asarray(data['y']) - np.mean(data['y'])) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        sc_collected_data.append({
            'standard_curve_uuid': key,
            'fit_model': data['fit_model'],
            'parameters': popt,
            'r2': r2,
            'points': list(zip(data['x'], data['y'])),
            'chip_id': data['chip_id'],
            'datetime': data['datetime'],
            'a': data['a'],
            'b': data['b'],
            'c': data['c'],
            'd': data['d'],
            'e': data['e']
        })

    sc_collected_df = pd.DataFrame(sc_collected_data)

    return clean_data_points_SC, sc_collected_df
