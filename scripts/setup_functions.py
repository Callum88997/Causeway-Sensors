# Imports required packages
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from IPython.display import display, HTML
import ipywidgets as widgets
from scipy.stats import skew, kurtosis

# Data Quality and Visualisation
def check_data_quality(dfs_dict, stage_name):
    '''Prints missing values, duplicated rows, and summary statistics for generic sensorgrams.

    Args:
        dfs_dict (dict): Dictionary of sensorgram dataframes.
        stage_name (str): The name of the experimental stage.
    '''

    # Loops through each key-value pair in the dictionary
    for name, df in dfs_dict.items():
        
        # Display missing values and duplicated rows
        print(f'Metrics for {stage_name} - {name}:\n')
        print(f'\tMissing values: {df.isnull().sum().sum()}')
        print(f'\tDuplicated rows: {df.duplicated().sum()}\n')

        # Declares channel variables
        channels = df.columns.tolist()[1:]

        # Loops through each chip channel
        for chip in channels:
            print(f'\t{chip.capitalize()} Metrics:')
            
            # Display summary statistics for each chip channel
            display(df[chip].describe())

            # Display first 5 rows
            display(df[chip].head())

            # Display last 5 rows
            display(df[chip].tail())

def plot_sensorgrams(dfs_dict, stage_name):
    '''Plots raw sensorgrams, dynamically finding time and channel columns.

    Args:
        dfs_dict (dict): Dictionary of sensorgram dataframes.
        stage_name (str): The name of the experimental stage.
    '''

    # Loops through each key-value pair in the dictionary
    for name, df in dfs_dict.items():

        # Declares time and channel variables
        time_col = df.columns[0]
        channels = df.columns.tolist()[1:]

        # Plots raw channel response against time for each sensorgram
        plt.figure(figsize=(10, 6))
        
        # Loops through each channel
        for ch in channels:
            plt.plot(df[time_col], df[ch], label=ch)

        plt.title(f'{stage_name} Data for {name}')
        plt.xlabel('Time (s)')
        plt.ylabel('RU')
        plt.legend()
        plt.show()

def plot_flags_on_sensorgrams(df_sens, times, labels, title, colours=None):
    '''Overlays vertical flag lines and text on a generic sensorgram.

    Args:
        df_sens (pd.DataFrame): Sensorgram data.
        times (list[float]): List of time points for the flags.
        labels (list[str]): List of labels corresponding to the time points.
        title (str): Title of the plot.
        colours (list[str], optional): List of colours for the flags.
    '''

    # Declares time, channel, and colour variables
    time_col = df_sens.columns[0]
    channels = df_sens.columns.tolist()[1:]
    colours = colours or ['k'] * len(times)

    # Initialises the plot figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Loops through each time, label, colour, and index tuple
    for i, (t, label, c) in enumerate(zip(times, labels, colours)):
        
        ax.axvline(x=t, linestyle='--', color=c)

        if 'plateau' in label.lower():
            ax.text(t, 1.1, label, rotation=90, transform=ax.get_xaxis_transform(), verticalalignment='bottom', horizontalalignment='center', color=c)
        elif 'baseline' in label.lower():
            ax.text(t, 0.5, label, rotation=45, transform=ax.get_xaxis_transform(), verticalalignment='center', horizontalalignment='center', color=c)
        else:
            ax.text(t, -0.1, label, rotation=90, transform=ax.get_xaxis_transform(), verticalalignment='top', horizontalalignment='center', color=c)

    # Loops through each channel
    for ch in channels:
        ax.plot(df_sens[time_col], df_sens[ch], label=ch)

    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('RU')
    ax.legend()
    plt.show()

def plot_5s_averaged_windows(sens, avg_flags, title_id):
    '''Plots sensorgram responses with five-second averaging windows highlighted, specific to immobilisation data.

    Args:
        sens (pd.DataFrame): Raw sensorgram data.
        avg_flags (pd.DataFrame): Processed flags containing window boundaries.
        title_id (str): Identifier to include in the plot title.
    '''

    # Declares time and channel variables
    time_col = sens.columns[0]
    channels = sens.columns.tolist()[1:]
    
    # Plots sensorgram responses with the five-second averaging windows highlighted
    plt.figure(figsize=(10, 6))

    # Loops through each channel
    for ch in channels:
        plt.plot(sens[time_col], sens[ch], label=ch)
        
    # Loops through each row in the averaged flags dataframe
    for _, row in avg_flags.iterrows():

        w_start, w_end = row['Window_Start'], row['Window_End']
        plt.axvspan(w_start, w_end, color='black', alpha=0.2, linestyle='--')
        
        t_flag = (float(w_start) + float(w_end)) / 2
        avg_val = row.get(f'{channels[0]}_avg', 0)
        plt.text(t_flag, avg_val, f" {row['information']} ({row['Window_Label']})", fontsize=8, verticalalignment='bottom', rotation=15, color='darkred')
                  
    plt.title(f'Averaging Windows: {title_id}')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def plot_all_statistics(df, x_col, y_col, ylabel='Signal', title_prefix=''):
    '''Generates box, violin, density, correlation, and heatmap plots sequentially for a sensorgram stage metrics or changes dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing the statistics to plot.
        x_col (str): Column name for the x-axis.
        y_col (str): Column name for the y-axis.
        ylabel (str): Label for the y-axis.
        title_prefix (str): Prefix to prepend to plot titles.
    '''
    
    # Declares title prefix variable
    title_prefix = f'{title_prefix} - ' if title_prefix else ''
    
    # Plots the spread and density of signal values at each experimental stage using box and strip plots
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=x_col, y=y_col)
    sns.stripplot(data=df, x=x_col, y=y_col, color='black', alpha=0.5, jitter=False)
    plt.xticks(rotation=90)
    plt.ylabel(ylabel)
    plt.ylim(-10, 15)
    plt.title(f'{title_prefix}Stage Boxplot & Stripplot')
    plt.tight_layout()
    plt.show()
    plt.close()

    # Plots the distribution of data across different stages using violin plots
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x=x_col, y=y_col, palette='muted')
    plt.xticks(rotation=45)
    plt.ylabel(ylabel)
    plt.title(f'{title_prefix}Stage Violin Plot')
    plt.tight_layout()
    plt.show()
    plt.close()

    # Plots the density profiles for signal values categorised by stage
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x=y_col, hue=x_col, fill=True, common_norm=False, alpha=0.3)
    plt.xlabel(ylabel)
    plt.title(f'{title_prefix}Signal Density Profiles Per Stage')
    plt.tight_layout()
    plt.show()
    plt.close()

    # Plots the correlation matrix to show how numeric measurements vary together
    plt.figure(figsize=(10, 6))
    corr_matrix = df.corr(numeric_only=True)

    # Checks whether the correlation matrix contains data
    if not corr_matrix.empty:

        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
        plt.title(f'{title_prefix}Channel & Time Correlation Matrix')
        plt.tight_layout()
        plt.show()

    plt.close()

    # Checks if chip_id is present to plot overview heatmap
    if 'chip_id' in df.columns:

        # Plots mean signal values for every chip and experimental stage as a heatmap
        plt.figure(figsize=(10, 6))

        pivot_df = df.pivot_table(index='chip_id', columns=x_col, values=y_col, aggfunc='mean')
        
        sns.heatmap(pivot_df, annot=False, cmap='viridis', fmt='.2f', cbar_kws={'label': ylabel})
        plt.xticks(rotation=45)
        plt.ylabel('Chip ID')
        plt.xlabel('Experimental Stage')
        plt.title(f'{title_prefix}Overview Heatmap (Mean Value per Chip per Stage)')
        plt.tight_layout()
        plt.show()
        plt.close()

# Data Processing and Feature Extraction
def build_stage_summary(df, group_col, val_cols):
    '''Builds statistical summaries mapped over the given channel columns.

    Args:
        df (pd.DataFrame): Dataframe containing stage measurements.
        group_col (str): Column to group the summary by.
        val_cols (list[str]): List of value columns to aggregate.

    Returns:
        pd.DataFrame: Aggregated statistical summary dataframe.
    '''

    # Calculates aggregate statistics grouped by the specified column
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

    # Loops through each value column to calculate coefficient of variation
    for col in val_cols:
        summary[f'{col}_cv_pct'] = 100 * summary[f'{col}_std'] / summary[f'{col}_mean'].abs()

    # Declares variables for column ordering
    ordered_cols = [group_col, 'n']
    stat_suffixes = ['mean', 'median', 'std', 'min', 'max', 'iqr', 'cv_pct']
    
    # Loops through each value column to sequence the statistics
    for col in val_cols:

        # Loops through each statistic suffix
        for stat in stat_suffixes:

            col_name = f'{col}_{stat}'

            if col_name in summary.columns:
                ordered_cols.append(col_name)
                
    # Loops through each column in the summary dataframe to capture remaining fields
    for col in summary.columns:

        if col not in ordered_cols:
            ordered_cols.append(col)

    return summary[ordered_cols]

def add_stage_changes(event_table, channels):
    '''Calculates differences dynamically for provided channel columns and labels stage transitions.

    Args:
        event_table (pd.DataFrame): Dataframe containing event data.
        channels (list[str]): List of channel columns to process.

    Returns:
        pd.DataFrame: Dataframe with calculated stage changes.
    '''

    # Loops through each channel to calculate the difference
    for ch in channels:
        event_table[f'{ch}_change'] = event_table[ch].diff()

    # Shifts the stage column to create transition labels
    event_table['stage'] = event_table['stage'].shift(1) + ' -> ' + event_table['stage']

    # Removes the first row which contains NaN values from the shift operation
    event_table = event_table.iloc[1:].reset_index(drop=True)

    return event_table

def extract_flags_from_pushes(flag_df):
    '''Parses a flag dataframe to extract buffers, reagents, and their plateaus based on the syringe push event sequences.

    Args:
        flag_df (pd.DataFrame): Raw flag dataframe.

    Returns:
        pd.DataFrame: Parsed and ordered flag dataframe.
    '''

    # Creates a copy of the flag dataframe to preserve the original data
    df = flag_df.copy().sort_values(by='time').reset_index(drop=True)
    
    # Identifies rows corresponding to syringe pushes
    is_push = df['information'].str.contains("No 'Concentration'", na=False)
    new_records = []

    # Identifies indices for reagent events
    reagent_indices = df.index[~is_push].tolist()
    
    # Checks whether reagent indices exist
    if not reagent_indices:
        first_reagent_time = float('inf')
        last_reagent_plateau_time = float('inf')
    else:
        first_reagent_time = df.loc[reagent_indices[0], 'time']
        last_reagent_plateau_time = df.loc[reagent_indices[-1], 'time']

    # Loops through each reagent index
    for idx in reagent_indices:

        reagent_name = df.loc[idx, 'information']
        
        if idx + 1 < len(df):

            plateau_time = df.loc[idx + 1, 'time']
            new_records.append({'time': plateau_time, 'information': f'{reagent_name} - Plateau'})
            last_reagent_plateau_time = max(last_reagent_plateau_time, plateau_time)

    # Creates a copy of the push dataframe to preserve the original data
    push_df = df[is_push].copy()

    # Checks whether the push dataframe contains data
    if push_df.empty:
        return df[~is_push].copy().reset_index(drop=True)

    # Calculates time differences and groups pushes
    push_df['time_diff'] = push_df['time'].diff().fillna(0)
    push_df['group'] = (push_df['time_diff'] > 150).cumsum()

    # Filters valid bunches containing at least 3 events
    valid_bunches = [group for _, group in push_df.groupby('group') if len(group) >= 3]

    bunches_before = [b for b in valid_bunches if b['time'].max() < first_reagent_time]
    bunches_after = [b for b in valid_bunches if b['time'].min() > last_reagent_plateau_time]

    # Evaluates and adds buffer records before the first reagent
    if bunches_before:

        bunch = max(bunches_before, key=len)
        last_idx = bunch.index[-1]

        new_records.append({'time': df.loc[last_idx, 'time'], 'information': 'Buffer 1 - NHS EDC'})

        if last_idx + 1 < len(df):
            new_records.append({'time': df.loc[last_idx + 1, 'time'], 'information': 'Buffer 1 - Plateau'})

    buffer_suffixes = {2: 'ETA', 3: 'Casein Block'}

    # Loops through each enumerated bunch after the reagents
    for i, bunch in enumerate(bunches_after[:2]):
        
        buffer_num = i + 2
        last_idx = bunch.index[-1]

        suffix = buffer_suffixes.get(buffer_num, '')
        buffer_name = f'Buffer {buffer_num} - {suffix}' if suffix else f'Buffer {buffer_num}'

        new_records.append({'time': df.loc[last_idx, 'time'], 'information': buffer_name})
        
        if last_idx + 1 < len(df):
            new_records.append({'time': df.loc[last_idx + 1, 'time'], 'information': f'Buffer {buffer_num} - Plateau'})

    # Creates a copy of the final dataframe to preserve the original data
    final_df = df[~is_push].copy()
    final_df = pd.concat([final_df, pd.DataFrame(new_records)], ignore_index=True)
    
    return final_df.sort_values(by='time').reset_index(drop=True)

def parse_stage_label(info):
    '''Extracts a concise stage label from an immobilisation flag description.

    Args:
        info (str): Raw descriptive text from a flag record.

    Returns:
        tuple: Normalised stage label variables suitable for analysis output.
    '''
    
    # Splits the string to extract the stage and reagent type
    stage_raw = re.split(r'\s*-\s*Plateau', str(info))[0].strip()

    if stage_raw.lower().startswith('buffer') and '-' in stage_raw:
        stage_raw = stage_raw.split('-')[0].strip()

    stage = stage_raw.lower()
    reagent_type = stage.split()[0] if stage.split() else ''
    is_buffer = reagent_type == 'buffer'

    # Searches for chip or fc numbers using regular expressions
    chip_match = re.search(r'chip\s*(\d+)', stage)
    fc_match = re.search(r'fc\s*(\d+)', stage)
    all_numbers = re.findall(r'\d+', stage)

    # Evaluates the extracted matches to determine the chip number
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
    '''Calculates baseline and peak sampling positions from reagent flag timings.

    Args:
        time (pd.Series): Sensorgram time values.
        reagent_flags (pd.DataFrame): Flag records defining reagent transitions.

    Returns:
        list[dict]: Baseline and peak sampling definitions for each stage.
    '''
    
    baseline_records = []
    
    # Creates a copy of the valid flags to preserve the original data
    valid_flags = reagent_flags[reagent_flags['information'].str.contains('chip|fc|buffer', case=False, na=False) & ~reagent_flags['information'].str.contains('baseline', case=False, na=False)].copy()

    # Loops through each valid flag index
    for i in range(len(valid_flags) - 1):

        row1 = valid_flags.iloc[i]
        row2 = valid_flags.iloc[i + 1]
        
        info1 = str(row1['information'])
        info2 = str(row2['information'])
        
        t_start = row1['time']
        t_end = row2['time']
        
        # Parses the stage labels for both rows
        stage1_raw, stage1, type1, num1, is_buffer1 = parse_stage_label(info1)
        stage2_raw, stage2, type2, num2, is_buffer2 = parse_stage_label(info2)
        
        is_valid_transition = False
        
        # Determines if the transition between stages is valid
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
                    
        # Calculates the target time for baseline sampling
        if is_valid_transition:
            
            time_diff = t_end - t_start
            
            start_after_gap = t_start + (time_diff * 0.1)
            target_time = start_after_gap + (t_end - start_after_gap) * 0.8
            
            closest_idx = np.argmin(np.abs(time - target_time))
            
            baseline_records.append({'time': time[closest_idx], 'information': f'Baseline - {stage1_raw}'})
                    
    return baseline_records

def update_immob_flags(immob_df, immobilisation_dfs, flags_dfs, flags_dir='data/buckets/AMF_FLAGS'):
    '''Calculates and writes buffer, peak, baseline, and flat flags.

    Args:
        immob_df (pd.DataFrame): Immobilisation metadata dataframe.
        immobilisation_dfs (dict): Dictionary of sensorgram dataframes.
        flags_dfs (dict): Dictionary of flag dataframes.
        flags_dir (str): Output directory for updated flag files.

    Returns:
        dict: Updated dictionary of flag dataframes.
    '''

    os.makedirs(flags_dir, exist_ok=True)
    
    # Loops through each row in the immobilisation dataframe
    for _, immob_data in immob_df.iterrows():

        filename = immob_data['amfFlags']

        # Checks whether the file exists in the flag dataframes
        if filename not in flags_dfs: 
            continue
            
        file_path = os.path.join(flags_dir, filename)
        
        # Creates a copy of the flag data to preserve the original data
        flag_data = flags_dfs[filename].copy()
        
        if flag_data['information'].str.contains('Buffer 1', na=False).any():
            continue

        # Display processing output
        print(f'Processing: {filename}')

        plot_data = immobilisation_dfs[immob_data['refPoly4']]
        time_array = plot_data['time'].sort_values().values
        signal = plot_data['channel1'].values
        
        start_time = time_array[0]
        finish_time = time_array.max()

        # Creates a copy of the filtered flag data to preserve the original data
        flag_data = flag_data[flag_data['time'] <= finish_time].copy()
        parsed_flags_df = extract_flags_from_pushes(flag_data)
        has_buffer_3_plateau = 'Buffer 3 - Plateau' in parsed_flags_df['information'].values

        time_delta = time_array[:-1]
        delta = np.diff(signal)
        initial_records = []
        final_records = []

        is_buffer_1 = parsed_flags_df['information'].str.startswith('Buffer 1', na=False)
        not_plateau = ~parsed_flags_df['information'].str.endswith('Plateau', na=False)

        b1_flags = parsed_flags_df[is_buffer_1 & not_plateau]
        
        # Checks whether the dataframe contains data
        if not b1_flags.empty:

            b1_time = b1_flags['time'].iloc[0]
            b1_idx = np.abs(time_array - b1_time).argmin()
            
            initial_offset = (time_array[b1_idx] - start_time) * 0.75
            initial_idx = np.where((time_delta > (start_time + initial_offset)) & (time_delta < time_array[b1_idx]))[0]

            # Finds the flattest region in the initial window
            if initial_idx.size > 0:

                flat_idx = initial_idx[np.argmin(np.abs(delta[initial_idx]))]
                initial_records.append({'time': time_array[flat_idx], 'information': 'Initial'})
        
        if has_buffer_3_plateau:

            b3p_flags = parsed_flags_df[parsed_flags_df['information'] == 'Buffer 3 - Plateau']

            # Checks whether the dataframe contains data
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
                    
        boundary_list = [{'time': start_time, 'information': 'Start'}]

        if has_buffer_3_plateau:
            boundary_list.append({'time': finish_time, 'information': 'Finish'})
            
        boundary_records = pd.DataFrame(boundary_list)
        all_new_records = pd.concat([boundary_records, parsed_flags_df, pd.DataFrame(initial_records + final_records)], ignore_index=True)
        updated_df = all_new_records.sort_values('time').reset_index(drop=True)

        baseline_flags = updated_df[~updated_df['information'].str.contains('Concentration|Flat|Start|Finish', na=False)]
        baseline_list = calculate_baseline_peaks(time_array, baseline_flags)
        
        updated_df = pd.concat([updated_df, pd.DataFrame(baseline_list)], ignore_index=True)
        updated_df = updated_df.drop_duplicates(subset=['time', 'information'], keep='first').sort_values('time').reset_index(drop=True)
        
        updated_df.to_csv(file_path, index=False)
        
        # Creates a copy of the flags dataframe filename to preserve the original data
        flags_dfs[filename] = updated_df.copy()
        
    return flags_dfs

def calculate_5s_averages(immob_df, immobilisation_dfs, flags_dfs, averaged_flags_dfs, avg_dir='data/buckets/averaged_flags'):
    '''Calculates 5-second averages for flags dynamically across all channels.

    Args:
        immob_df (pd.DataFrame): Immobilisation metadata dataframe.
        immobilisation_dfs (dict): Dictionary of sensorgram dataframes.
        flags_dfs (dict): Dictionary of flag dataframes.
        averaged_flags_dfs (dict): Dictionary of averaged flag dataframes.
        avg_dir (str): Output directory for the averaged flag files.

    Returns:
        dict: Updated dictionary of averaged flag dataframes.
    '''

    os.makedirs(avg_dir, exist_ok=True)
    
    # Checks whether the averaged flags dictionary is initialised
    if averaged_flags_dfs is None: 
        averaged_flags_dfs = {}

    # Loops through each row in the immobilisation dataframe
    for _, immob_data in immob_df.iterrows():

        filename = immob_data['amfFlags']

        if filename in averaged_flags_dfs:
            continue
            
        file_path = os.path.join(avg_dir, filename)

        if filename in flags_dfs.keys():

            # Creates a copy of the flag data to preserve the original data
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

        # Loops through each row in the flag dataframe
        for idx, row in flag_data.iterrows():

            t_flag = row['time']
            info = str(row['information']).lower()
            
            # Evaluates start flags
            if 'start' in info:

                # Loops through each channel
                for ch in channels: 
                    avg_data[f'{ch}_avg'].append(plot_data[ch].values[0])

                window_starts.append(t_flag)
                window_ends.append(t_flag)
                window_labels.append('Start')
                continue

            # Evaluates finish flags
            elif 'finish' in info:

                # Loops through each channel
                for ch in channels: 
                    avg_data[f'{ch}_avg'].append(plot_data[ch].values[-1])
                
                window_starts.append(sensor_time[-1])
                window_ends.append(sensor_time[-1])
                window_labels.append('Finish')
                continue
                
            t_prev = flag_data.iloc[idx - 1]['time'] if idx > 0 else -np.inf
            t_next = flag_data.iloc[idx + 1]['time'] if idx < len(flag_data) - 1 else np.inf
            
            # Determines flag type for window calculation
            is_plateau_or_flat = any(w in info for w in ['plateau', 'flat'])
            is_baseline = 'baseline' in info

            t_start = None
            t_end = None
            w_label = ''
            
            # Calculates start and end times based on flag type
            if is_plateau_or_flat:

                if t_flag - 5 >= t_prev + 2:
                    t_start, t_end = t_flag - 5, t_flag
                    w_label = 'Prev 5s'
                else:
                    t_start, t_end = t_prev + 2, t_flag
                    w_label = 'Fallback'

            elif is_baseline:

                if t_flag + 5 <= t_next:
                    t_start, t_end = t_flag, t_flag + 5
                    w_label = 'Next 5s'
            
                elif (t_flag - 2.5 >= t_prev + 2) and (t_flag + 2.5 <= t_next):
                    t_start, t_end = t_flag - 2.5, t_flag + 2.5
                    w_label = 'Split 2.5s'

                elif t_flag - 5 >= t_prev + 2:
                    t_start, t_end = t_flag - 5, t_flag
                    w_label = 'Prev 5s'

                else:
                    t_start, t_end = t_prev + 2, t_next - 0.1
                    w_label = 'Gap Fallback'
            
            # Interpolates signal across the defined window to calculate the average
            if t_start is not None and t_end > t_start:

                interp_grid = np.linspace(t_start, t_end, num=51)

                # Loops through each channel
                for ch in channels:
                    avg_data[f'{ch}_avg'].append(np.mean(np.interp(interp_grid, sensor_time, plot_data[ch].values)))

                window_starts.append(t_start)
                window_ends.append(t_end)
                window_labels.append(w_label)
            else:

                # Loops through each channel
                for ch in channels: 
                    avg_data[f'{ch}_avg'].append(np.nan)

                window_starts.append(np.nan)
                window_ends.append(np.nan)
                window_labels.append('')

        # Loops through each key-value pair in the dictionary
        for col, data_list in avg_data.items(): 
            flag_data[col] = data_list

        flag_data['Window_Start'] = window_starts
        flag_data['Window_End'] = window_ends
        flag_data['Window_Label'] = window_labels
        
        flag_data = flag_data.dropna(subset=[f'{ch}_avg' for ch in channels]).reset_index(drop=True).drop('time', axis=1)
        flag_data.to_csv(file_path, index=False)
        
        # Creates a copy of the averaged flags dataframe filename to preserve the original data
        averaged_flags_dfs[filename] = flag_data.copy()
        
    return averaged_flags_dfs

def collapse_combined_reagents(event_df):
    '''Collapses consecutive identical stages (ignoring buffers) and normalises names while preserving original case.

    Args:
        event_df (pd.DataFrame): Dataframe containing event stages.

    Returns:
        pd.DataFrame: Collapsed event dataframe.
    '''

    # Creates a copy of the event dataframe to preserve the original data
    df = event_df.copy()

    # Removes trailing bracket information excluding buffers
    df['stage'] = df['stage'].str.replace(r'(Baseline\s*\(\s*)(?!buffer)([\w/-]+)[^)]*(\))', r'\1\2\3', regex=True, flags=re.IGNORECASE)
    
    is_not_buffer = ~df['stage'].str.contains('buffer|baseline', case=False, na=False)
    
    # Extracts the base stage name
    extracted_stages = df['stage'].str.strip().str.extract(r'^([\w/-]+)', expand=False)
    
    df['base_stage'] = np.where(is_not_buffer, extracted_stages, df['stage'].str.strip())
    
    # Identifies Protein G (PrG) related stages
    is_prg = df['stage'].str.contains('prg', case=False, na=False)
    is_chip1 = df['stage'].str.contains('chip 1', case=False, na=False)
    is_plateau = df['stage'].str.contains('plateau', case=False, na=False)
    
    mask_prg_chip1 = is_prg & is_chip1 & ~is_plateau
    
    mask_all_prg_plateaus = is_prg & is_plateau
    prg_plateau_indices = df[mask_all_prg_plateaus].index
    
    mask_last_prg_plateau = pd.Series(False, index=df.index)

    # Checks whether the dataframe contains data
    if not prg_plateau_indices.empty:
        mask_last_prg_plateau[prg_plateau_indices[-1]] = True
        
    mask_prg_inter = is_prg & ~mask_prg_chip1 & ~mask_last_prg_plateau
    
    # Assigns unified base stage names for PrG transitions
    df.loc[mask_prg_chip1, 'base_stage'] = 'PrG'
    df.loc[mask_last_prg_plateau, 'base_stage'] = 'PrG - Plateau'
    df.loc[mask_prg_inter, 'base_stage'] = 'PrG_Intermediate'
    
    df['compare_stage'] = df['base_stage'].str.lower()

    # Calculates sequential block IDs to group consecutive identical stages
    df['block_id'] = (df['compare_stage'] != df['compare_stage'].shift()).cumsum()
    
    # Collapses consecutive stages and drops intermediate PrG steps
    collapsed = df.groupby('block_id').last().reset_index(drop=True)
    collapsed = collapsed[collapsed['base_stage'] != 'PrG_Intermediate'].reset_index(drop=True)
    
    counts = {}
    unique_stages = []

    # Loops through each original name and comparison name tuple
    for orig_name, comp_name in zip(collapsed['base_stage'], collapsed['compare_stage']):

        counts[comp_name] = counts.get(comp_name, 0) + 1
        unique_stages.append(orig_name if counts[comp_name] == 1 else f'{orig_name}_{counts[comp_name]}')
        
    collapsed['stage'] = unique_stages
    
    collapsed = collapsed.drop(columns=['base_stage', 'compare_stage'])
    
    return collapsed

def calculate_custom_immob_changes(event_df, channels):
    '''Calculates standard stage-to-stage differences and clean custom baseline metrics.

    Args:
        event_df (pd.DataFrame): Dataframe containing event changes.
        channels (list[str]): List of channels to calculate differences for.

    Returns:
        pd.DataFrame: Dataframe containing custom metrics.
    '''

    # Creates a copy of the event dataframe to preserve the original data
    df = add_stage_changes(event_df.copy(), channels)
    
    # Creates a mask to identify baseline, initial, and final stages
    stage_lower = event_df['stage'].str.lower()
    mask = (stage_lower.str.startswith('baseline', na=False) | stage_lower.str.startswith(('initial', 'final'), na=False))

    # Creates a copy of the baselines dataframe to preserve the original data
    baselines = event_df[mask].copy()
    
    chip_id = event_df['chip_id'].iloc[0]
    
    custom_metrics = []
    
    # Checks whether sufficient baselines exist to calculate shifts
    if len(baselines) > 1:

        # Loops through each baseline index
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

def normalise_by_initial_flag(df, channels, drop_start=True):
    '''Zero-bases channel values using the 'Initial' stage flag if present.
    Falls back to 'Start' or the first available row if 'Initial' is missing.
    'Initial' is NEVER dropped. 'Start' is dropped only if drop_start=True.

    Args:
        df (pd.DataFrame): Dataframe to normalise.
        channels (list[str]): List of channels to normalise.
        drop_start (bool): Whether to drop the 'Start' stage.

    Returns:
        pd.DataFrame: Normalised dataframe.
    '''

    # Creates a copy of the dataframe to preserve the original data
    df_norm = df.copy()
    
    # Identifies the index of the reference stage for normalisation
    if (df_norm['stage'] == 'Initial').any():
        idx = df_norm[df_norm['stage'] == 'Initial'].index[0]
    elif (df_norm['stage'] == 'Start').any():
        idx = df_norm[df_norm['stage'] == 'Start'].index[0]
    else:
        idx = df_norm.index[0]
        
    # Loops through each channel
    for ch in channels:
        initial_val = df_norm.loc[idx, ch]
        df_norm[ch] = df_norm[ch] - initial_val
        
    # Conditionally drops the 'Start' stage row
    if drop_start:
        df_norm = df_norm[df_norm['stage'] != 'Start'].reset_index(drop=True)
    
    return df_norm

def extract_intrastage_features(sens_df, event_df, channels, chip_id):
    '''Slices raw sensorgram data between consecutive stage flags and extracts 
    kinetic and statistical intra-stage features for anomaly detection.

    Args:
        sens_df (pd.DataFrame): Sensorgram data.
        event_df (pd.DataFrame): Event flags data.
        channels (list[str]): List of channels to extract features for.
        chip_id (str): Identifier for the current chip.

    Returns:
        pd.DataFrame: Dataframe containing intra-stage features.
    '''

    time_col = sens_df.columns[0]
    sens_time = sens_df[time_col].values
    features_list = []

    # Sorts events chronologically
    events = event_df.sort_values('time').reset_index(drop=True)
    
    # Loops through each event index
    for i in range(len(events) - 1):

        start_time = events.loc[i, 'time']
        end_time = events.loc[i + 1, 'time']
        stage_transition = f"{events.loc[i, 'stage']} -> {events.loc[i+1, 'stage']}"

        if start_time == end_time:

            # Zero duration between flags
            stage_data = sens_df[sens_time == start_time]
        else:

            # Creates a mask to slice sensorgram data for the current stage
            mask = (sens_time >= start_time) & (sens_time < end_time)
            stage_data = sens_df[mask]
        
        # Initialises the features dictionary for the current row
        row_features = {
            'chip_id': chip_id,
            'stage': stage_transition
        }

        # Extract time array for the current stage (needed for slope)
        t_arr = stage_data[time_col].values
        
        # Loops through each channel
        for ch in channels:

            signal = stage_data[ch].values

            if len(signal) > 1:
                row_features[f'{ch}_mean'] = np.mean(signal)
                row_features[f'{ch}_median'] = np.median(signal)
                row_features[f'{ch}_std'] = np.std(signal)
                row_features[f'{ch}_min'] = np.min(signal)
                row_features[f'{ch}_max'] = np.max(signal)
                row_features[f'{ch}_range'] = np.ptp(signal)
                
                # Calculate linear slope
                t_shifted = t_arr - t_arr[0]
                slope, _ = np.polyfit(t_shifted, signal, 1)
                row_features[f'{ch}_slope'] = slope
                
                # Shape statistics 
                row_features[f'{ch}_skew'] = skew(signal)
                row_features[f'{ch}_kurtosis'] = kurtosis(signal)
                
            elif len(signal) == 1:

                # Fallback if only 1 data point is captured
                val = signal[0]
                row_features.update({
                    f'{ch}_mean': val, 
                    f'{ch}_median': val, 
                    f'{ch}_std': 0,
                    f'{ch}_min': val, 
                    f'{ch}_max': val, 
                    f'{ch}_range': 0,
                    f'{ch}_slope': 0, 
                    f'{ch}_skew': 0, 
                    f'{ch}_kurtosis': 0
                })
            else:

                # Fallback if 0 data points are captured
                row_features.update({
                    f'{ch}_mean': 0,
                    f'{ch}_median': 0, 
                    f'{ch}_std': 0,
                    f'{ch}_min': 0, 
                    f'{ch}_max': 0, 
                    f'{ch}_range': 0,
                    f'{ch}_slope': 0, 
                    f'{ch}_skew': 0, 
                    f'{ch}_kurtosis': 0
                })
                
        features_list.append(row_features)
        
    return pd.DataFrame(features_list)

# Parameter Logistic Equations and Curve Fits
def five_pl(x, bottom, top, ec50, hill, asym):
    '''Evaluates the five-parameter logistic model for standard-curve fitting.

    Args:
        x (array-like): Concentration values.
        bottom, top, ec50, hill, asym (float): Logistic-model coefficients.

    Returns:
        array-like: Predicted signal values.
    '''
    return top + (bottom - top) / (1 + (x / ec50)**(hill))**asym

def four_pl(x, bottom, top, ec50, hill):
    '''Evaluates the four-parameter logistic model for standard-curve fitting.

    Args:
        x (array-like): Concentration values.
        bottom, top, ec50, hill (float): Logistic-model coefficients.

    Returns:
        array-like: Predicted signal values.
    '''
    return top + (bottom - top) / (1 + (x / ec50)**(hill))

def fit_curve(data):
    '''Fits candidate logistic models and selects the best standard-curve result.

    Args:
        data (pd.DataFrame): Concentration and response measurements.

    Returns:
        tuple: Fitted curve values and the optimized parameters.
    '''

    # Extracts concentration and response arrays
    x, y = data['x'], data['y']
    xfit = np.logspace(np.log10(min(x)), np.log10(max(x)), 1000)

    # Evaluates and fits the 5-parameter logistic model
    if data['fit_model'] == '5PL':

        bounds = ([-np.inf, -np.inf, min(x)/100, -10, 0.01], [np.inf, np.inf, max(x)*100, 10, 10])

        p0 = [min(y), max(y), np.median(x), 1.0, 1.0]
        popt, _ = curve_fit(five_pl, x, y, p0=p0, bounds=bounds, maxfev=50000)
        yfit = five_pl(xfit, *popt)

    # Evaluates and fits the 4-parameter logistic model
    else:

        bounds = ([-np.inf, -np.inf, min(x)/100, -10], [np.inf, np.inf, max(x)*100, 10])

        p0 = [min(y), max(y), np.median(x), 1.0]
        popt, _ = curve_fit(four_pl, x, y, p0=p0, bounds=bounds, maxfev=50000)
        yfit = four_pl(xfit, *popt)
        
    return xfit, yfit, popt

# UI Display Helpers
class collapsible_output:

    def __init__(self, title):
        '''Initialises a collapsible output section.

        Args:
            title (str): Text displayed on the section toggle button.
        '''
        self.title = title
        
        # Configures the output layout and styling
        self.out = widgets.Output(layout=widgets.Layout(display='none', margin='10px 0 10px 15px'))
        self.out.add_class('custom-clean-output')
        
        self.expanded_label = f'[v] {self.title}'
        self.collapsed_label = f'[>] {self.title}'
        
        self.btn = widgets.Button(description=self.collapsed_label, layout=widgets.Layout(width='auto', border='none', padding='0', margin='0'))
        self.btn.style.button_color = 'transparent'
        self.btn.style.text_color = 'black' 
        self.btn.style.font_weight = 'bold'
        
        self.btn.on_click(self.toggle)
        self.container = widgets.VBox([self.btn, self.out])

    def toggle(self, *args):
        '''Shows or hides the output section when its toggle button is clicked.'''

        if self.out.layout.display == 'none':
            self.out.layout.display = 'block'
            self.btn.description = self.expanded_label 
        else:
            self.out.layout.display = 'none'
            self.btn.description = self.collapsed_label 

    def __enter__(self):
        '''Redirects displayed output into the collapsible section.

        Returns:
            collapsible_output: The active output container.
        '''
        
        display(self.container)
        
        # Injects custom CSS styling for the collapsible output
        display(HTML('''
        <style>
            .custom-clean-output, 
            .custom-clean-output .jp-RenderedText, 
            .custom-clean-output pre, 
            .custom-clean-output .output_area {
                color: black !important;
            }
            
            .custom-clean-output .output_area {
                display: flex;
                justify-content: flex-start !important;
                align-items: flex-start !important;
            }
            
            .custom-clean-output table {
                margin-left: 0 !important;
            }
            
            .jupyter-widgets.jupyter-button:hover,
            .jupyter-widgets.jupyter-button:focus {
                background-color: transparent !important;
                outline: none !important;
                box-shadow: none !important;
            }
        </style>
        '''))
        
        self.out.__enter__()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        '''Restores normal output handling when the section context closes.'''

        self.out.__exit__(exc_type, exc_value, traceback)

        return False
    
def generate_and_display_summary(data_list, val_cols, title, drop_na_col=None):
    '''Creates and displays a consolidated summary for a collection of dataframes.

    Args:
        data_list (list[pd.DataFrame]): Tables to combine into the summary.
        val_cols (list[str]): Measurement columns to summarise.
        title (str): Heading displayed with the summary.
        drop_na_col (str | None): Optional column used to exclude incomplete rows.

    Returns:
        pd.DataFrame: Generated summary table.
    '''
    
    # Checks whether the data list is populated
    if not data_list: 
        return None
    
    df = pd.concat(data_list, ignore_index=True)

    # Concatenates the dataframes and drops NA values if a column is specified
    if drop_na_col: 
        # Creates a copy of the dataframe to preserve the original data
        df = df.dropna(subset=[drop_na_col]).copy()
        
    # Applies categorical ordering to the stage column
    df['stage'] = pd.Categorical(df['stage'], categories=pd.unique(df['stage']), ordered=True)

    summary = build_stage_summary(df, 'stage', val_cols)

    # Display the summary table in a collapsible output section
    with collapsible_output(title): 
        display(summary)
        
    return df

# Core Method Runners
def run_pel_analysis(pel_upload_df, sensorgram_dfs, flags_PEL_df):
    '''Runs the complete PEL event, change, and intra-stage analysis workflow.

    Args:
        pel_upload_df (pd.DataFrame): PEL upload metadata.
        sensorgram_dfs (dict): Sensorgrams keyed by source file name.
        flags_PEL_df (pd.DataFrame): Processed PEL flag records.

    Returns:
        tuple: PEL event, change, intra-stage, and normalised.
    '''

    # Initialises the quality check output widget
    qc_output = collapsible_output('Quality Checks')

    with qc_output:
        check_data_quality(sensorgram_dfs, 'PEL')

    # Initialises the sensorgram plots output widget
    plot_output = collapsible_output('Sensorgram Plots')

    with plot_output:
        plot_sensorgrams(sensorgram_dfs, 'PEL')

    # Extracts unique time columns to define stage labels
    time_cols = pd.unique(flags_PEL_df.columns)[1:]

    # Declares lists to store event, change, and intra-stage dataframes
    all_events_PEL = []
    all_changes_PEL = []
    all_intrastage_PEL = []
    all_events_PEL_norm = []

    # Initialises the sensorgram flag plots output widget
    flag_output = collapsible_output('Sensorgram Flag Plots')

    with flag_output:

        # Loops through each row in the dataframe
        for _, upload_data in pel_upload_df.iterrows():

            # Extracts the chip ID for the current row
            chip_id = upload_data['chip_id']

            # Checks whether the sensorgram exists in the dictionary
            if upload_data['sensorgram'] in sensorgram_dfs.keys():
                sens = sensorgram_dfs[upload_data['sensorgram']]
            else:
                continue

            # Retrieves flag data corresponding to the current flags ID
            flag_data = flags_PEL_df[flags_PEL_df['flags_id'] == upload_data['flags_id']].iloc[0]
            times = flag_data[time_cols].values
            labels = time_cols

            # Plots the sensorgram with flag overlays
            plot_flags_on_sensorgrams(sens, times, labels, title=f'PEL Sensorgram for {chip_id}')

            # Declares channel variables
            channels = sens.columns.tolist()[1:]
            
            # Initialises a dictionary to hold stage and time events
            event_dict = {
                'stage': labels,
                'time': times,
                'chip_id': chip_id
            }
            
            # Loops through each channel to interpolate signal values at flag times
            for ch in channels:
                event_dict[ch] = np.interp(times, sens.iloc[:,0], sens[ch])

            # Converts the event dictionary to a dataframe
            event_df = pd.DataFrame(event_dict)

            all_events_PEL.append(event_df)

            # Displays raw metrics for the current chip
            print(f'Raw Metrics for Chip {chip_id}')
            display(event_df)

            # Creates a copy of the change dataframe to preserve the original data
            change_df = event_df.copy().drop(columns=['chip_id'])

            # Calculates stage changes and initial/final shifts
            change_df = add_stage_changes(change_df, channels)

            # Identifies the initial and final stage rows
            initial_row = event_df[event_df['stage'] == 'Initial']
            final_row = event_df[event_df['stage'] == 'Final']

            # Determines the appropriate time for the summary row
            summary_time = final_row['time'].iloc[0] if not final_row.empty else event_df['time'].iloc[-1]
            summary_row = {
                'stage': 'Total_Shift_Initial_to_Final',
                'time': summary_time
            }

            # Creates a copy of the event normalisation dataframe to preserve the original data
            event_norm_df = event_df.copy()
            
            # Creates a copy of the sensorgram normalisation dataframe to preserve the original data
            sens_norm = sens.copy()

            # Loops through each channel to calculate changes and normalise signals
            for ch in channels:

                val_initial = initial_row[ch].iloc[0] if not initial_row.empty else event_df[ch].iloc[0]
                val_final = final_row[ch].iloc[0] if not final_row.empty else event_df[ch].iloc[-1]

                summary_row[ch] = [val_initial, val_final]
                summary_row[f'{ch}_change'] = val_final - val_initial

                # Normalises channel signals against the initial value
                event_norm_df[ch] = event_norm_df[ch] - val_initial

                sens_norm[ch] = sens_norm[ch] - val_initial

            # Appends the summary row and chip ID to the change dataframe
            change_df = pd.concat([change_df, pd.DataFrame([summary_row])], ignore_index=True)
            change_df['chip_id'] = chip_id
            all_changes_PEL.append(change_df)
            
            # Drops the start stage from the normalised dataframe
            event_norm_df = event_norm_df[event_norm_df['stage'] != 'Start']
            
            all_events_PEL_norm.append(event_norm_df)

            # Displays normalised metrics for the current chip
            print(f'Normalised Metrics for Chip {chip_id}')
            display(event_norm_df)
            
            # Displays stage changes for the current chip
            print(f'Stage Changes for Chip {chip_id}')
            display(change_df)

            # Extracts intra-stage features
            intra_df = extract_intrastage_features(sens_norm, event_norm_df, channels, chip_id)
            all_intrastage_PEL.append(intra_df)

            display(intra_df)

    # Concatenates all event dataframes into a single dataframe
    all_events_PEL_df = pd.concat(all_events_PEL, ignore_index=True)
    
    # Applies categorical ordering to the stage column
    all_events_PEL_df['stage'] = pd.Categorical(all_events_PEL_df['stage'], categories=time_cols, ordered=True)

    # Builds the absolute stage metrics summary
    summary_PEL = build_stage_summary(all_events_PEL_df, 'stage', channels)

    # Initialises the PEL stage metrics output widget
    metrics_output = collapsible_output('PEL Stage Metrics Summary')

    # Displays the PEL stage metrics summary
    with metrics_output:
        display(summary_PEL)

    # Concatenates all normalised event dataframes
    all_events_PEL_norm_df = pd.concat(all_events_PEL_norm, ignore_index=True)
    
    # Filters out the start stage from the time columns
    norm_time_cols = [c for c in time_cols if c not in ['Start']]
    
    # Applies categorical ordering to the stage column
    all_events_PEL_norm_df['stage'] = pd.Categorical(all_events_PEL_norm_df['stage'], categories=norm_time_cols, ordered=True)
    
    # Builds the normalised stage metrics summary
    summary_PEL_norm = build_stage_summary(all_events_PEL_norm_df, 'stage', channels)

    # Initialises the PEL stage normalised metrics output widget
    norm_metrics_output = collapsible_output('PEL Stage Normalised Metrics Summary')

    # Displays the PEL stage normalised metrics summary
    with norm_metrics_output:
        display(summary_PEL_norm)

    # Concatenates all stage change dataframes
    all_changes_PEL_df = pd.concat(all_changes_PEL, ignore_index=True)
    
    # Creates a copy of the change PEL dataframe to preserve the original data
    change_PEL_df = all_changes_PEL_df.dropna(subset=[f'{ch}_change' for ch in channels]).copy()

    # Applies categorical ordering to the stage column
    stage_cols_pel = pd.unique(change_PEL_df['stage'])
    change_PEL_df['stage'] = pd.Categorical(change_PEL_df['stage'], categories=stage_cols_pel, ordered=True)
    
    # Builds the stage change metrics summary
    change_summary_PEL = build_stage_summary(change_PEL_df, 'stage', [f'{ch}_change' for ch in channels])

    # Initialises the PEL stage change metrics output widget
    change_output = collapsible_output('PEL Stage Change Metrics Summary')

    # Displays the PEL stage change metrics summary
    with change_output:
        display(change_summary_PEL)

    # Loops through each channel to plot statistical summaries
    for ch in channels:

        # Initialises the statistical plots output widget
        stats_output = collapsible_output(f'Statistical Plots ({ch})')

        with stats_output:
            plot_all_statistics(all_events_PEL_df, 'stage', ch, ylabel=f'{ch} Signal', title_prefix=f'Absolute - {ch}')
            plot_all_statistics(all_events_PEL_norm_df, 'stage', ch, ylabel=f'{ch} Normalised Signal', title_prefix=f'Normalised - {ch}')
            plot_all_statistics(change_PEL_df, 'stage', f'{ch}_change', ylabel=f'Delta {ch} Signal', title_prefix=f'Delta - {ch}')

    # Concatenates all intra-stage feature dataframes
    pel_intra_df = pd.concat(all_intrastage_PEL, ignore_index=True)            

    return all_events_PEL_df, change_PEL_df, pel_intra_df, all_events_PEL_norm_df

def run_immob_analysis(immob_df, immobilisation_dfs, flags_dfs, averaged_flags_dfs):
    '''Runs the complete immobilisation analysis workflow across all split types.

    Args:
        immob_df (pd.DataFrame): Immobilisation metadata.
        immobilisation_dfs (dict): RefPoly4 data keyed by source file name.
        flags_dfs (dict): Immediate flag data keyed by source file name.
        averaged_flags_dfs (dict): Five-second averaged flag data.

    Returns:
        tuple: Event, change, intra-stage and normalised..
    '''

    # Initialises the quality check collapsible output widget
    qc_output = collapsible_output('Quality Checks')

    with qc_output:
        check_data_quality(immobilisation_dfs, 'Immobilisation')

    # Initialises the sensorgram plots collapsible output widget
    sensor_output = collapsible_output('Sensorgram Plots')

    # Plots the raw sensorgrams dynamically
    with sensor_output:
        plot_sensorgrams(immobilisation_dfs, 'Immobilisation')

    # Initialises the flag calculations collapsible output widget
    calc_output = collapsible_output('Flag Calculations')

    with calc_output:

        print('Calculating and updating buffer, plateau, and baseline flags...')
        
        # Calculates and updates buffer, plateau, and baseline flags
        flags_dfs = update_immob_flags(immob_df, immobilisation_dfs, flags_dfs)

        print('Calculating 5-second averages...')

        # Calculates five-second averages for the detected flags
        averaged_flags_dfs = calculate_5s_averages(immob_df, immobilisation_dfs, flags_dfs, averaged_flags_dfs)

    # Declares channel-order variables for the comparison results
    all_events_imp, all_changes_imp, all_intra_imp = [], [], []
    all_events_comb, all_changes_comb, all_intra_comb = [], [], []
    all_events_avg_imp, all_changes_avg_imp, all_intra_avg_imp = [], [], []
    all_events_avg_comb, all_changes_avg_comb, all_intra_avg_comb = [], [], []

    all_events_imp_norm, all_events_comb_norm = [], []
    all_events_avg_imp_norm, all_events_avg_comb_norm = [], []

    # Loops through each row in the immobilisation dataframe
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

        # Interpolates sensorgram signal at the specified flag times
        for ch in channels: 
            event_dict[ch] = np.interp(times, sens.iloc[:, 0], sens[ch])
        
        event_df = pd.DataFrame(event_dict)

        all_events_imp.append(event_df)
        all_changes_imp.append(calculate_custom_immob_changes(event_df, channels))

        initial_row = event_df[event_df['stage'] == 'Initial']
        
        # Determines the initial time to establish a normalisation baseline
        if not initial_row.empty:
            initial_time = initial_row['time'].iloc[0]
        else:

            non_start_df = event_df[event_df['stage'] != 'Start']
            initial_time = non_start_df['time'].iloc[0] if not non_start_df.empty else event_df['time'].iloc[0]
            initial_row = event_df[event_df['time'] == initial_time]

        # Creates a copy of the sensorgram normalisation dataframe starting from the initial time
        sens_norm = sens[sens.iloc[:, 0] >= initial_time].copy()
        
        # Loops through each channel
        for ch in channels:
            
            # Zero-bases the normalisation dataframe using the initial value
            val_initial = initial_row[ch].iloc[0]
            sens_norm[ch] = sens_norm[ch] - val_initial

        event_norm_df = normalise_by_initial_flag(event_df, channels)
        event_norm_df = event_norm_df[event_norm_df['stage'] != 'Start']
        all_events_imp_norm.append(event_norm_df)
        all_intra_imp.append(extract_intrastage_features(sens_norm, event_norm_df, channels, chip_id))

        # Collapses consecutive combined reagents
        comb_df = collapse_combined_reagents(event_df)
        all_events_comb.append(comb_df)
        all_changes_comb.append(calculate_custom_immob_changes(comb_df, channels))

        comb_norm_df = normalise_by_initial_flag(comb_df, channels)
        comb_norm_df = comb_norm_df[comb_norm_df['stage'] != 'Start']
        all_events_comb_norm.append(comb_norm_df)
        all_intra_comb.append(extract_intrastage_features(sens_norm, comb_norm_df, channels, chip_id))

        if averaged_flags_dfs and immob_data['amfFlags'] in averaged_flags_dfs:

            # Creates a copy of the averaged flags dataframe to preserve the original data
            avg_flags = averaged_flags_dfs[immob_data['amfFlags']].copy()
            
            # Constructs the five-second averaged event dictionary
            avg_dict = {
                'stage': avg_flags['information'].tolist(),
                'time': ((avg_flags['Window_Start'].astype(float) + avg_flags['Window_End'].astype(float)) / 2).tolist(),
                'chip_id': chip_id
            }

            # Loops through each channel
            for ch in channels: 
                avg_dict[ch] = avg_flags[f'{ch}_avg'].tolist()
                
            avg_df = pd.DataFrame(avg_dict)
            all_events_avg_imp.append(avg_df)
            all_changes_avg_imp.append(calculate_custom_immob_changes(avg_df, channels))
            all_intra_avg_imp.append(extract_intrastage_features(sens, avg_df, channels, chip_id))
            all_events_avg_imp_norm.append(normalise_by_initial_flag(avg_df, channels, drop_start=False))

            avg_comb_df = collapse_combined_reagents(avg_df)
            all_events_avg_comb.append(avg_comb_df)
            all_changes_avg_comb.append(calculate_custom_immob_changes(avg_comb_df, channels))
            all_intra_avg_comb.append(extract_intrastage_features(sens, avg_comb_df, channels, chip_id))
            all_events_avg_comb_norm.append(normalise_by_initial_flag(avg_comb_df, channels, drop_start=False))

    flag_output = collapsible_output('Sensorgram Flag Plots')

    with flag_output:

        i = 0

        # Loops through each row in the immobilisation dataframe
        for _, immob_data in immob_df.iterrows():

            if immob_data['amfFlags'] not in flags_dfs.keys(): 
                continue

            sens = immobilisation_dfs[immob_data['refPoly4']]
            flag_data = flags_dfs[immob_data['amfFlags']]
            flag_data = flag_data[~flag_data.iloc[:, 1].str.contains('Concentration', case=False, na=False)]
            
            # Plots the flag markers on the sensorgrams
            plot_flags_on_sensorgrams(sens, flag_data['time'].tolist(), flag_data['information'].tolist(), title=f"Immob Sensorgram for {immob_data['chip_id']}", colours=['r' if 'Buffer' in str(lbl) else 'k' for lbl in flag_data['information']])

            print(f"Chip {immob_data['chip_id']} Metrics")
            
            # Displays chip metrics and changes
            display(all_events_imp[i])
            display(all_events_imp_norm[i])
            display(all_changes_imp[i])
            display(all_intra_imp[i])

            display(all_events_comb[i])
            display(all_events_comb_norm[i])
            display(all_changes_comb[i])
            display(all_intra_comb[i])

            i += 1

    if averaged_flags_dfs:

        avg_output = collapsible_output('Averaged Window Plots')

        with avg_output:

            i = 0

            # Loops through each row in the immobilisation dataframe
            for _, immob_data in immob_df.iterrows():

                if immob_data['amfFlags'] not in averaged_flags_dfs:
                    continue

                sens = immobilisation_dfs[immob_data['refPoly4']]
                avg_flags = averaged_flags_dfs[immob_data['amfFlags']]
                
                # Constructs and displays the five-second averaged window plots
                plot_5s_averaged_windows(sens, avg_flags, immob_data['chip_id'])

                print(f"Chip {immob_data['chip_id']} (5s Avg) Metrics")
                
                # Displays 5s average chip metrics and changes
                display(all_events_avg_imp[i])
                display(all_events_avg_imp_norm[i])
                display(all_changes_avg_imp[i])

                display(all_events_avg_comb[i])
                display(all_events_avg_comb_norm[i])
                display(all_changes_avg_comb[i])

                i += 1

    change_cols = [f'{ch}_change' for ch in channels]

    # Generates and displays the immediate stage metrics summary
    imp_df = generate_and_display_summary(all_events_imp, channels, 'Immediate: Metrics by Stage')
    imp_norm_df = generate_and_display_summary(all_events_imp_norm, channels, 'Immediate: Normalised Metrics by Stage')
    imp_change_df = generate_and_display_summary(all_changes_imp, change_cols, 'Immediate: Stage Changes', drop_na_col=change_cols[0])
    
    comb_df = generate_and_display_summary(all_events_comb, channels, 'Combined Reagents: Metrics by Stage')
    comb_norm_df = generate_and_display_summary(all_events_comb_norm, channels, 'Combined Reagents: Normalised Metrics by Stage')
    comb_change_df = generate_and_display_summary(all_changes_comb, change_cols, 'Combined Reagents: Stage Changes', drop_na_col=change_cols[0])

    imp_intra_df = pd.concat(all_intra_imp, ignore_index=True) if all_intra_imp else pd.DataFrame()
    comb_intra_df = pd.concat(all_intra_comb, ignore_index=True) if all_intra_comb else pd.DataFrame()

    avg_imp_intra_df = pd.DataFrame()
    avg_comb_intra_df = pd.DataFrame()

    if averaged_flags_dfs:

        avg_imp_df = generate_and_display_summary(all_events_avg_imp, channels, '5s Avg: Metrics by Stage')
        avg_imp_norm_df = generate_and_display_summary(all_events_avg_imp_norm, channels, '5s Avg: Normalised Metrics by Stage')
        avg_imp_change_df = generate_and_display_summary(all_changes_avg_imp, change_cols, '5s Avg: Stage Changes', drop_na_col=change_cols[0])
        
        avg_comb_df = generate_and_display_summary(all_events_avg_comb, channels, '5s Avg Combined Reagents: Metrics by Stage')
        avg_comb_norm_df = generate_and_display_summary(all_events_avg_comb_norm, channels, '5s Avg Combined Reagents: Normalised Metrics by Stage')
        avg_comb_change_df = generate_and_display_summary(all_changes_avg_comb, change_cols, '5s Avg Combined Reagents: Stage Changes', drop_na_col=change_cols[0])

        avg_imp_intra_df = pd.concat(all_intra_avg_imp, ignore_index=True) if all_intra_avg_imp else pd.DataFrame()
        avg_comb_intra_df = pd.concat(all_intra_avg_comb, ignore_index=True) if all_intra_avg_comb else pd.DataFrame()

    # Loops through each channel
    for ch in channels:

        stats_output = collapsible_output(f'Statistical Plots ({ch})')

        with stats_output:

            plot_all_statistics(imp_df, 'stage', ch, ylabel=f'{ch} Signal', title_prefix=f'Immediate Absolute - {ch}')        
            plot_all_statistics(imp_norm_df, 'stage', ch, ylabel=f'{ch} Norm Signal', title_prefix=f'Immediate Normalised - {ch}')        
            plot_all_statistics(imp_change_df, 'stage', f'{ch}_change', ylabel=f'Delta {ch}', title_prefix=f'Immediate Delta - {ch}')        
            
            plot_all_statistics(comb_df, 'stage', ch, ylabel=f'{ch} Signal', title_prefix=f'Combined Absolute - {ch}')        
            plot_all_statistics(comb_norm_df, 'stage', ch, ylabel=f'{ch} Norm Signal', title_prefix=f'Combined Normalised - {ch}')        
            plot_all_statistics(comb_change_df, 'stage', f'{ch}_change', ylabel=f'Delta {ch}', title_prefix=f'Combined Delta - {ch}')        
            
            if averaged_flags_dfs:

                plot_all_statistics(avg_imp_df, 'stage', ch, ylabel=f'{ch} Signal', title_prefix=f'5s Avg Absolute - {ch}')
                plot_all_statistics(avg_imp_norm_df, 'stage', ch, ylabel=f'{ch} Norm Signal', title_prefix=f'5s Avg Normalised - {ch}')
                plot_all_statistics(avg_imp_change_df, 'stage', f'{ch}_change', ylabel=f'Delta {ch}', title_prefix=f'5s Avg Delta - {ch}')
                
                plot_all_statistics(avg_comb_df, 'stage', ch, ylabel=f'{ch} Signal', title_prefix=f'5s Avg Comb Absolute - {ch}')
                plot_all_statistics(avg_comb_norm_df, 'stage', ch, ylabel=f'{ch} Norm Signal', title_prefix=f'5s Avg Comb Normalised - {ch}')
                plot_all_statistics(avg_comb_change_df, 'stage', f'{ch}_change', ylabel=f'Delta {ch}', title_prefix=f'5s Avg Comb Delta - {ch}')

    return (
        imp_df, imp_change_df, imp_intra_df, imp_norm_df,
        comb_df, comb_change_df, comb_intra_df, comb_norm_df,
        avg_imp_df, avg_imp_change_df, avg_imp_intra_df, avg_imp_norm_df,
        avg_comb_df, avg_comb_change_df, avg_comb_intra_df, avg_comb_norm_df
    )

def run_standard_curve_analysis(standard_curves_df):
    '''Fits and summarises standard curves for every valid chip measurement.

    Args:
        standard_curves_df (pd.DataFrame): Raw standard-curve records.

    Returns:
        tuple: Curve data dictionary and a dataframe of fit metrics.
    '''

    qc_output = collapsible_output('Standard Curve Quality Checks')

    with qc_output:

        duplicate_count = standard_curves_df['standard_curve_uuid'].duplicated().sum()

        # Counts and displays duplicated standard curve UUIDs
        print('Duplicated standard curve UUIDs:', duplicate_count)

        def parse_delimited(s):
            '''Parses a delimited string field into a list of numeric values.

            Args:
                s (str): Serialised list from a standard-curve record.

            Returns:
                list[float]: Parsed numeric values.
            '''

            sep = '|' if '|' in str(s) else ','
            return [float(v) for v in str(s).split(sep)]

        data_points_SC = {}

        # Loops through each row in the dataframe
        for row in standard_curves_df.itertuples():

            x = parse_delimited(row.x_csv)
            y = parse_delimited(row.y_csv)

            # Checks for mismatched coordinates
            if len(x) != len(y):
                print(f'Row {row.Index}: mismatched x/y lengths ({len(x)} vs {len(y)})')

            # Standardises the time string format and converts it to a datetime object
            time_str = str(row.time).strip()

            if '-' in time_str and ':' not in time_str:
                time_str = time_str.replace('-', ':')

            dt = pd.to_datetime(f'{row.date} {time_str}', format='%Y-%m-%d %H:%M:%S', errors='coerce')

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

        # Loops through each key-value pair in the dictionary
        for k, v in data_points_SC.items():

            # Evaluates and removes curves containing missing values
            if pd.isnull(v['x']).any() or pd.isnull(v['y']).any():

                print(f'Removing curve: {k} due to NaN values.')
                print(f"x:\n{v['x']}\ny:\n{v['y']}\n")
            else:
                clean_data_points_SC[k] = v
                
        print(f'Remaining valid curves: {len(clean_data_points_SC)}')

    stored_plot_output = collapsible_output('Standard Curve Plots (Stored Parameters)')

    with stored_plot_output:

        # Loops through each key-value pair in the dictionary
        for key, data in clean_data_points_SC.items():

            # Generates logarithmic points to plot the smooth fitted curve
            x_pts = np.logspace(np.log10(min(data['x'])), np.log10(max(data['x'])), 1000)

            # Evaluates the five-parameter logistic model using stored parameters
            if data['fit_model'] == '5PL':
                y_pts = five_pl(x_pts, data['a'], data['d'], data['c'], data['b'], data['e']) 
            else:
                y_pts = four_pl(x_pts, data['a'], data['d'], data['c'], data['b'])

            # Plots measured standard-curve points against the stored fit parameters
            plt.figure(figsize=(10, 6))
            plt.scatter(data['x'], data['y'], label='Data', color='blue')
            plt.plot(x_pts, y_pts, color='red', label='Stored Fit')

            plt.xscale('log')
            plt.title(f"Standard Curve {key} (Chip: {data['chip_id']}) - Stored Values")
            plt.xlabel('Concentration (ug/ml)')
            plt.ylabel('Signal')
            plt.legend()
            plt.show()

    calc_plot_output = collapsible_output('Standard Curve Plots (Calculated Parameters)')
    unique_chips_SC = list({v['chip_id'] for v in clean_data_points_SC.values()})
    
    with calc_plot_output:

        print(f'Total Unique Chips: {len(unique_chips_SC)}')

        # Loops through each chip ID
        for chip_id in unique_chips_SC:

            # Loops through each key-value pair in the dictionary
            for key, data in clean_data_points_SC.items():

                if data['chip_id'] != chip_id:
                    continue

                # Recalculates the logistic fit parameters for the current data
                xfit, yfit, _ = fit_curve(data)

                # Plots measured standard-curve points against the recalculated fit parameters
                plt.figure(figsize=(10, 6))
                plt.scatter(data['x'], data['y'], label='Data', color='blue')
                plt.plot(xfit, yfit, color='green', label='Calculated Fit')

                plt.xscale('log')
                plt.title(f'Standard Curve {key} (Chip: {chip_id}) - Calculated Fit')
                plt.xlabel('Concentration (ug/ml)')
                plt.ylabel('Signal')
                plt.legend()
                plt.show()

    sc_collected_data = []

    # Loops through each key-value pair in the dictionary
    for key, data in clean_data_points_SC.items():

        xfit, yfit, popt = fit_curve(data)

        if data['fit_model'] == '5PL':
            y_pred = five_pl(np.asarray(data['x']), *popt)
        else:
            y_pred = four_pl(np.asarray(data['x']), *popt)

        # Calculates the residual sum of squares and r-squared values for the fit
        ss_res = np.sum((np.asarray(data['y']) - y_pred) ** 2)
        ss_tot = np.sum((np.asarray(data['y']) - np.mean(data['y'])) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Appends the standard curve parameters and metadata to the collection list
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

# SC extra data
def calculate_sc_flags(files, save_dir='Standard Curve Files'):
    '''Calculates dynamic baseline and peak flags from sensorgram data, saves them to a CSV, updates the dictionary, and optionally plots the results.

    Args:
        files (dict): Dictionary holding the parsed file data.
        save_dir (str): Root directory where the CSV files should be saved.

    Returns:
        dict: The updated files dictionary containing the new baseline flag dataframes.
    '''

    # Loops through each folder in the files dictionary
    for folder in files.keys():
        previous_file = ''

        # Skips the folder if baseline flags are already present
        if any('baseline' in key for key in files[folder].keys()):
            continue

        # Loops through each file in the current folder
        for file in list(files[folder].keys()):

            if 'sensorgram' in file:

                file_data = files[folder][file].sort_values('time').reset_index(drop=True)
                previous_file_data = files[folder][previous_file].sort_values('time').reset_index(drop=True)

                # Calculates the median time step to define window row counts
                median_time_step = file_data['time'].diff().median()
                lookahead_rows = max(1, int(5 / median_time_step))
                
                total_block_rows = int(30 / median_time_step) 
                tail_sample_offset = int(5 / median_time_step)
                
                intervals = []

                # Loops through each previous file data index to extract interval times and measurements
                for i in range(len(previous_file_data) - 1):

                    t_start = previous_file_data.iloc[i]['time']
                    t_end = previous_file_data.iloc[i+1]['time']
                    meas_1 = str(previous_file_data.iloc[i]['conc']).split('-')[0]
                    meas_2 = str(previous_file_data.iloc[i+1]['conc']).split('-')[0]
                    intervals.append((t_start, t_end, meas_1, meas_2))
                    
                last_meas = str(previous_file_data.iloc[-1]['conc']).split('-')[0]
                intervals.append((previous_file_data.iloc[-1]['time'], file_data['time'].max(), last_meas, 'Final'))
                    
                saved_file_rows = []
                
                t_absolute_first = file_data['time'].min()
                t_start_init = t_absolute_first
                t_end_init = t_absolute_first + 5.0
                
                # Isolates the initial baseline zone at the absolute start of the recording
                init_zone = file_data[(file_data['time'] >= t_start_init) & (file_data['time'] <= t_end_init)]

                # Checks whether the dataframe contains data
                if not init_zone.empty:

                    saved_file_rows.append({
                        'Window_Start': t_start_init,
                        'Window_End': t_end_init,
                        'information': 'Initial Baseline',
                        'ch1_avg': init_zone['channel1'].mean(),
                        'ch2_avg': init_zone['channel2'].mean(),
                        'ch1_peak': np.nan,
                        'ch2_peak': np.nan,
                        'peak_time': np.nan
                    })
    
                # Loops through each enumerated interval
                for idx, (t_start, t_end, meas_1, meas_2) in enumerate(intervals):

                    zone = file_data[(file_data['time'] >= t_start) & (file_data['time'] <= t_end)]

                    # Checks whether the dataframe contains data
                    if zone.empty:
                        continue
                        
                    t_mid = t_start + (t_end - t_start) / 2

                    # Bisects the zone to isolate the relevant half based on measurement consistency
                    sub_zone = zone[zone['time'] >= t_mid] if meas_1 == meas_2 else zone[zone['time'] <= t_mid]

                    # Checks whether the dataframe contains data
                    if sub_zone.empty:
                        sub_zone = zone
    
                    # Calculates the signal drop size to identify the injection peak
                    drop_size = sub_zone['channel1'] - sub_zone['channel1'].shift(-lookahead_rows)

                    # Checks whether the dataframe contains data
                    if drop_size.dropna().empty:
                        drop_size = sub_zone['channel1'] - sub_zone['channel1'].shift(-1)

                    # Locates the maximum drop index to define the exact peak position
                    peak_global_idx = drop_size.idxmax()
                    exact_row = file_data.loc[peak_global_idx]

                    ch1_peak_val = exact_row['channel1']
                    ch2_peak_val = exact_row['channel2']
                    peak_time_val = exact_row['time']

                    max_zone_idx = zone.index.max()

                    # Determines the target baseline index by stepping forward, avoiding out-of-bounds errors
                    target_baseline_idx = peak_global_idx + total_block_rows - tail_sample_offset
        
                    if target_baseline_idx > max_zone_idx:
                        target_baseline_idx = max(peak_global_idx + lookahead_rows, max_zone_idx - tail_sample_offset)
                    
                    target_baseline_idx = max(peak_global_idx, min(target_baseline_idx, max_zone_idx))
                    baseline_row = file_data.loc[target_baseline_idx]
        
                    t_base = baseline_row['time']
                    t_start_win = t_base - 2.5
                    t_end_win = t_base + 2.5
                    
                    # Isolates the five-second window zone centred around the target baseline time
                    win_zone = file_data[(file_data['time'] >= t_start_win) & (file_data['time'] <= t_end_win)]
                    
                    ch1_win_avg = win_zone['channel1'].mean() if not win_zone.empty else baseline_row['channel1']
                    ch2_win_avg = win_zone['channel2'].mean() if not win_zone.empty else baseline_row['channel2']
                    
                    # Stores the calculated baseline and peak metrics into the row collection
                    saved_file_rows.append({
                        'Window_Start': t_start_win,
                        'Window_End': t_end_win,
                        'information': f'Baseline Window ({meas_1} / {meas_2})',
                        'ch1_avg': ch1_win_avg,
                        'ch2_avg': ch2_win_avg,
                        'ch1_peak': ch1_peak_val,
                        'ch2_peak': ch2_peak_val,
                        'peak_time': peak_time_val
                    })
                    
                t_absolute_last = file_data['time'].max()
                t_start_final = t_absolute_last - 5.0
                t_end_final = t_absolute_last
                
                # Isolates the final baseline zone at the end of the recording and calculates averages
                final_zone = file_data[(file_data['time'] >= t_start_final) & (file_data['time'] <= t_end_final)]
        
                saved_file_rows.append({
                    'Window_Start': t_start_final,
                    'Window_End': t_end_final,
                    'information': 'Final Baseline',
                    'ch1_avg': final_zone['channel1'].mean(),
                    'ch2_avg': final_zone['channel2'].mean(),
                    'ch1_peak': np.nan,
                    'ch2_peak': np.nan,
                    'peak_time': np.nan
                })

                print(f'Calculated flags for {folder}/{file}')
                flag_df = pd.DataFrame(saved_file_rows)
                file_name = file.split('_')[1] + '_baseline_flags.csv'
                file_path = os.path.join(save_dir, str(folder), file_name)
                
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Saves the calculated baseline flags to a new CSV file
                flag_df.to_csv(file_path, index=False)
        
                files[folder][file_name.replace('.csv', '')] = flag_df

            previous_file = file

    return files

def plot_baselines_and_peaks(file_data, previous_file_data, baseline_flags, title):
    '''Generates the detailed Plotly visualisation with tail-sampling windows and peaks.

    Args:
        file_data (pd.DataFrame): The raw sensorgram file data.
        previous_file_data (pd.DataFrame): The previous flag file data.
        baseline_flags (pd.DataFrame): The calculated baseline and peak flags.
        title (str): Title for the generated plot.
    '''

    # Creates a copy of the baseline flags to preserve the original data
    baseline_flags = baseline_flags.copy()
    baseline_flags['mid_time'] = (baseline_flags['Window_Start'] + baseline_flags['Window_End']) / 2
            
    # Plots the raw sensorgram with reagent flags, baseline windows, and detected peaks
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=file_data['time'], y=file_data['channel1'], mode='lines', name='Channel 1 Raw Data', line=dict(color='#1f77b4', width=1.5)))

    # Checks whether the dataframe contains data
    if previous_file_data is not None and not previous_file_data.empty:

        # Loops through each row in the dataframe
        for _, flag_row in previous_file_data.iterrows():

            fig.add_vline(x=flag_row['time'], line_width=1, line_dash='dash', line_color='black', opacity=1)
            fig.add_annotation(x=flag_row['time'], y=1.02, yref='paper', text=str(flag_row['conc']), showarrow=False, textangle=-45, xanchor='left', yanchor='bottom', font=dict(size=9, color='#0f0f0f'))

    fig.add_trace(go.Scatter(
        x=baseline_flags['mid_time'], y=baseline_flags['ch1_avg'],
        mode='markers', name='Saved Baseline (5s Avg)',
        marker=dict(color='red', symbol='circle', size=9),
        hoverinfo='text',
        text=[
            f"Type: {r['information']}<br>"
            f"Range: {r['Window_Start']:.1f}s to {r['Window_End']:.1f}s<br>"
            f"CH1 Avg: {r['ch1_avg']:.2f} RU<br>"
            f"CH2 Avg: {r['ch2_avg']:.2f} RU" 
            for _, r in baseline_flags.iterrows()
        ]
    ))

    # Evaluates and plots valid pre-drop peaks
    if 'ch1_peak' in baseline_flags.columns and 'peak_time' in baseline_flags.columns:

        valid_peaks = baseline_flags.dropna(subset=['ch1_peak', 'peak_time'])
        fig.add_trace(go.Scatter(
            x=valid_peaks['peak_time'], y=valid_peaks['ch1_peak'],
            mode='markers', name='Pre-Drop Peak',
            marker=dict(color='orange', symbol='x', size=9, line=dict(color='black', width=1)),
            hoverinfo='text',
            text=[
                f"Peak Value: {r['ch1_peak']:.2f} RU<br>Time: {r['peak_time']:.1f}s" 
                for _, r in valid_peaks.iterrows()
            ]
        ))
    
    # Loops through each row in the dataframe
    for _, r in baseline_flags.iterrows():

        # Adds a shaded vertical rectangle for each baseline window
        fig.add_vrect(
            x0=r['Window_Start'], x1=r['Window_End'],
            fillcolor='rgba(44, 160, 44, 0.12)', layer='below', 
            line_width=1, line_color='rgba(44, 160, 44, 0.4)', line_dash='dot'
        )
    
    start_avg_ru = baseline_flags.iloc[0]['ch1_avg']
    fig.add_hline(y=start_avg_ru, line_width=1.5, line_dash='dash', line_color='purple', annotation_text='Start Average', annotation_position='top left')

    final_avg_ru = baseline_flags.iloc[-1]['ch1_avg']
    fig.add_hline(y=final_avg_ru, line_width=1.5, line_dash='dash', line_color='#e67e22', annotation_text='Final Average', annotation_position='bottom left')

    # Configures the layout and styling of the Plotly figure
    fig.update_layout(
        title=title,
        xaxis_title='Time (Seconds)',
        yaxis_title='Response Units (RU)',
        yaxis=dict(range=[file_data['channel1'][0] - 0.5, file_data['channel1'].max() + 0.5]),
        template='plotly_white',
        hovermode='closest'
    )
    
    fig.show()

def calculate_and_plot_shift(shift_data, title):
    '''Calculates baseline shifts and plots the twin-axis visualisation.

    Args:
        shift_data (pd.DataFrame): Dataframe containing the baseline shift data.
        title (str): Title for the generated plot.
    '''

    # Creates a copy of the shift data to preserve the original data
    shift_data = shift_data.copy()
    initial_baseline = shift_data['ch1_avg'].iloc[0]

    # Calculates absolute baseline shifts and percentage of peak response
    baseline_shift = shift_data['ch1_avg'] - initial_baseline
    peak_response = shift_data['ch1_peak'] - initial_baseline
    shift_percentage = (baseline_shift / peak_response) * 100
    
    shift_data['baseline_shift_raw'] = baseline_shift
    shift_data['peak_response_raw'] = peak_response
    shift_data['ch1_shift_pct'] = shift_percentage

    summary_cols = ['information', 'ch1_peak', 'ch1_avg', 'baseline_shift_raw', 'ch1_shift_pct']

    # Creates a copy of the summary table to preserve the original data
    summary_table = shift_data[summary_cols].copy()
    
    # Standardises the summary table column names
    summary_table.columns = ['Measurement Step', 'Absolute Peak (RU)', 'Absolute Baseline (RU)', 'Baseline Shift (ΔRU)', '% Shift of Peak']
    summary_table = summary_table.round(3)
    
    # Initialises the plot with twin y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x_times = (shift_data['Window_Start'] + shift_data['Window_End']) / 2
    
    # Plots the absolute baseline shift on the primary y-axis
    ax1.plot(x_times, baseline_shift, 'x-', color='b', label='Baseline Shift (Absolute)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Response (a.u.)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Plots the percentage shift on the secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(x_times, shift_percentage, 'o--', color='r', alpha=0.6, label='% Shift of Peak')
    ax2.set_ylabel('% of Peak Response', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title(f'{title} - Baseline-shifted response (initial = {initial_baseline:.2f})')
    ax1.grid(True, alpha=0.3)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.show()

    # Display the summary table
    display(summary_table)

def analyse_standard_curves_extra(files, save_dir='Standard Curve Files'):
    '''Iterates over the processed files dictionary, generating and displaying all requested outputs neatly inside collapsible UI widgets.

    Args:
        files (dict): Processed dictionary containing standard curve data.
        save_dir (str): Directory where standard curve files are stored.

    Returns:
        dict: The updated files dictionary.
    '''

    files = calculate_sc_flags(files)

    # Loops through each folder in the files dictionary
    for folder in files.keys():

        print(f'\nEvaluating Folder: {folder}')
        
        previous_file = ''

        # Loops through each file in the folder
        for file in list(files[folder].keys()):

            # Cleans the filename and extracts chip and measurement IDs
            clean_filename = file.replace('Copy of ', '').strip()
            parts = clean_filename.split('_')
            chip_id = parts[0] if len(parts) > 0 else 'Unknown'
            measurement_id = parts[1] if len(parts) > 1 else 'Unknown'

            if 'sensorgram' in file:

                print(f'\n--- Chip ID: {chip_id} | Measurement ID: {measurement_id} ---')

                file_data = files[folder][file]
                previous_file_data = files[folder].get(previous_file, None)

                # Display the basic sensorgram plot
                with collapsible_output(f'Basic Sensorgram: {file}'):
                    plot_sensorgrams({file: file_data}, 'Standard Curve')
                
                # Checks whether the dataframe contains data
                if previous_file_data is not None and not previous_file_data.empty:
                    
                    with collapsible_output(f'Sensorgram with Flags: {file}'):

                        times = previous_file_data['time'].tolist()
                        labels = previous_file_data['conc'].tolist()
                        title = f'Standard Curve Sensorgram for {file}'
                        colours = ['tab:blue'] * len(times)
                        
                        # Display the flags on the sensorgram
                        plot_flags_on_sensorgrams(file_data, times, labels, title, colours)

                file_name_flags = file.split('_')[1] + '_baseline_flags'

                if file_name_flags in files[folder]:

                    baseline_flags = files[folder][file_name_flags]

                    # Display the baselines and peaks plot
                    with collapsible_output(f'Baseline & Peak Tail-Sampling: {file}'):
                        plot_baselines_and_peaks(file_data, previous_file_data, baseline_flags, f'Tail-Sampling Window Method with Peaks - {file}')

                    # Display the calculated baseline shift plots
                    with collapsible_output(f'Baseline Shift Analysis: {file}'):
                        calculate_and_plot_shift(baseline_flags, file)

            previous_file = file

    return files
