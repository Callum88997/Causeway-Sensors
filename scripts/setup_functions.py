# Imports required packages
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from scipy.optimize import curve_fit
from IPython.display import display, HTML
import ipywidgets as widgets
from scipy.stats import skew

# Data Quality and Visualisation
def check_data_quality(dfs_dict, stage_name):
    '''Prints missing values, duplicated rows, and summary statistics for generic sensorgrams.

    Args:
        dfs_dict (dict): Dictionary of sensorgram dataframes.
        stage_name (str): The name of the experimental stage.
    '''

    # Loops through each key-value pair in the dictionary
    for name, df in dfs_dict.items():
        
        # Displays the missing values and duplicated rows metrics header
        print(f'Metrics for {stage_name} - {name}:\n')

        # Prints the total sum of missing values across the dataframe
        print(f'\tMissing values: {df.isnull().sum().sum()}')

        # Prints the total count of duplicated rows in the dataframe
        print(f'\tDuplicated rows: {df.duplicated().sum()}\n')

        # Extracts the channel column names by skipping the time column
        channels = df.columns.tolist()[1:]

        # Loops through each extracted chip channel
        for chip in channels:

            # Prints the specific channel metrics header
            print(f'\t{chip.capitalize()} Metrics:')
            
            # Displays summary statistics for the current chip channel
            display(df[chip].describe())

            # Displays the first five rows of the channel data
            display(df[chip].head())

            # Displays the last five rows of the channel data
            display(df[chip].tail())

def plot_sensorgrams(dfs_dict, stage_name):
    '''Plots raw sensorgrams, dynamically finding time and channel columns.

    Args:
        dfs_dict (dict): Dictionary of sensorgram dataframes.
        stage_name (str): The name of the experimental stage.
    '''

    # Loops through each key-value pair in the dictionary
    for name, df in dfs_dict.items():

        # Extracts the time column name from the first position
        time_col = df.columns[0]

        # Extracts the channel column names by skipping the time column
        channels = df.columns.tolist()[1:]

        # Initialises a Matplotlib figure with specific dimensions
        plt.figure(figsize=(10, 6))
        
        # Loops through each channel
        for ch in channels:

            # Plots the raw channel response against time for the current sensorgram
            plt.plot(df[time_col], df[ch], label=ch)

        # Sets the title for the sensorgram plot
        plt.title(f'{stage_name} Data for {name}')

        # Sets the x-axis label for time
        plt.xlabel('Time (s)')

        # Sets the y-axis label for response units
        plt.ylabel('RU')

        # Adds a legend to the plot
        plt.legend()

        # Renders the sensorgram plot
        plt.show()

def plot_flags_on_sensorgrams(df_sens, times, labels, title, colours=None):
    '''Overlays vertical flag lines and text on a generic sensorgram.

    Args:
        df_sens (pd.DataFrame): Sensorgram data.
        times (list[float]): List of time points for the flags.
        labels (list[str]): List of labels corresponding to the time points.
        title (str): Title of the plot.
        colours (list[str], optional): List of colours for the flags. Defaults to None.
    '''

    # Extracts the time column name from the first position
    time_col = df_sens.columns[0]

    # Extracts the channel column names by skipping the time column
    channels = df_sens.columns.tolist()[1:]

    # Assigns the provided colours or defaults to black for all flags
    colours = colours or ['k'] * len(times)

    # Initialises the plot figure and axes with specific dimensions
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Loops through each time, label, and colour tuple with its index
    for i, (t, label, c) in enumerate(zip(times, labels, colours)):
        
        # Draws a dashed vertical line at the specified time point
        ax.axvline(x=t, linestyle='--', color=c)

        # Evaluates if the label indicates a plateau
        if 'plateau' in label.lower():

            # Annotates the plateau flag at the top of the plot
            ax.text(t, 1.1, label, rotation=90, transform=ax.get_xaxis_transform(), verticalalignment='bottom', horizontalalignment='center', color=c)
        
        # Evaluates if the label indicates a baseline
        elif 'baseline' in label.lower():

            # Annotates the baseline flag at the middle of the plot with an angle
            ax.text(t, 0.5, label, rotation=45, transform=ax.get_xaxis_transform(), verticalalignment='center', horizontalalignment='center', color=c)
        
        # Handles all other label types
        else:

            # Annotates the standard flag at the bottom of the plot
            ax.text(t, -0.1, label, rotation=90, transform=ax.get_xaxis_transform(), verticalalignment='top', horizontalalignment='center', color=c)

    # Loops through each channel
    for ch in channels:

        # Plots the channel response data against time
        ax.plot(df_sens[time_col], df_sens[ch], label=ch)

    # Sets the title for the overlaid sensorgram plot
    ax.set_title(title)

    # Sets the x-axis label for time
    ax.set_xlabel('Time (s)')

    # Sets the y-axis label for response units
    ax.set_ylabel('RU')

    # Adds a legend to the plot
    ax.legend()

    # Renders the overlaid sensorgram plot
    plt.show()

def plot_5s_averaged_windows(sens, avg_flags, title_id):
    '''Plots sensorgram responses with five-second averaging windows highlighted, specific to immobilisation data.

    Args:
        sens (pd.DataFrame): Raw sensorgram data.
        avg_flags (pd.DataFrame): Processed flags containing window boundaries.
        title_id (str): Identifier to include in the plot title.
    '''

    # Extracts the time column name from the first position
    time_col = sens.columns[0]

    # Extracts the channel column names by skipping the time column
    channels = sens.columns.tolist()[1:]
    
    # Initialises a Matplotlib figure with specific dimensions
    plt.figure(figsize=(10, 6))

    # Loops through each channel
    for ch in channels:

        # Plots the raw sensorgram response against time
        plt.plot(sens[time_col], sens[ch], label=ch)
        
    # Loops through each row in the averaged flags dataframe
    for _, row in avg_flags.iterrows():

        # Extracts the start and end times for the current averaging window
        w_start, w_end = row['Window_Start'], row['Window_End']

        # Highlights the averaging window with a shaded vertical span
        plt.axvspan(w_start, w_end, color='black', alpha=0.2, linestyle='--')
        
        # Calculates the midpoint of the flag window for text placement
        t_flag = (float(w_start) + float(w_end)) / 2

        # Retrieves the calculated average value for the primary channel or defaults to zero
        avg_val = row.get(f'{channels[0]}_avg', 0)

        # Annotates the highlighted window with its descriptive label and average value
        plt.text(t_flag, avg_val, f" {row['information']} ({row['Window_Label']})", fontsize=8, verticalalignment='bottom', rotation=15, color='darkred')
                  
    # Sets the title incorporating the provided identifier
    plt.title(f'Averaging Windows: {title_id}')

    # Sets the x-axis label for time
    plt.xlabel('Time (s)')

    # Sets the y-axis label for the signal
    plt.ylabel('Signal')

    # Adds a subtle background grid to the plot
    plt.grid(True, alpha=0.3)

    # Adds a legend to the plot
    plt.legend()

    # Renders the highlighted sensorgram plot
    plt.show()

def plot_all_statistics(df, x_col, y_col, ylabel='Signal', title_prefix=''):
    '''Generates box, violin, density, correlation, and heatmap plots sequentially for a sensorgram stage metrics or changes dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing the statistics to plot.
        x_col (str): Column name for the x-axis.
        y_col (str): Column name for the y-axis.
        ylabel (str, optional): Label for the y-axis. Defaults to 'Signal'.
        title_prefix (str, optional): Prefix to prepend to plot titles. Defaults to ''.
    '''
    
    # Formats the title prefix with a trailing dash if it is provided
    title_prefix = f'{title_prefix} - ' if title_prefix else ''

    # Sets the global plot height for consistent aesthetics
    plot_height = 650

    # Sets the global colour scheme for the categorical plots
    colour_scheme = px.colors.qualitative.Vivid

    # Safely determines hover columns to prevent KeyError if 'chip_id' is missing
    hover_cols = ['chip_id'] if 'chip_id' in df.columns else None

    # Generates a boxplot and stripplot representing the spread of signal values per stage
    fig_box = px.box(df, x=x_col, y=y_col, color=x_col, points='all', hover_data=hover_cols, title=f'{title_prefix}Stage Boxplot & Stripplot', labels={x_col: 'Experimental Stage', y_col: ylabel}, color_discrete_sequence=colour_scheme, template='plotly_white')

    # Updates the trace styling to adjust jitter, opacity, and borders
    fig_box.update_traces(jitter=0.2, pointpos=0, marker=dict(opacity=0.5, line=dict(width=1, color='black'))) 

    # Updates the layout to enforce dimensions and normal y-axis scaling
    fig_box.update_layout(autosize=True, height=plot_height, showlegend=False, yaxis=dict(autorange=True, rangemode='normal'))

    # Renders the boxplot figure
    fig_box.show()

    # Generates a violin plot to display the data distribution across different stages
    fig_violin = px.violin(df, x=x_col, y=y_col, color=x_col, box=False, hover_data=hover_cols, title=f'{title_prefix}Stage Violin Plot', labels={x_col: 'Experimental Stage', y_col: ylabel}, color_discrete_sequence=colour_scheme, template='plotly_white')
                           
    # Updates the layout to enforce dimensions and angle the x-axis labels
    fig_violin.update_layout(autosize=True, height=plot_height, showlegend=False, xaxis_tickangle=45)

    # Renders the violin plot figure
    fig_violin.show()

    # Drops rows with missing values in the target columns for density plotting
    df_clean = df.dropna(subset=[x_col, y_col])

    # Extracts the unique categorical stages from the cleaned dataframe
    categories = df_clean[x_col].unique()
    
    # Initialises an empty list to store the sliced density data
    hist_data = []

    # Initialises an empty list to store the category group labels
    group_labels = []

    # Loops through each unique categorical stage
    for cat in categories:

        # Slices the dataframe to isolate the y-values for the current category
        cat_data = df_clean[df_clean[x_col] == cat][y_col]
        
        # Evaluates if the category data has sufficient variance for kernel density estimation
        if cat_data.nunique() > 1:

            # Appends the valid category data to the list
            hist_data.append(cat_data)

            # Appends the string representation of the category to the labels list
            group_labels.append(str(cat))

    # Checks whether any valid data remains for the density plot
    if hist_data:

        # Creates the overlapping density curves, hiding underlying histogram bars and rug marks
        fig_density = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False, colors=colour_scheme[:len(hist_data)])
        
        # Updates the layout with appropriate titles and dimensions
        fig_density.update_layout(title=f'{title_prefix}Signal Density Profiles Per Stage', xaxis_title=ylabel, yaxis_title='Density', autosize=True, height=plot_height, template='plotly_white')

        # Renders the density plot figure
        fig_density.show()

    else:

        # Prints a warning message indicating insufficient variance for density estimation
        print(f"Skipping Density Plot: Not enough variance in '{y_col}' across '{x_col}' to calculate KDE.")

    # Initialises a Matplotlib figure for the correlation matrix heatmap
    plt.figure(figsize=(10, 6))

    # Calculates the correlation matrix for all numeric columns in the dataframe
    corr_matrix = df.corr(numeric_only=True)

    # Checks whether the correlation matrix contains data
    if not corr_matrix.empty:

        # Generates a heatmap visualisation of the correlation matrix
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)

        # Sets the title for the correlation heatmap
        plt.title(f'{title_prefix}Channel & Time Correlation Matrix')

        # Adjusts the layout to prevent clipping
        plt.tight_layout()

        # Renders the correlation heatmap
        plt.show()

    # Closes the Matplotlib figure to free memory
    plt.close()

    # Evaluates if the chip identifier column is present in the dataframe
    if 'chip_id' in df.columns:

        # Initialises a Matplotlib figure for the overview heatmap
        plt.figure(figsize=(10, 6))

        # Pivots the dataframe to calculate mean signal values per chip per stage
        pivot_df = df.pivot_table(index='chip_id', columns=x_col, values=y_col, aggfunc='mean', observed=False)
        
        # Generates a heatmap visualisation of the pivoted overview data
        sns.heatmap(pivot_df, annot=False, cmap='viridis', fmt='.2f', cbar_kws={'label': ylabel})

        # Angles the x-axis tick labels for readability
        plt.xticks(rotation=45)

        # Sets the y-axis label
        plt.ylabel('Chip ID')

        # Sets the x-axis label
        plt.xlabel('Experimental Stage')

        # Sets the title for the overview heatmap
        plt.title(f'{title_prefix}Overview Heatmap (Mean Value per Chip per Stage)')

        # Adjusts the layout to prevent clipping
        plt.tight_layout()

        # Renders the overview heatmap
        plt.show()

        # Closes the Matplotlib figure to free memory
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

    # Calculates aggregate statistics grouped by the specified column, generating multiple metrics per value column
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

    # Loops through each specified value column to calculate the coefficient of variation percentage
    for col in val_cols:

        # Calculates and assigns the coefficient of variation percentage
        summary[f'{col}_cv_pct'] = 100 * summary[f'{col}_std'] / summary[f'{col}_mean'].abs()

    # Initialises a list with the foundational columns to enforce a specific output order
    ordered_cols = [group_col, 'n']

    # Defines the standard sequence of statistical suffixes
    stat_suffixes = ['mean', 'median', 'std', 'min', 'max', 'iqr', 'cv_pct']
    
    # Loops through each value column to sequence the generated statistics
    for col in val_cols:

        # Loops through each defined statistic suffix
        for stat in stat_suffixes:

            # Constructs the expected column name
            col_name = f'{col}_{stat}'

            # Evaluates if the constructed column name exists in the summary dataframe
            if col_name in summary.columns:

                # Appends the validated column name to the ordering list
                ordered_cols.append(col_name)
                
    # Loops through every column present in the generated summary dataframe
    for col in summary.columns:

        # Checks if the column is missing from the designated ordering list
        if col not in ordered_cols:

            # Appends the remaining column to the end of the ordering list
            ordered_cols.append(col)

    # Returns the summary dataframe filtered and ordered by the constructed column list
    return summary[ordered_cols]

def add_stage_changes(event_table, channels):
    '''Calculates differences dynamically for provided channel columns and labels stage transitions.

    Args:
        event_table (pd.DataFrame): Dataframe containing event data.
        channels (list[str]): List of channel columns to process.

    Returns:
        pd.DataFrame: Dataframe with calculated stage changes.
    '''

    # Loops through each provided channel to calculate its stage-to-stage difference
    for ch in channels:

        # Calculates the numerical difference and stores it in a new change column
        event_table[f'{ch}_change'] = event_table[ch].diff()

    # Shifts the stage column values downwards and concatenates them to create transition labels
    event_table['stage'] = event_table['stage'].shift(1) + ' -> ' + event_table['stage']

    # Removes the first row, which contains NaN values resulting from the shift operation, and resets the index
    event_table = event_table.iloc[1:].reset_index(drop=True)

    # Returns the updated dataframe containing the stage changes
    return event_table

def extract_flags_from_pushes(flag_df):
    '''Parses a flag dataframe to extract buffers, reagents, and their plateaus based on the syringe push event sequences.

    Args:
        flag_df (pd.DataFrame): Raw flag dataframe.

    Returns:
        pd.DataFrame: Parsed and ordered flag dataframe.
    '''

    # Creates a copy of the flag dataframe, sorts it chronologically, and resets the index
    df = flag_df.copy().sort_values(by='time').reset_index(drop=True)
    
    # Creates a boolean mask identifying rows that correspond to syringe push events
    is_push = df['information'].str.contains("No 'Concentration'", na=False)

    # Initialises an empty list to store newly generated synthetic records
    new_records = []

    # Identifies the indices of all non-push reagent events
    reagent_indices = df.index[~is_push].tolist()
    
    # Checks whether any reagent indices were found
    if not reagent_indices:

        # Assigns infinite placeholder values if no reagents are present
        first_reagent_time = float('inf')

        # Assigns infinite placeholder values if no reagents are present
        last_reagent_plateau_time = float('inf')

    else:

        # Extracts the timestamp of the first detected reagent event
        first_reagent_time = df.loc[reagent_indices[0], 'time']

        # Extracts the timestamp of the last detected reagent event to initialise plateau tracking
        last_reagent_plateau_time = df.loc[reagent_indices[-1], 'time']

    # Loops through each identified reagent index
    for idx in reagent_indices:

        # Extracts the name of the reagent from the information column
        reagent_name = df.loc[idx, 'information']
        
        # Evaluates if a subsequent row exists in the dataframe
        if idx + 1 < len(df):

            # Extracts the timestamp of the immediate subsequent row to define the plateau
            plateau_time = df.loc[idx + 1, 'time']

            # Appends a newly generated synthetic plateau record to the tracking list
            new_records.append({'time': plateau_time, 'information': f'{reagent_name} - Plateau'})

            # Updates the last reagent plateau time if the current one is greater
            last_reagent_plateau_time = max(last_reagent_plateau_time, plateau_time)

    # Creates a copy of the dataframe containing only the syringe push events
    push_df = df[is_push].copy()

    # Checks whether the extracted push dataframe is empty
    if push_df.empty:

        # Returns the original dataframe stripped of push events, ignoring synthetic generation
        return df[~is_push].copy().reset_index(drop=True)

    # Calculates the time difference between consecutive push events
    push_df['time_diff'] = push_df['time'].diff().fillna(0)

    # Groups the push events into clusters based on a time gap threshold
    push_df['group'] = (push_df['time_diff'] > 150).cumsum()

    # Filters and extracts valid clusters containing at least three push events
    valid_bunches = [group for _, group in push_df.groupby('group') if len(group) >= 3]

    # Filters the valid clusters that occur entirely before the first reagent event
    bunches_before = [b for b in valid_bunches if b['time'].max() < first_reagent_time]

    # Filters the valid clusters that occur entirely after the last reagent plateau
    bunches_after = [b for b in valid_bunches if b['time'].min() > last_reagent_plateau_time]

    # Evaluates if any valid push clusters exist before the reagents
    if bunches_before:

        # Selects the largest push cluster occurring before the reagents
        bunch = max(bunches_before, key=len)

        # Identifies the index of the final event in the selected cluster
        last_idx = bunch.index[-1]

        # Appends a synthetic record denoting the first buffer phase
        new_records.append({'time': df.loc[last_idx, 'time'], 'information': 'Buffer 1 - NHS EDC'})

        # Evaluates if a subsequent row exists in the parent dataframe
        if last_idx + 1 < len(df):

            # Appends a synthetic record denoting the plateau of the first buffer phase
            new_records.append({'time': df.loc[last_idx + 1, 'time'], 'information': 'Buffer 1 - Plateau'})

    # Defines a mapping dictionary for subsequent buffer suffix labels
    buffer_suffixes = {2: 'ETA', 3: 'Casein Block'}

    # Loops through up to the first two valid push clusters occurring after the reagents
    for i, bunch in enumerate(bunches_after[:2]):
        
        # Calculates the buffer sequence number based on the iteration index
        buffer_num = i + 2

        # Identifies the index of the final event in the current cluster
        last_idx = bunch.index[-1]

        # Retrieves the descriptive suffix from the mapping dictionary or defaults to an empty string
        suffix = buffer_suffixes.get(buffer_num, '')

        # Constructs the full buffer name incorporating the suffix if present
        buffer_name = f'Buffer {buffer_num} - {suffix}' if suffix else f'Buffer {buffer_num}'

        # Appends a synthetic record denoting the constructed buffer phase
        new_records.append({'time': df.loc[last_idx, 'time'], 'information': buffer_name})
        
        # Evaluates if a subsequent row exists in the parent dataframe
        if last_idx + 1 < len(df):

            # Appends a synthetic record denoting the plateau of the constructed buffer phase
            new_records.append({'time': df.loc[last_idx + 1, 'time'], 'information': f'Buffer {buffer_num} - Plateau'})

    # Creates a copy of the original dataframe stripped of raw push events
    final_df = df[~is_push].copy()

    # Concatenates the stripped dataframe with all the newly generated synthetic records
    final_df = pd.concat([final_df, pd.DataFrame(new_records)], ignore_index=True)
    
    # Returns the final combined dataframe sorted chronologically with a reset index
    return final_df.sort_values(by='time').reset_index(drop=True)

def parse_stage_label(info):
    '''Extracts a concise stage label from an immobilisation flag description.

    Args:
        info (str): Raw descriptive text from a flag record.

    Returns:
        tuple: Normalised stage label variables suitable for analysis output.
    '''
    
    # Splits the raw string to remove plateau suffixes and strips trailing whitespace
    stage_raw = re.split(r'\s*-\s*Plateau', str(info))[0].strip()

    # Evaluates if the raw stage name begins with buffer and contains a hyphen
    if stage_raw.lower().startswith('buffer') and '-' in stage_raw:

        # Splits the string at the hyphen and retains the primary buffer designation
        stage_raw = stage_raw.split('-')[0].strip()

    # Converts the raw stage string to lowercase for standardisation
    stage = stage_raw.lower()

    # Extracts the first word of the stage string to determine the reagent type
    reagent_type = stage.split()[0] if stage.split() else ''

    # Evaluates if the extracted reagent type designates a buffer
    is_buffer = reagent_type == 'buffer'

    # Searches the standard lowercase stage string for chip identifiers
    chip_match = re.search(r'chip\s*(\d+)', stage)

    # Searches the standard lowercase stage string for flow cell identifiers
    fc_match = re.search(r'fc\s*(\d+)', stage)

    # Extracts all numeric sequences present within the stage string
    all_numbers = re.findall(r'\d+', stage)

    # Evaluates if a specific chip identifier was successfully matched
    if chip_match:

        # Assigns the extracted chip sequence as an integer
        chip_num = int(chip_match.group(1))

    # Evaluates if a specific flow cell identifier was successfully matched
    elif fc_match:

        # Assigns the extracted flow cell sequence as an integer
        chip_num = int(fc_match.group(1))

    # Evaluates if any generic numerical sequences were extracted
    elif all_numbers:

        # Assigns the final numerical sequence found as an integer
        chip_num = int(all_numbers[-1])

    else:

        # Assigns a null value if no identifiers or numbers are present
        chip_num = None

    # Returns the tuple containing all parsed and standard variables
    return stage_raw, stage, reagent_type, chip_num, is_buffer

def calculate_baseline_peaks(time, reagent_flags):
    '''Calculates baseline and peak sampling positions from reagent flag timings.

    Args:
        time (pd.Series): Sensorgram time values.
        reagent_flags (pd.DataFrame): Flag records defining reagent transitions.

    Returns:
        list[dict]: Baseline and peak sampling definitions for each stage.
    '''
    
    # Initialises an empty list to store the calculated baseline records
    baseline_records = []
    
    # Creates a filtered copy of the flags retaining specific identifiers and excluding baseline markers
    valid_flags = reagent_flags[reagent_flags['information'].str.contains('chip|fc|buffer', case=False, na=False) & ~reagent_flags['information'].str.contains('baseline', case=False, na=False)].copy()

    # Loops through the index range of the valid flags up to the penultimate record
    for i in range(len(valid_flags) - 1):

        # Extracts the current flag row
        row1 = valid_flags.iloc[i]

        # Extracts the immediate subsequent flag row
        row2 = valid_flags.iloc[i + 1]
        
        # Casts the information column of the first row to a string
        info1 = str(row1['information'])

        # Casts the information column of the second row to a string
        info2 = str(row2['information'])
        
        # Extracts the start time from the first row
        t_start = row1['time']

        # Extracts the end time from the second row
        t_end = row2['time']
        
        # Parses the label information for the first row into standard variables
        stage1_raw, stage1, type1, num1, is_buffer1 = parse_stage_label(info1)

        # Parses the label information for the second row into standard variables
        stage2_raw, stage2, type2, num2, is_buffer2 = parse_stage_label(info2)
        
        # Initialises a boolean flag to track transition validity
        is_valid_transition = False
        
        # Evaluates if the transition crosses between a buffer and a non-buffer stage
        if is_buffer1 != is_buffer2:

            # Flags the transition as valid
            is_valid_transition = True

        # Evaluates if the transition moves between two distinct buffer stages
        elif is_buffer1 and is_buffer2 and (stage1 != stage2):

            # Flags the transition as valid
            is_valid_transition = True

        # Evaluates if the transition moves between two distinct non-buffer reagent types
        elif not is_buffer1 and not is_buffer2 and (type1 != type2):
            
            # Evaluates if one stage type is dependent upon or nested within the other
            is_dependent = (type1 in stage2) or (type2 in stage1)
            
            # Calculates the total time duration between the two stages
            time_diff = t_end - t_start

            # Evaluates if the transition occurs too rapidly to sample properly
            is_too_fast = time_diff < 10.0
            
            # Checks if the transition is dependent or excessively rapid
            if is_dependent or is_too_fast:

                # Flags the transition as invalid
                is_valid_transition = False

            else:

                # Checks if specific numerical identifiers were extracted for both stages
                if num1 is not None and num2 is not None:

                    # Flags the transition valid only if the first number strictly precedes the second
                    is_valid_transition = num1 > num2

                else:

                    # Flags the transition valid as a fallback for missing identifiers
                    is_valid_transition = True
                    
        # Evaluates whether the logic checks determined the transition to be valid
        if is_valid_transition:
            
            # Calculates the total time duration of the transition phase
            time_diff = t_end - t_start
            
            # Calculates a staggered start time incorporating a proportional delay
            start_after_gap = t_start + (time_diff * 0.1)

            # Calculates the target baseline time resting deep within the valid plateau zone
            target_time = start_after_gap + (t_end - start_after_gap) * 0.8
            
            # Identifies the index of the closest physical timestamp in the raw sensorgram series
            closest_idx = np.argmin(np.abs(time - target_time))
            
            # Appends the calculated and mapped baseline definition to the tracking list
            baseline_records.append({'time': time[closest_idx], 'information': f'Baseline - {stage1_raw}'})
                    
    # Returns the compiled list of baseline sampling definitions
    return baseline_records

def update_immob_flags(flag_df, plot_data, file_path=None):
    '''Calculates and updates buffer, peak, baseline, and flat flags for a single sensorgram.

    Args:
        flag_df (pd.DataFrame): Flag dataframe for a single run.
        plot_data (pd.DataFrame): Sensorgram dataframe containing 'time' and channel columns.
        file_path (str | None, optional): Output CSV path to write the updated flags. Defaults to None.

    Returns:
        pd.DataFrame: Updated flag dataframe.
    '''

    # Creates a copy of the flag data to preserve the original dataframe
    flag_data = flag_df.copy()

    # Skips calculation if an initial flag already exists in the dataframe
    if flag_data['information'].astype(str).str.contains('initial', case=False, na=False).any():
        
        # Returns the original flag dataframe
        return flag_data

    # Extracts and sorts the time array from the sensorgram data
    time_array = plot_data['time'].sort_values().values

    # Extracts the primary channel signal array
    signal = plot_data['channel1'].values
    
    # Assigns the first recorded time as the start time
    start_time = time_array[0]

    # Assigns the maximum recorded time as the finish time
    finish_time = time_array.max()

    # Filters flags occurring within the valid sensorgram time range
    flag_data = flag_data[flag_data['time'] <= finish_time].copy()

    # Parses the flag dataframe to extract synthetic reagent and buffer events
    parsed_flags_df = extract_flags_from_pushes(flag_data)

    # Evaluates whether a plateau flag for the third buffer exists
    has_buffer_3_plateau = 'Buffer 3 - Plateau' in parsed_flags_df['information'].values

    # Evaluates whether the parsed flags dataframe contains data
    if not parsed_flags_df.empty:

        # Extracts the information string of the final recorded flag
        last_flag = parsed_flags_df.iloc[-1]['information']

        # Checks if the final flag lacks a plateau designation
        if not str(last_flag).endswith('Plateau'):

            # Evaluates if the final flag corresponds to a buffer phase
            if last_flag.startswith('Buffer'):

                # Extracts the base buffer name
                base_name = last_flag.split(' - ')[0]

                # Constructs the missing plateau flag string
                missing_plateau = f'{base_name} - Plateau'

            # Handles non-buffer final flags
            else:

                # Constructs the missing plateau flag string for the reagent
                missing_plateau = f'{last_flag} - Plateau'

            # Appends the constructed plateau flag to the parsed flags dataframe at the finish time
            parsed_flags_df = pd.concat([parsed_flags_df, pd.DataFrame([{'time': finish_time, 'information': missing_plateau}])], ignore_index=True)

    # Slices the time array to align with the differences array
    time_delta = time_array[:-1]

    # Calculates the step-to-step differences in the signal array
    delta = np.diff(signal)

    # Initialises an empty list to store the initial flat region records
    initial_records = []

    # Initialises an empty list to store the final flat region records
    final_records = []

    # Creates a boolean mask identifying flags related to the first buffer
    is_buffer_1 = parsed_flags_df['information'].str.startswith('Buffer 1', na=False)

    # Creates a boolean mask identifying flags that are not plateaus
    not_plateau = ~parsed_flags_df['information'].str.endswith('Plateau', na=False)

    # Filters the dataframe to isolate non-plateau flags for the first buffer
    b1_flags = parsed_flags_df[is_buffer_1 & not_plateau]
    
    # Evaluates whether any valid first buffer flags exist
    if not b1_flags.empty:

        # Extracts the time of the first buffer event
        b1_time = b1_flags['time'].iloc[0]

        # Finds the index of the closest physical timestamp in the time array
        b1_idx = np.abs(time_array - b1_time).argmin()
        
        # Calculates a proportional offset to define the initial flat search window
        initial_offset = (time_array[b1_idx] - start_time) * 0.75

        # Identifies indices falling within the calculated initial search window
        initial_idx = np.where((time_delta > (start_time + initial_offset)) & (time_delta < time_array[b1_idx]))[0]

        # Checks if any indices fall within the initial search window
        if initial_idx.size > 0:

            # Identifies the index with the minimum signal change to define the flattest point
            flat_idx = initial_idx[np.argmin(np.abs(delta[initial_idx]))]

            # Appends the identified initial flat record to the tracking list
            initial_records.append({'time': time_array[flat_idx], 'information': 'Initial'})
    
    # Evaluates whether the third buffer plateau exists to calculate the final flat region
    if has_buffer_3_plateau:

        # Filters the dataframe to isolate the third buffer plateau flag
        b3p_flags = parsed_flags_df[parsed_flags_df['information'] == 'Buffer 3 - Plateau']

        # Checks whether the filtered plateau flag dataframe contains data
        if not b3p_flags.empty:

            # Extracts the time of the third buffer plateau
            b3p_time = b3p_flags['time'].iloc[0]

            # Finds the index of the closest physical timestamp in the time array
            b3p_idx = np.abs(time_array - b3p_time).argmin()
            
            # Calculates a proportional offset to define the final flat search window
            final_offset = (finish_time - time_array[b3p_idx]) * 0.75

            # Identifies indices falling within the calculated final search window
            final_idx = np.where((time_delta > (time_array[b3p_idx] + final_offset)) & (time_delta < finish_time))[0]

            # Checks if any indices fall within the final search window
            if final_idx.size > 0:

                # Identifies the index with the minimum signal change to define the flattest point
                flat_idx = final_idx[np.argmin(np.abs(delta[final_idx]))]

                # Appends the identified final flat record to the tracking list
                final_records.append({'time': time_array[flat_idx], 'information': 'Final'})

            # Handles cases where the final search window is empty
            else:

                # Appends a fallback final record near the end of the time array
                final_records.append({'time': time_array[max(0, len(time_array) - 40)], 'information': 'Final'})
                
    # Initialises a list with the starting boundary record
    boundary_list = [{'time': start_time, 'information': 'Start'}]

    # Checks if the third buffer plateau exists to define the finish boundary
    if has_buffer_3_plateau:

        # Appends the finish boundary record to the list
        boundary_list.append({'time': finish_time, 'information': 'Finish'})
        
    # Converts the boundary records list into a pandas dataframe
    boundary_records = pd.DataFrame(boundary_list)

    # Concatenates all newly generated boundary, parsed, initial, and final records
    all_new_records = pd.concat([boundary_records, parsed_flags_df, pd.DataFrame(initial_records + final_records)], ignore_index=True)

    # Sorts the combined records chronologically and resets the index
    updated_df = all_new_records.sort_values('time').reset_index(drop=True)

    # Filters out non-reagent flags to prepare for baseline sampling calculation
    baseline_flags = updated_df[~updated_df['information'].str.contains('Concentration|Flat|Start|Finish', na=False)]

    # Calculates the baseline sampling positions based on the valid flags
    baseline_list = calculate_baseline_peaks(time_array, baseline_flags)
    
    # Appends the calculated baseline records to the updated dataframe
    updated_df = pd.concat([updated_df, pd.DataFrame(baseline_list)], ignore_index=True)

    # Deduplicates, chronologically sorts, and resets the index of the final dataframe
    updated_df = updated_df.drop_duplicates(subset=['time', 'information'], keep='first').sort_values('time').reset_index(drop=True)
    
    # Checks if a file path is provided for saving
    if file_path:

        # Creates the target directory structure if it does not already exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Saves the updated flags dataframe to a CSV file
        updated_df.to_csv(file_path, index=False)
        
    # Returns the fully updated flags dataframe
    return updated_df

def calculate_5s_averages(flag_df, plot_data, file_path=None):
    '''Calculates 5-second averages for flags dynamically across all channels for a single sensorgram.

    Args:
        flag_df (pd.DataFrame): Flag dataframe for a single run.
        plot_data (pd.DataFrame): Sensorgram dataframe containing 'time' and channel columns.
        file_path (str | None, optional): Output CSV path to write the averaged flags. Defaults to None.

    Returns:
        pd.DataFrame: Averaged flag dataframe.
    '''

    # Filters out concentration flags, creates a copy, and resets the index
    flag_data = flag_df[~flag_df['information'].astype(str).str.contains('Concentration', case=False, na=False)].reset_index(drop=True).copy()
    
    # Extracts the channel column names by skipping the time column
    channels = plot_data.columns[1:]

    # Extracts the raw time array from the sensorgram data
    sensor_time = plot_data['time'].values
    
    # Initialises a dictionary to store the calculated averages for each channel
    avg_data = {f'{ch}_avg': [] for ch in channels}

    # Initialises a list to track the start times of the averaging windows
    window_starts = []

    # Initialises a list to track the end times of the averaging windows
    window_ends = []

    # Initialises a list to track the descriptive labels of the averaging windows
    window_labels = []

    # Loops through each row in the filtered flags dataframe
    for idx, row in flag_data.iterrows():

        # Extracts the time of the current flag
        t_flag = row['time']

        # Casts the information string to lowercase for standardisation
        info = str(row['information']).lower()
        
        # Evaluates if the current flag marks the start of the measurement
        if 'start' in info:

            # Loops through each channel to assign the initial values
            for ch in channels: 

                # Appends the first recorded signal value to the averages dictionary
                avg_data[f'{ch}_avg'].append(plot_data[ch].values[0])

            # Appends the flag time to the window starts list
            window_starts.append(t_flag)

            # Appends the flag time to the window ends list
            window_ends.append(t_flag)

            # Appends the start label to the window labels list
            window_labels.append('Start')

            # Proceeds to the next flag row
            continue

        # Evaluates if the current flag marks the finish of the measurement
        elif 'finish' in info:

            # Loops through each channel to assign the final values
            for ch in channels: 

                # Appends the last recorded signal value to the averages dictionary
                avg_data[f'{ch}_avg'].append(plot_data[ch].values[-1])
            
            # Appends the final sensorgram time to the window starts list
            window_starts.append(sensor_time[-1])

            # Appends the final sensorgram time to the window ends list
            window_ends.append(sensor_time[-1])

            # Appends the finish label to the window labels list
            window_labels.append('Finish')

            # Proceeds to the next flag row
            continue
            
        # Determines the time of the previous flag or assigns negative infinity
        t_prev = flag_data.iloc[idx - 1]['time'] if idx > 0 else -np.inf

        # Determines the time of the next flag or assigns positive infinity
        t_next = flag_data.iloc[idx + 1]['time'] if idx < len(flag_data) - 1 else np.inf
        
        # Evaluates if the current flag represents a plateau
        is_plateau = any(w in info for w in ['plateau'])

        # Initialises the start time variable for the averaging window
        t_start = None

        # Initialises the end time variable for the averaging window
        t_end = None

        # Initialises the label variable for the averaging window
        w_label = ''
        
        # Checks if the current flag is a plateau to define its specific window
        if is_plateau:

            # Evaluates if a full five-second window fits without overlapping the previous flag
            if t_flag - 5 >= t_prev + 2:

                # Assigns the previous five seconds as the averaging window
                t_start, t_end = t_flag - 5, t_flag

                # Assigns the corresponding descriptive label
                w_label = 'Prev 5s'

            # Handles cases where the full window would overlap the previous flag
            else:

                # Truncates the window to avoid overlap
                t_start, t_end = t_prev + 2, t_flag

                # Assigns the fallback descriptive label
                w_label = 'Fallback'

        # Handles non-plateau flags
        else:

            # Evaluates if a full five-second forward window fits without overlapping the next flag
            if t_flag + 5 <= t_next:

                # Assigns the next five seconds as the averaging window
                t_start, t_end = t_flag, t_flag + 5

                # Assigns the corresponding descriptive label
                w_label = 'Next 5s'
        
            # Evaluates if a split window fits around the current flag
            elif (t_flag - 2.5 >= t_prev + 2) and (t_flag + 2.5 <= t_next):

                # Assigns a symmetrical window around the flag
                t_start, t_end = t_flag - 2.5, t_flag + 2.5

                # Assigns the corresponding descriptive label
                w_label = 'Split 2.5s'

            # Evaluates if a backward window fits without overlapping the previous flag
            elif t_flag - 5 >= t_prev + 2:

                # Assigns the previous five seconds as the averaging window
                t_start, t_end = t_flag - 5, t_flag

                # Assigns the corresponding descriptive label
                w_label = 'Prev 5s'

            # Handles cases where no standard window fits between adjacent flags
            else:

                # Assigns a truncated window constrained by adjacent flags
                t_start, t_end = t_prev + 2, t_next - 0.1

                # Assigns the corresponding gap fallback label
                w_label = 'Gap Fallback'
        
        # Checks if a valid time window was successfully calculated
        if t_start is not None and t_end > t_start:

            # Generates an interpolation grid within the defined window
            interp_grid = np.linspace(t_start, t_end, num=51)

            # Loops through each channel to calculate the interpolated average
            for ch in channels:

                # Appends the interpolated average to the tracking dictionary
                avg_data[f'{ch}_avg'].append(np.mean(np.interp(interp_grid, sensor_time, plot_data[ch].values)))

            # Appends the calculated start time to the tracking list
            window_starts.append(t_start)

            # Appends the calculated end time to the tracking list
            window_ends.append(t_end)

            # Appends the assigned label to the tracking list
            window_labels.append(w_label)

        # Handles cases where the calculated window is invalid
        else:

            # Loops through each channel to assign missing values
            for ch in channels: 

                # Appends a missing value to the averages dictionary
                avg_data[f'{ch}_avg'].append(np.nan)

            # Appends a missing value to the start times list
            window_starts.append(np.nan)

            # Appends a missing value to the end times list
            window_ends.append(np.nan)

            # Appends an empty label to the labels list
            window_labels.append('')

    # Loops through each calculated averages list
    for col, data_list in avg_data.items(): 

        # Assigns the averages list as a new column in the flag dataframe
        flag_data[col] = data_list

    # Assigns the start times list as a new column
    flag_data['Window_Start'] = window_starts

    # Assigns the end times list as a new column
    flag_data['Window_End'] = window_ends

    # Assigns the labels list as a new column
    flag_data['Window_Label'] = window_labels
    
    # Drops rows with missing averages, resets the index, and removes the raw time column
    flag_data = flag_data.dropna(subset=[f'{ch}_avg' for ch in channels]).reset_index(drop=True).drop('time', axis=1)
    
    # Checks if a file path is provided for saving
    if file_path:

        # Creates the target directory structure if it does not already exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Saves the averaged flags dataframe to a CSV file
        flag_data.to_csv(file_path, index=False)
        
    # Returns the final averaged flags dataframe
    return flag_data

def collapse_combined_reagents(event_df):
    '''Collapses consecutive identical stages (ignoring buffers) and normalises names while preserving original case.

    Args:
        event_df (pd.DataFrame): Dataframe containing event stages.

    Returns:
        pd.DataFrame: Collapsed event dataframe.
    '''

    # Creates a copy of the event dataframe to preserve the original data
    df = event_df.copy()

    # Removes trailing bracket information from stage names, excluding buffers
    df['stage'] = df['stage'].str.replace(r'(Baseline\s*\(\s*)(?!buffer)([\w/-]+)[^)]*(\))', r'\1\2\3', regex=True, flags=re.IGNORECASE)
    
    # Creates a boolean mask identifying non-buffer and non-baseline stages
    is_not_buffer = ~df['stage'].str.contains('buffer|baseline', case=False, na=False)
    
    # Extracts the primary base stage name using regular expressions
    extracted_stages = df['stage'].str.strip().str.extract(r'^([\w/-]+)', expand=False)
    
    # Conditionally assigns the extracted base stage or the stripped original stage
    df['base_stage'] = np.where(is_not_buffer, extracted_stages, df['stage'].str.strip())
    
    # Creates a boolean mask identifying Protein G related stages
    is_prg = df['stage'].str.contains('prg', case=False, na=False)

    # Creates a boolean mask identifying stages associated with the first chip
    is_chip1 = df['stage'].str.contains('chip 1', case=False, na=False)

    # Creates a boolean mask identifying plateau stages
    is_plateau = df['stage'].str.contains('plateau', case=False, na=False)
    
    # Creates a combined mask for primary Protein G stages excluding plateaus
    mask_prg_chip1 = is_prg & is_chip1 & ~is_plateau
    
    # Creates a combined mask for all Protein G plateaus
    mask_all_prg_plateaus = is_prg & is_plateau

    # Extracts the indices for all identified Protein G plateaus
    prg_plateau_indices = df[mask_all_prg_plateaus].index
    
    # Initialises a boolean series to track the final Protein G plateau
    mask_last_prg_plateau = pd.Series(False, index=df.index)

    # Checks whether any Protein G plateaus exist in the dataframe
    if not prg_plateau_indices.empty:

        # Flags the final identified Protein G plateau
        mask_last_prg_plateau[prg_plateau_indices[-1]] = True
        
    # Creates a combined mask for intermediate Protein G transitions
    mask_prg_inter = is_prg & ~mask_prg_chip1 & ~mask_last_prg_plateau
    
    # Standardises the base stage name for the primary Protein G phase
    df.loc[mask_prg_chip1, 'base_stage'] = 'PrG'

    # Standardises the base stage name for the final Protein G plateau
    df.loc[mask_last_prg_plateau, 'base_stage'] = 'PrG - Plateau'

    # Standardises the base stage name for intermediate Protein G steps
    df.loc[mask_prg_inter, 'base_stage'] = 'PrG_Intermediate'
    
    # Generates a lowercase comparison column for subsequent grouping
    df['compare_stage'] = df['base_stage'].str.lower()

    # Calculates sequential block identifiers to group consecutive identical stages
    df['block_id'] = (df['compare_stage'] != df['compare_stage'].shift()).cumsum()
    
    # Collapses consecutive stages by retaining the last record of each block
    collapsed = df.groupby('block_id').last().reset_index(drop=True)

    # Removes the intermediate Protein G steps and resets the index
    collapsed = collapsed[collapsed['base_stage'] != 'PrG_Intermediate'].reset_index(drop=True)
    
    # Initialises a dictionary to track the occurrence counts of each stage
    counts = {}

    # Initialises a list to construct unique stage names
    unique_stages = []

    # Loops through each original and comparison name tuple
    for orig_name, comp_name in zip(collapsed['base_stage'], collapsed['compare_stage']):

        # Increments the occurrence count for the current stage
        counts[comp_name] = counts.get(comp_name, 0) + 1

        # Appends a suffix to the stage name if it occurs multiple times
        unique_stages.append(orig_name if counts[comp_name] == 1 else f'{orig_name}_{counts[comp_name]}')
        
    # Overwrites the stage column with the generated unique names
    collapsed['stage'] = unique_stages
    
    # Drops the temporary comparison columns from the dataframe
    collapsed = collapsed.drop(columns=['base_stage', 'compare_stage'])
    
    # Returns the final collapsed dataframe
    return collapsed

def calculate_custom_immob_changes(event_df, channels):
    '''Calculates standard stage-to-stage differences and clean custom baseline metrics.

    Args:
        event_df (pd.DataFrame): Dataframe containing event changes.
        channels (list[str]): List of channels to calculate differences for.

    Returns:
        pd.DataFrame: Dataframe containing custom metrics.
    '''

    # Filters out the start stage and resets the index
    event_df = event_df[event_df['stage'] != 'Start'].copy().reset_index(drop=True)

    # Calculates stage-to-stage differences and stores them in a new dataframe
    df = add_stage_changes(event_df.copy(), channels)
    
    # Casts the stage column to lowercase for reliable matching
    stage_lower = event_df['stage'].str.lower()

    # Creates a mask to identify baseline, initial, and final stages
    mask = (stage_lower.str.startswith('baseline', na=False) | stage_lower.str.startswith(('initial', 'final'), na=False))

    # Filters the event dataframe to isolate the key reference stages
    baselines = event_df[mask].copy()
    
    # Extracts the chip identifier from the first record
    chip_id = event_df['chip_id'].iloc[0]
    
    # Initialises a list to collect the generated custom metric dictionaries
    custom_metrics = []
    
    # Evaluates whether sufficient reference stages exist to calculate shifts
    if len(baselines) > 1:

        # Loops through each reference stage starting from the second one
        for i in range(1, len(baselines)):
            
            # Extracts the previous and current reference stages
            prev_b, curr_b = baselines.iloc[i-1], baselines.iloc[i]

            # Constructs a descriptive name for the shift between stages
            stage_name = f"Net_Shift_{prev_b['stage']}_to_{curr_b['stage']}"
            
            # Initialises the metric dictionary with standard identifiers
            metric = {'stage': stage_name, 'time': curr_b['time'], 'chip_id': chip_id, 'base_stage': prev_b['stage']}

            # Adds the current stage signal values for each channel to the dictionary
            metric.update({ch: curr_b[ch] for ch in channels})

            # Calculates and adds the absolute shift for each channel
            metric.update({f'{ch}_change': curr_b[ch] - prev_b[ch] for ch in channels})

            # Appends the compiled metric dictionary to the tracking list
            custom_metrics.append(metric)

        # Extracts the final reference stage
        final_b = baselines.iloc[-1]

        # Extracts the initial reference stage
        initial_b = baselines.iloc[0]
        
        # Constructs the total shift metric dictionary
        metric = {'stage': 'Total_Shift_Initial_to_Final', 'time': final_b['time'], 'chip_id': chip_id, 'base_stage': initial_b['stage']}

        # Adds the final stage signal values for each channel to the dictionary
        metric.update({ch: final_b[ch] for ch in channels})

        # Calculates and adds the overall shift between the initial and final stages
        metric.update({f'{ch}_change': final_b[ch] - initial_b[ch] for ch in channels})

        # Appends the total shift metric to the tracking list
        custom_metrics.append(metric)

    # Concatenates the standard changes and custom metrics into a single dataframe
    return pd.concat([df, pd.DataFrame(custom_metrics)], ignore_index=True)

def normalise_by_initial_flag(df, channels, drop_start=True):
    '''Zero-bases channel values using the 'Initial' stage flag if present.
    Falls back to 'Start' or the first available row if 'Initial' is missing.
    'Initial' is NEVER dropped. 'Start' is dropped only if drop_start=True.

    Args:
        df (pd.DataFrame): Dataframe to normalise.
        channels (list[str]): List of channels to normalise.
        drop_start (bool, optional): Whether to drop the 'Start' stage. Defaults to True.

    Returns:
        pd.DataFrame: Normalised dataframe.
    '''

    # Creates a copy of the dataframe to preserve the original data
    df_norm = df.copy()
    
    # Checks if the initial stage flag exists in the dataframe
    if (df_norm['stage'] == 'Initial').any():

        # Assigns the index of the initial stage
        idx = df_norm[df_norm['stage'] == 'Initial'].index[0]

    # Checks if the start stage flag exists as a fallback
    elif (df_norm['stage'] == 'Start').any():

        # Assigns the index of the start stage
        idx = df_norm[df_norm['stage'] == 'Start'].index[0]

    # Handles cases where neither flag is present
    else:

        # Assigns the index of the first available row
        idx = df_norm.index[0]
        
    # Loops through each specified channel
    for ch in channels:

        # Extracts the baseline value for the current channel
        initial_val = df_norm.loc[idx, ch]

        # Subtracts the baseline value to zero-base the channel data
        df_norm[ch] = df_norm[ch] - initial_val
        
    # Evaluates whether to remove the start stage from the output
    if drop_start:

        # Filters out the start stage and resets the index
        df_norm = df_norm[df_norm['stage'] != 'Start'].reset_index(drop=True)
    
    # Returns the normalised dataframe
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

    # Extracts the time column name from the sensorgram dataframe
    time_col = sens_df.columns[0]

    # Extracts the raw time array for faster mathematical operations
    sens_time = sens_df[time_col].values

    # Initialises an empty list to store the extracted feature dictionaries
    features_list = []

    # Sorts the event flags chronologically and resets the index
    events = event_df.sort_values('time').reset_index(drop=True)
    
    # Loops through each event index up to the penultimate record
    for i in range(len(events) - 1):

        # Extracts the start time for the current stage
        start_time = events.loc[i, 'time']

        # Extracts the end time for the current stage from the subsequent event
        end_time = events.loc[i + 1, 'time']

        # Constructs a descriptive label for the stage transition
        stage_transition = f"{events.loc[i, 'stage']} -> {events.loc[i+1, 'stage']}"

        # Evaluates whether the stage duration is zero
        if start_time == end_time:

            # Extracts the exact data point corresponding to the zero-duration flag
            stage_data = sens_df[sens_time == start_time]

        # Handles stages with a positive duration
        else:

            # Creates a boolean mask to slice the sensorgram data
            mask = (sens_time >= start_time) & (sens_time < end_time)

            # Slices the sensorgram dataframe using the generated mask
            stage_data = sens_df[mask]
        
        # Initialises a dictionary to store the extracted features for the current stage
        row_features = {
            'chip_id': chip_id,
            'stage': stage_transition
        }

        # Extracts the time array specific to the sliced stage data
        t_arr = stage_data[time_col].values
        
        # Loops through each channel to extract statistical features
        for ch in channels:

            # Extracts the signal array for the current channel
            signal = stage_data[ch].values

            # Checks if the signal array contains sufficient data points for statistical analysis
            if len(signal) > 1:

                # Calculates and stores the standard deviation of the signal
                row_features[f'{ch}_std'] = np.std(signal)

                # Calculates and stores the peak-to-peak range of the signal
                row_features[f'{ch}_range'] = np.ptp(signal)
                
                # Shifts the time array to start at zero for linear fitting
                t_shifted = t_arr - t_arr[0]

                # Calculates the linear slope of the signal across the stage
                slope, _ = np.polyfit(t_shifted, signal, 1)

                # Stores the calculated slope
                row_features[f'{ch}_slope'] = slope
                
                # Evaluates if the signal variance is too low to calculate skewness
                if np.std(signal) < 1e-8:

                    # Assigns a default skewness of zero to prevent calculation errors
                    row_features[f'{ch}_skew'] = 0.0

                # Handles signals with sufficient variance
                else:

                    # Calculates and stores the skewness of the signal
                    row_features[f'{ch}_skew'] = skew(signal)
                
            # Evaluates if the signal array contains exactly one data point
            elif len(signal) == 1:

                # Assigns zero to all statistical features as a fallback
                row_features.update({
                    f'{ch}_std': 0,
                    f'{ch}_range': 0,
                    f'{ch}_slope': 0, 
                    f'{ch}_skew': 0
                })

            # Handles cases where the signal array is entirely empty
            else:

                # Assigns zero to all statistical features as a fallback
                row_features.update({
                    f'{ch}_std': 0,
                    f'{ch}_range': 0,
                    f'{ch}_slope': 0, 
                    f'{ch}_skew': 0
                })
                
        # Appends the completed features dictionary to the tracking list
        features_list.append(row_features)
        
    # Converts the features list into a pandas dataframe and returns it
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

    # Evaluates and returns the five-parameter logistic model prediction
    return top + (bottom - top) / (1 + (x / ec50)**(hill))**asym

def four_pl(x, bottom, top, ec50, hill):
    '''Evaluates the four-parameter logistic model for standard-curve fitting.

    Args:
        x (array-like): Concentration values.
        bottom, top, ec50, hill (float): Logistic-model coefficients.

    Returns:
        array-like: Predicted signal values.
    '''

    # Evaluates and returns the four-parameter logistic model prediction
    return top + (bottom - top) / (1 + (x / ec50)**(hill))

def fit_curve(data):
    '''Fits candidate logistic models and selects the best standard-curve result.

    Args:
        data (pd.DataFrame): Concentration and response measurements.

    Returns:
        tuple: Fitted curve values and the optimized parameters.
    '''

    # Extracts the concentration and response arrays from the data
    x, y = data['x'], data['y']

    # Generates logarithmically spaced concentration points for the fitted curve
    xfit = np.logspace(np.log10(min(x)), np.log10(max(x)), 1000)

    # Evaluates whether the data requires a five-parameter logistic model
    if data['fit_model'] == '5PL':

        # Defines the parameter bounds for the five-parameter curve fit
        bounds = ([-np.inf, -np.inf, min(x)/100, -10, 0.01], [np.inf, np.inf, max(x)*100, 10, 10])

        # Establishes the initial parameter guesses for the five-parameter curve fit
        p0 = [min(y), max(y), np.median(x), 1.0, 1.0]

        # Executes the curve fitting algorithm to find optimal five-parameter coefficients
        popt, _ = curve_fit(five_pl, x, y, p0=p0, bounds=bounds, maxfev=50000)

        # Calculates the predicted response values using the optimised five-parameter model
        yfit = five_pl(xfit, *popt)

    # Handles cases where a four-parameter logistic model is required
    else:

        # Defines the parameter bounds for the four-parameter curve fit
        bounds = ([-np.inf, -np.inf, min(x)/100, -10], [np.inf, np.inf, max(x)*100, 10])

        # Establishes the initial parameter guesses for the four-parameter curve fit
        p0 = [min(y), max(y), np.median(x), 1.0]

        # Executes the curve fitting algorithm to find optimal four-parameter coefficients
        popt, _ = curve_fit(four_pl, x, y, p0=p0, bounds=bounds, maxfev=50000)

        # Calculates the predicted response values using the optimised four-parameter model
        yfit = four_pl(xfit, *popt)
        
    # Returns the generated curve coordinates alongside the optimal parameters
    return xfit, yfit, popt

# UI Display Helpers
class collapsible_output:

    def __init__(self, title):
        '''Initialises a collapsible output section.

        Args:
            title (str): Text displayed on the section toggle button.
        '''

        # Assigns the provided title to the class instance
        self.title = title
        
        # Configures the output layout and styling for the collapsible section
        self.out = widgets.Output(layout=widgets.Layout(display='none', margin='10px 0 10px 15px', overflow='auto'))

        # Applies a custom CSS class to the output widget for targeted styling
        self.out.add_class('custom-clean-output')
        
        # Defines the button text displayed when the section is expanded
        self.expanded_label = f'[v] {self.title}'

        # Defines the button text displayed when the section is collapsed
        self.collapsed_label = f'[>] {self.title}'
        
        # Initialises the toggle button widget with specific layout parameters
        self.btn = widgets.Button(description=self.collapsed_label, layout=widgets.Layout(width='auto', border='none', padding='0', margin='0'))

        # Sets the background colour of the toggle button to transparent
        self.btn.style.button_color = 'transparent'

        # Sets the text colour of the toggle button to black
        self.btn.style.text_color = 'black' 

        # Applies bold formatting to the toggle button text
        self.btn.style.font_weight = 'bold'
        
        # Binds the toggle method to the button click event
        self.btn.on_click(self.toggle)

        # Groups the toggle button and output area into a vertical box container
        self.container = widgets.VBox([self.btn, self.out])

    def toggle(self, *args):
        '''Shows or hides the output section when its toggle button is clicked.'''

        # Evaluates whether the output section is currently hidden
        if self.out.layout.display == 'none':

            # Displays the hidden output section
            self.out.layout.display = 'block'

            # Updates the button text to the expanded label
            self.btn.description = self.expanded_label 

        # Handles cases where the output section is currently visible
        else:

            # Hides the visible output section
            self.out.layout.display = 'none'

            # Updates the button text to the collapsed label
            self.btn.description = self.collapsed_label 

    def __enter__(self):
        '''Redirects displayed output into the collapsible section.

        Returns:
            collapsible_output: The active output container.
        '''
        
        # Renders the collapsible container widget in the notebook
        display(self.container)
        
        # Injects custom CSS styling to format the collapsible output elements
        display(HTML('''
        <style>
            .custom-clean-output, 
            .custom-clean-output .jp-RenderedText, 
            .custom-clean-output pre, 
            .custom-clean-output .output_area {
                color: black !important;
            }
            
            .custom-clean-output .output_area {
                display: block !important; 
                overflow-x: auto !important;
                max-width: 100% !important;
                width: 100% !important;
            }
            
            .custom-clean-output .dataframe {
                max-width: none !important;
                display: block !important;
                overflow-x: auto !important;
                white-space: nowrap !important;
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

            .jupyter-widgets.jupyter-button {
                text-align: left !important;
                justify-content: flex-start !important;
            }   
        </style>
        '''))
        
        # Enters the context manager of the output widget to capture printed content
        self.out.__enter__()

        # Returns the active collapsible output instance
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        '''Restores normal output handling when the section context closes.'''

        # Exits the output widget context manager to restore normal rendering
        self.out.__exit__(exc_type, exc_value, traceback)

        # Returns False to allow exceptions to propagate normally
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
    
    # Checks whether the provided data list is empty
    if not data_list: 

        # Returns None if no data is available for summarisation
        return None
    
    # Concatenates the list of dataframes into a single combined dataframe
    df = pd.concat(data_list, ignore_index=True)

    # Evaluates whether a specific column is provided to filter missing values
    if drop_na_col:

        # Drops rows containing missing values in the specified column and copies the dataframe
        df = df.dropna(subset=[drop_na_col]).copy()
        
    # Applies ordered categorical typing to the stage column to preserve sequence
    df['stage'] = pd.Categorical(df['stage'], categories=pd.unique(df['stage']), ordered=True)

    # Builds the aggregated statistical summary for the specified measurement columns
    summary = build_stage_summary(df, 'stage', val_cols)

    # Displays the generated summary table within a collapsible output section
    with collapsible_output(title): 

        # Renders the summary dataframe
        display(summary)
        
    # Returns the combined dataframe used for summarisation
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

    # Initialises the quality check collapsible output widget
    qc_output = collapsible_output('Quality Checks')

    # Executes and displays the data quality checks within the collapsible widget
    with qc_output:

        # Runs the quality checks on the PEL sensorgram data
        check_data_quality(sensorgram_dfs, 'PEL')

    # Initialises the sensorgram plots collapsible output widget
    plot_output = collapsible_output('Sensorgram Plots')

    # Executes and displays the sensorgram plots within the collapsible widget
    with plot_output:

        # Generates the raw sensorgram plots for the PEL data
        plot_sensorgrams(sensorgram_dfs, 'PEL')

    # Extracts the unique time columns from the flags dataframe to define stage labels
    time_cols = pd.unique(flags_PEL_df.columns)[1:]

    # Initialises an empty list to store absolute event dataframes
    all_events_PEL = []

    # Initialises an empty list to store stage change dataframes
    all_changes_PEL = []

    # Initialises an empty list to store intra-stage feature dataframes
    all_intrastage_PEL = []

    # Initialises an empty list to store normalised event dataframes
    all_events_PEL_norm = []

    # Initialises the sensorgram flag plots collapsible output widget
    flag_output = collapsible_output('Sensorgram Flag Plots')

    # Executes and displays the sensorgram flag plots within the collapsible widget
    with flag_output:

        # Loops through each row in the PEL upload metadata dataframe
        for _, upload_data in pel_upload_df.iterrows():

            # Extracts the chip identifier for the current upload row
            chip_id = upload_data['chip_id']

            # Evaluates whether the mapped sensorgram exists in the provided dictionary
            if upload_data['sensorgram'] in sensorgram_dfs.keys():

                # Assigns the matched sensorgram dataframe
                sens = sensorgram_dfs[upload_data['sensorgram']]

            # Handles cases where the mapped sensorgram is missing
            else:

                # Skips to the next upload row
                continue

            # Retrieves the flag data row corresponding to the current flags identifier
            flag_data = flags_PEL_df[flags_PEL_df['flags_id'] == upload_data['flags_id']].iloc[0]

            # Extracts the timestamp values from the flag data row
            times = flag_data[time_cols].values

            # Assigns the time column names as the stage labels
            labels = time_cols

            # Plots the sensorgram with vertical flag overlays
            plot_flags_on_sensorgrams(sens, times, labels, title=f'PEL Sensorgram for {chip_id}')

            # Extracts the channel column names by skipping the time column
            channels = sens.columns.tolist()[1:]
            
            # Initialises a dictionary to compile the stage and time events for the chip
            event_dict = {
                'stage': labels,
                'time': times,
                'chip_id': chip_id
            }
            
            # Loops through each channel to interpolate signal values at the flag times
            for ch in channels:

                # Calculates and assigns the interpolated signal values to the event dictionary
                event_dict[ch] = np.interp(times, sens.iloc[:,0], sens[ch])

            # Converts the compiled event dictionary into a pandas dataframe
            event_df = pd.DataFrame(event_dict)

            # Appends the generated event dataframe to the collective list
            all_events_PEL.append(event_df)

            # Prints a header indicating the display of raw metrics for the current chip
            print(f'Raw Metrics for Chip {chip_id}')

            # Renders the raw event dataframe
            display(event_df)

            # Creates a filtered copy of the event dataframe, removing the start stage and chip identifier
            change_df = event_df[event_df['stage'] != 'Start'].copy().drop(columns=['chip_id']).reset_index(drop=True)

            # Calculates the stage-to-stage differences and updates the change dataframe
            change_df = add_stage_changes(change_df, channels)

            # Identifies and extracts the initial stage row from the event dataframe
            initial_row = event_df[event_df['stage'] == 'Initial']

            # Identifies and extracts the final stage row from the event dataframe
            final_row = event_df[event_df['stage'] == 'Final']

            # Determines the appropriate timestamp for the total shift summary row
            summary_time = final_row['time'].iloc[0] if not final_row.empty else event_df['time'].iloc[-1]

            # Initialises the total shift summary row dictionary
            summary_row = {
                'stage': 'Total_Shift_Initial_to_Final',
                'time': summary_time
            }

            # Creates a copy of the event dataframe for normalisation purposes
            event_norm_df = event_df.copy()
            
            # Creates a copy of the raw sensorgram dataframe for normalisation purposes
            sens_norm = sens.copy()

            # Loops through each channel to calculate absolute shifts and normalise signals
            for ch in channels:

                # Extracts the initial reference value for the current channel
                val_initial = initial_row[ch].iloc[0] if not initial_row.empty else event_df[ch].iloc[0]

                # Extracts the final reference value for the current channel
                val_final = final_row[ch].iloc[0] if not final_row.empty else event_df[ch].iloc[-1]

                # Stores the initial and final values as a list in the summary row
                summary_row[ch] = [val_initial, val_final]

                # Calculates and stores the total absolute shift for the channel
                summary_row[f'{ch}_change'] = val_final - val_initial

                # Normalises the event dataframe channel signals against the initial value
                event_norm_df[ch] = event_norm_df[ch] - val_initial

                # Normalises the raw sensorgram channel signals against the initial value
                sens_norm[ch] = sens_norm[ch] - val_initial

            # Appends the completed total shift summary row to the change dataframe
            change_df = pd.concat([change_df, pd.DataFrame([summary_row])], ignore_index=True)

            # Reassigns the chip identifier to the updated change dataframe
            change_df['chip_id'] = chip_id

            # Appends the updated change dataframe to the collective list
            all_changes_PEL.append(change_df)
            
            # Drops the start stage row from the normalised event dataframe
            event_norm_df = event_norm_df[event_norm_df['stage'] != 'Start']
            
            # Appends the normalised event dataframe to the collective list
            all_events_PEL_norm.append(event_norm_df)

            # Prints a header indicating the display of normalised metrics for the current chip
            print(f'Normalised Metrics for Chip {chip_id}')

            # Renders the normalised event dataframe
            display(event_norm_df)
            
            # Prints a header indicating the display of stage changes for the current chip
            print(f'Stage Changes for Chip {chip_id}')

            # Renders the change dataframe
            display(change_df)

            # Extracts the intra-stage statistical features using the normalised data
            intra_df = extract_intrastage_features(sens_norm, event_norm_df, channels, chip_id)

            # Appends the extracted intra-stage dataframe to the collective list
            all_intrastage_PEL.append(intra_df)

            # Renders the intra-stage feature dataframe
            display(intra_df)

    # Concatenates all individual event dataframes into a single consolidated dataframe
    all_events_PEL_df = pd.concat(all_events_PEL, ignore_index=True)
    
    # Applies ordered categorical typing to the consolidated stage column
    all_events_PEL_df['stage'] = pd.Categorical(all_events_PEL_df['stage'], categories=time_cols, ordered=True)

    # Builds the statistical metrics summary for the consolidated absolute events
    summary_PEL = build_stage_summary(all_events_PEL_df, 'stage', channels)

    # Initialises the PEL stage metrics collapsible output widget
    metrics_output = collapsible_output('PEL Stage Metrics Summary')

    # Displays the PEL stage metrics summary within the collapsible widget
    with metrics_output:

        # Renders the absolute metrics summary dataframe
        display(summary_PEL)

    # Concatenates all individual normalised event dataframes into a single consolidated dataframe
    all_events_PEL_norm_df = pd.concat(all_events_PEL_norm, ignore_index=True)
    
    # Filters out the start stage from the expected time columns
    norm_time_cols = [c for c in time_cols if c not in ['Start']]
    
    # Applies ordered categorical typing to the normalised stage column
    all_events_PEL_norm_df['stage'] = pd.Categorical(all_events_PEL_norm_df['stage'], categories=norm_time_cols, ordered=True)
    
    # Builds the statistical metrics summary for the consolidated normalised events
    summary_PEL_norm = build_stage_summary(all_events_PEL_norm_df, 'stage', channels)

    # Initialises the PEL stage normalised metrics collapsible output widget
    norm_metrics_output = collapsible_output('PEL Stage Normalised Metrics Summary')

    # Displays the PEL stage normalised metrics summary within the collapsible widget
    with norm_metrics_output:

        # Renders the normalised metrics summary dataframe
        display(summary_PEL_norm)

    # Concatenates all individual change dataframes into a single consolidated dataframe
    all_changes_PEL_df = pd.concat(all_changes_PEL, ignore_index=True)
    
    # Creates a cleaned copy of the consolidated change dataframe by dropping incomplete transition rows
    change_PEL_df = all_changes_PEL_df.dropna(subset=[f'{ch}_change' for ch in channels]).copy()

    # Extracts the unique sequential stage transitions present in the change dataframe
    stage_cols_pel = pd.unique(change_PEL_df['stage'])

    # Applies ordered categorical typing to the change stage column
    change_PEL_df['stage'] = pd.Categorical(change_PEL_df['stage'], categories=stage_cols_pel, ordered=True)
    
    # Builds the statistical metrics summary for the consolidated stage changes
    change_summary_PEL = build_stage_summary(change_PEL_df, 'stage', [f'{ch}_change' for ch in channels])

    # Initialises the PEL stage change metrics collapsible output widget
    change_output = collapsible_output('PEL Stage Change Metrics Summary')

    # Displays the PEL stage change metrics summary within the collapsible widget
    with change_output:

        # Renders the change metrics summary dataframe
        display(change_summary_PEL)

    # Loops through each channel to generate specific statistical summaries
    for ch in channels:

        # Initialises the statistical plots collapsible output widget for the current channel
        stats_output = collapsible_output(f'Statistical Plots ({ch})')

        # Executes and displays the statistical plots within the collapsible widget
        with stats_output:

            # Generates the statistical plots for the absolute signals
            plot_all_statistics(all_events_PEL_df, 'stage', ch, ylabel=f'{ch} Signal', title_prefix=f'Absolute - {ch}')

            # Generates the statistical plots for the normalised signals
            plot_all_statistics(all_events_PEL_norm_df, 'stage', ch, ylabel=f'{ch} Normalised Signal', title_prefix=f'Normalised - {ch}')

            # Generates the statistical plots for the delta signals
            plot_all_statistics(change_PEL_df, 'stage', f'{ch}_change', ylabel=f'Delta {ch} Signal', title_prefix=f'Delta - {ch}')

    # Concatenates all individual intra-stage feature dataframes into a single consolidated dataframe
    pel_intra_df = pd.concat(all_intrastage_PEL, ignore_index=True)            

    # Returns the consolidated event, change, intra-stage, and normalised dataframes
    return all_events_PEL_df, change_PEL_df, pel_intra_df, all_events_PEL_norm_df

def run_immob_analysis(immob_df, immobilisation_dfs, flags_dfs, averaged_flags_dfs):
    '''Runs the complete immobilisation analysis workflow across all split types.

    Args:
        immob_df (pd.DataFrame): Immobilisation metadata.
        immobilisation_dfs (dict): RefPoly4 data keyed by source file name.
        flags_dfs (dict): Immediate flag data keyed by source file name.
        averaged_flags_dfs (dict): Five-second averaged flag data.

    Returns:
        tuple: Event, change, intra-stage and normalised datasets for all immobilisation splits.
    '''

    # Initialises the quality check collapsible output widget
    qc_output = collapsible_output('Quality Checks')

    # Executes and displays the data quality checks within the collapsible widget
    with qc_output:

        # Runs the quality checks on the immobilisation sensorgram data
        check_data_quality(immobilisation_dfs, 'Immobilisation')

    # Initialises the sensorgram plots collapsible output widget
    sensor_output = collapsible_output('Sensorgram Plots')

    # Executes and displays the sensorgram plots within the collapsible widget
    with sensor_output:

        # Generates the raw sensorgram plots for the immobilisation data
        plot_sensorgrams(immobilisation_dfs, 'Immobilisation')

    # Initialises empty lists to store dataframes for immediate evaluations
    all_events_imp, all_changes_imp, all_intra_imp = [], [], []

    # Initialises empty lists to store dataframes for combined reagent evaluations
    all_events_comb, all_changes_comb, all_intra_comb = [], [], []

    # Initialises empty lists to store dataframes for averaged immediate evaluations
    all_events_avg_imp, all_changes_avg_imp, all_intra_avg_imp = [], [], []

    # Initialises empty lists to store dataframes for averaged combined evaluations
    all_events_avg_comb, all_changes_avg_comb, all_intra_avg_comb = [], [], []

    # Initialises empty lists to store normalised immediate dataframes
    all_events_imp_norm, all_events_comb_norm = [], []

    # Initialises empty lists to store normalised averaged dataframes
    all_events_avg_imp_norm, all_events_avg_comb_norm = [], []

    # Loops through each row in the immobilisation metadata dataframe
    for _, immob_data in immob_df.iterrows():

        # Extracts the chip identifier for the current row
        chip_id = immob_data['chip_id']

        # Evaluates whether the mapped sensorgram exists in the provided dictionary
        if immob_data['refPoly4'] in immobilisation_dfs.keys():

            # Assigns the matched sensorgram dataframe
            sens = immobilisation_dfs[immob_data['refPoly4']]

        # Handles cases where the mapped sensorgram is missing
        else:

            # Skips to the next metadata row
            continue
        
        # Extracts the channel column names by skipping the time column
        channels = sens.columns.tolist()[1:]
        
        # Evaluates whether the mapped flag data exists in the provided dictionary
        if immob_data['amfFlags'] not in flags_dfs: 

            # Skips to the next metadata row
            continue

        # Retrieves the flag data row corresponding to the current flags identifier
        flag_data = flags_dfs[immob_data['amfFlags']]

        # Filters out flag data rows containing concentration labels
        flag_data = flag_data[~flag_data.iloc[:, 1].str.contains('Concentration', case=False, na=False)]

        # Extracts the timestamps and labels from the filtered flag data
        times, labels = flag_data['time'].tolist(), flag_data['information'].tolist()

        # Initialises a dictionary to compile the stage and time events for the chip
        event_dict = {'stage': labels, 'time': times, 'chip_id': chip_id}

        # Loops through each channel to interpolate signal values at the flag times
        for ch in channels: 

            # Calculates and assigns the interpolated signal values to the event dictionary
            event_dict[ch] = np.interp(times, sens.iloc[:, 0], sens[ch])
        
        # Converts the compiled event dictionary into a pandas dataframe
        event_df = pd.DataFrame(event_dict)

        # Appends the generated immediate event dataframe to the collective list
        all_events_imp.append(event_df)

        # Calculates stage changes and appends the dataframe to the collective list
        all_changes_imp.append(calculate_custom_immob_changes(event_df, channels))

        # Identifies and extracts the initial stage row from the event dataframe
        initial_row = event_df[event_df['stage'] == 'Initial']
        
        # Evaluates whether the initial stage row exists
        if not initial_row.empty:

            # Assigns the initial time to establish a normalisation baseline
            initial_time = initial_row['time'].iloc[0]

        # Handles cases where the initial stage row is absent
        else:

            # Filters out the start stage from the event dataframe
            non_start_df = event_df[event_df['stage'] != 'Start']

            # Assigns the first available non-start timestamp or falls back to the absolute first timestamp
            initial_time = non_start_df['time'].iloc[0] if not non_start_df.empty else event_df['time'].iloc[0]

            # Reassigns the initial row matching the selected fallback timestamp
            initial_row = event_df[event_df['time'] == initial_time]

        # Creates a copy of the raw sensorgram dataframe starting from the initial time
        sens_norm = sens[sens.iloc[:, 0] >= initial_time].copy()
        
        # Loops through each channel to normalise the sensorgram
        for ch in channels:
            
            # Extracts the initial reference value for the current channel
            val_initial = initial_row[ch].iloc[0]

            # Zero-bases the sensorgram normalisation dataframe using the initial value
            sens_norm[ch] = sens_norm[ch] - val_initial

        # Normalises the immediate event dataframe using the initial stage flag
        event_norm_df = normalise_by_initial_flag(event_df, channels)

        # Drops the start stage row from the normalised immediate event dataframe
        event_norm_df = event_norm_df[event_norm_df['stage'] != 'Start']

        # Appends the normalised immediate event dataframe to the collective list
        all_events_imp_norm.append(event_norm_df)

        # Extracts intra-stage features and appends the dataframe to the collective list
        all_intra_imp.append(extract_intrastage_features(sens_norm, event_norm_df, channels, chip_id))

        # Collapses consecutive combined reagents to create a unified event dataframe
        comb_df = collapse_combined_reagents(event_df)

        # Appends the combined event dataframe to the collective list
        all_events_comb.append(comb_df)

        # Calculates stage changes for the combined events and appends the dataframe
        all_changes_comb.append(calculate_custom_immob_changes(comb_df, channels))

        # Normalises the combined event dataframe using the initial stage flag
        comb_norm_df = normalise_by_initial_flag(comb_df, channels)

        # Drops the start stage row from the normalised combined event dataframe
        comb_norm_df = comb_norm_df[comb_norm_df['stage'] != 'Start']

        # Appends the normalised combined event dataframe to the collective list
        all_events_comb_norm.append(comb_norm_df)

        # Extracts intra-stage features for the combined events and appends the dataframe
        all_intra_comb.append(extract_intrastage_features(sens_norm, comb_norm_df, channels, chip_id))

        # Evaluates whether averaged flag data exists for the current chip
        if averaged_flags_dfs and immob_data['amfFlags'] in averaged_flags_dfs:

            # Creates a copy of the averaged flags dataframe to preserve the original data
            avg_flags = averaged_flags_dfs[immob_data['amfFlags']].copy()
            
            # Constructs a dictionary for the five-second averaged events
            avg_dict = {
                'stage': avg_flags['information'].tolist(),
                'time': ((avg_flags['Window_Start'].astype(float) + avg_flags['Window_End'].astype(float)) / 2).tolist(),
                'chip_id': chip_id
            }

            # Loops through each channel to assign the averaged signal values
            for ch in channels: 

                # Appends the channel averages to the dictionary
                avg_dict[ch] = avg_flags[f'{ch}_avg'].tolist()
                
            # Converts the compiled averaged event dictionary into a pandas dataframe
            avg_df = pd.DataFrame(avg_dict)

            # Appends the generated averaged immediate event dataframe to the collective list
            all_events_avg_imp.append(avg_df)

            # Calculates stage changes for the averaged events and appends the dataframe
            all_changes_avg_imp.append(calculate_custom_immob_changes(avg_df, channels))

            # Extracts intra-stage features for the averaged events and appends the dataframe
            all_intra_avg_imp.append(extract_intrastage_features(sens, avg_df, channels, chip_id))

            # Normalises the averaged event dataframe and appends it to the collective list
            all_events_avg_imp_norm.append(normalise_by_initial_flag(avg_df, channels))

            # Collapses consecutive combined reagents from the averaged event dataframe
            avg_comb_df = collapse_combined_reagents(avg_df)

            # Appends the combined averaged event dataframe to the collective list
            all_events_avg_comb.append(avg_comb_df)

            # Calculates stage changes for the combined averaged events and appends the dataframe
            all_changes_avg_comb.append(calculate_custom_immob_changes(avg_comb_df, channels))

            # Extracts intra-stage features for the combined averaged events and appends the dataframe
            all_intra_avg_comb.append(extract_intrastage_features(sens, avg_comb_df, channels, chip_id))

            # Normalises the combined averaged event dataframe and appends it to the collective list
            all_events_avg_comb_norm.append(normalise_by_initial_flag(avg_comb_df, channels))

    # Initialises the sensorgram flag plots collapsible output widget
    flag_output = collapsible_output('Sensorgram Flag Plots')

    # Executes and displays the sensorgram flag plots within the collapsible widget
    with flag_output:

        # Initialises an index counter to access the correct stored dataframes
        i = 0

        # Loops through each row in the immobilisation metadata dataframe
        for _, immob_data in immob_df.iterrows():

            # Evaluates whether the mapped flag data exists in the provided dictionary
            if immob_data['amfFlags'] not in flags_dfs.keys(): 

                # Skips to the next metadata row
                continue

            # Assigns the matched sensorgram dataframe
            sens = immobilisation_dfs[immob_data['refPoly4']]

            # Assigns the matched flag dataframe
            flag_data = flags_dfs[immob_data['amfFlags']]

            # Filters out flag data rows containing concentration labels
            flag_data = flag_data[~flag_data.iloc[:, 1].str.contains('Concentration', case=False, na=False)]
            
            # Plots the flag markers overlaid on the sensorgrams with specific color coding
            plot_flags_on_sensorgrams(sens, flag_data['time'].tolist(), flag_data['information'].tolist(), title=f"Immob Sensorgram for {immob_data['chip_id']}", colours=['r' if 'Buffer' in str(lbl) else 'k' for lbl in flag_data['information']])

            # Prints a header indicating the display of metrics for the current chip
            print(f"Chip {immob_data['chip_id']} Metrics")
            
            # Renders the absolute immediate event dataframe
            display(all_events_imp[i])

            # Renders the normalised immediate event dataframe
            display(all_events_imp_norm[i])

            # Renders the immediate stage changes dataframe
            display(all_changes_imp[i])

            # Renders the immediate intra-stage feature dataframe
            display(all_intra_imp[i])

            # Renders the absolute combined event dataframe
            display(all_events_comb[i])

            # Renders the normalised combined event dataframe
            display(all_events_comb_norm[i])

            # Renders the combined stage changes dataframe
            display(all_changes_comb[i])

            # Renders the combined intra-stage feature dataframe
            display(all_intra_comb[i])

            # Increments the index counter for the next iteration
            i += 1

    # Evaluates whether averaged flag dataframes exist to process
    if averaged_flags_dfs:

        # Initialises the averaged window plots collapsible output widget
        avg_output = collapsible_output('Averaged Window Plots')

        # Executes and displays the averaged window plots within the collapsible widget
        with avg_output:

            # Initialises an index counter to access the correct stored dataframes
            i = 0

            # Loops through each row in the immobilisation metadata dataframe
            for _, immob_data in immob_df.iterrows():

                # Evaluates whether the mapped averaged flag data exists in the provided dictionary
                if immob_data['amfFlags'] not in averaged_flags_dfs:

                    # Skips to the next metadata row
                    continue

                # Assigns the matched sensorgram dataframe
                sens = immobilisation_dfs[immob_data['refPoly4']]

                # Assigns the matched averaged flag dataframe
                avg_flags = averaged_flags_dfs[immob_data['amfFlags']]
                
                # Constructs and displays the sensorgram plots with shaded five-second averaged windows
                plot_5s_averaged_windows(sens, avg_flags, immob_data['chip_id'])

                # Prints a header indicating the display of averaged metrics for the current chip
                print(f"Chip {immob_data['chip_id']} (5s Avg) Metrics")
                
                # Renders the absolute averaged immediate event dataframe
                display(all_events_avg_imp[i])

                # Renders the normalised averaged immediate event dataframe
                display(all_events_avg_imp_norm[i])

                # Renders the averaged immediate stage changes dataframe
                display(all_changes_avg_imp[i])

                # Renders the absolute averaged combined event dataframe
                display(all_events_avg_comb[i])

                # Renders the normalised averaged combined event dataframe
                display(all_events_avg_comb_norm[i])

                # Renders the averaged combined stage changes dataframe
                display(all_changes_avg_comb[i])

                # Increments the index counter for the next iteration
                i += 1

    # Formats a list of change column names for subsequent metric generations
    change_cols = [f'{ch}_change' for ch in channels]

    # Generates and displays the immediate stage metrics summary
    imp_df = generate_and_display_summary(all_events_imp, channels, 'Immediate: Metrics by Stage')

    # Generates and displays the normalised immediate stage metrics summary
    imp_norm_df = generate_and_display_summary(all_events_imp_norm, channels, 'Immediate: Normalised Metrics by Stage')

    # Generates and displays the immediate stage change metrics summary
    imp_change_df = generate_and_display_summary(all_changes_imp, change_cols, 'Immediate: Stage Changes', drop_na_col=change_cols[0])
    
    # Generates and displays the combined stage metrics summary
    comb_df = generate_and_display_summary(all_events_comb, channels, 'Combined Reagents: Metrics by Stage')

    # Generates and displays the normalised combined stage metrics summary
    comb_norm_df = generate_and_display_summary(all_events_comb_norm, channels, 'Combined Reagents: Normalised Metrics by Stage')

    # Generates and displays the combined stage change metrics summary
    comb_change_df = generate_and_display_summary(all_changes_comb, change_cols, 'Combined Reagents: Stage Changes', drop_na_col=change_cols[0])

    # Concatenates all immediate intra-stage feature dataframes if populated, else assigns an empty dataframe
    imp_intra_df = pd.concat(all_intra_imp, ignore_index=True) if all_intra_imp else pd.DataFrame()

    # Concatenates all combined intra-stage feature dataframes if populated, else assigns an empty dataframe
    comb_intra_df = pd.concat(all_intra_comb, ignore_index=True) if all_intra_comb else pd.DataFrame()

    # Initialises an empty dataframe for averaged immediate intra-stage features
    avg_imp_intra_df = pd.DataFrame()

    # Initialises an empty dataframe for averaged combined intra-stage features
    avg_comb_intra_df = pd.DataFrame()

    # Evaluates whether averaged flag dataframes exist to aggregate
    if averaged_flags_dfs:

        # Generates and displays the averaged immediate stage metrics summary
        avg_imp_df = generate_and_display_summary(all_events_avg_imp, channels, '5s Avg: Metrics by Stage')

        # Generates and displays the normalised averaged immediate stage metrics summary
        avg_imp_norm_df = generate_and_display_summary(all_events_avg_imp_norm, channels, '5s Avg: Normalised Metrics by Stage')

        # Generates and displays the averaged immediate stage change metrics summary
        avg_imp_change_df = generate_and_display_summary(all_changes_avg_imp, change_cols, '5s Avg: Stage Changes', drop_na_col=change_cols[0])
        
        # Generates and displays the averaged combined stage metrics summary
        avg_comb_df = generate_and_display_summary(all_events_avg_comb, channels, '5s Avg Combined Reagents: Metrics by Stage')

        # Generates and displays the normalised averaged combined stage metrics summary
        avg_comb_norm_df = generate_and_display_summary(all_events_avg_comb_norm, channels, '5s Avg Combined Reagents: Normalised Metrics by Stage')

        # Generates and displays the averaged combined stage change metrics summary
        avg_comb_change_df = generate_and_display_summary(all_changes_avg_comb, change_cols, '5s Avg Combined Reagents: Stage Changes', drop_na_col=change_cols[0])

        # Concatenates all averaged immediate intra-stage feature dataframes if populated, else assigns an empty dataframe
        avg_imp_intra_df = pd.concat(all_intra_avg_imp, ignore_index=True) if all_intra_avg_imp else pd.DataFrame()

        # Concatenates all averaged combined intra-stage feature dataframes if populated, else assigns an empty dataframe
        avg_comb_intra_df = pd.concat(all_intra_avg_comb, ignore_index=True) if all_intra_avg_comb else pd.DataFrame()

    # Loops through each channel to generate statistical plots for all consolidated splits
    for ch in channels:

        # Initialises the statistical plots collapsible output widget for the current channel
        stats_output = collapsible_output(f'Statistical Plots ({ch})')

        # Executes and displays the statistical plots within the collapsible widget
        with stats_output:

            # Generates the statistical plots for the absolute immediate signals
            plot_all_statistics(imp_df, 'stage', ch, ylabel=f'{ch} Signal', title_prefix=f'Immediate Absolute - {ch}')        

            # Generates the statistical plots for the normalised immediate signals
            plot_all_statistics(imp_norm_df, 'stage', ch, ylabel=f'{ch} Norm Signal', title_prefix=f'Immediate Normalised - {ch}')        

            # Generates the statistical plots for the immediate delta signals
            plot_all_statistics(imp_change_df, 'stage', f'{ch}_change', ylabel=f'Delta {ch}', title_prefix=f'Immediate Delta - {ch}')        
            
            # Generates the statistical plots for the absolute combined signals
            plot_all_statistics(comb_df, 'stage', ch, ylabel=f'{ch} Signal', title_prefix=f'Combined Absolute - {ch}')        

            # Generates the statistical plots for the normalised combined signals
            plot_all_statistics(comb_norm_df, 'stage', ch, ylabel=f'{ch} Norm Signal', title_prefix=f'Combined Normalised - {ch}')        

            # Generates the statistical plots for the combined delta signals
            plot_all_statistics(comb_change_df, 'stage', f'{ch}_change', ylabel=f'Delta {ch}', title_prefix=f'Combined Delta - {ch}')        
            
            # Evaluates whether averaged flag dataframes exist to generate corresponding plots
            if averaged_flags_dfs:

                # Generates the statistical plots for the absolute averaged immediate signals
                plot_all_statistics(avg_imp_df, 'stage', ch, ylabel=f'{ch} Signal', title_prefix=f'5s Avg Absolute - {ch}')

                # Generates the statistical plots for the normalised averaged immediate signals
                plot_all_statistics(avg_imp_norm_df, 'stage', ch, ylabel=f'{ch} Norm Signal', title_prefix=f'5s Avg Normalised - {ch}')

                # Generates the statistical plots for the averaged immediate delta signals
                plot_all_statistics(avg_imp_change_df, 'stage', f'{ch}_change', ylabel=f'Delta {ch}', title_prefix=f'5s Avg Delta - {ch}')
                
                # Generates the statistical plots for the absolute averaged combined signals
                plot_all_statistics(avg_comb_df, 'stage', ch, ylabel=f'{ch} Signal', title_prefix=f'5s Avg Comb Absolute - {ch}')

                # Generates the statistical plots for the normalised averaged combined signals
                plot_all_statistics(avg_comb_norm_df, 'stage', ch, ylabel=f'{ch} Norm Signal', title_prefix=f'5s Avg Comb Normalised - {ch}')

                # Generates the statistical plots for the averaged combined delta signals
                plot_all_statistics(avg_comb_change_df, 'stage', f'{ch}_change', ylabel=f'Delta {ch}', title_prefix=f'5s Avg Comb Delta - {ch}')

    # Returns a consolidated tuple containing all generated dataframes across every split and analysis type
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

    # Initialises the standard curve quality check collapsible output widget
    qc_output = collapsible_output('Standard Curve Quality Checks')

    # Executes and displays the standard curve data quality checks within the collapsible widget
    with qc_output:

        # Counts the total number of duplicated standard curve UUIDs
        duplicate_count = standard_curves_df['standard_curve_uuid'].duplicated().sum()

        # Displays the total count of duplicated UUIDs
        print('Duplicated standard curve UUIDs:', duplicate_count)

        def parse_delimited(s):
            '''Parses a delimited string field into a list of numeric values.

            Args:
                s (str): Serialised list from a standard-curve record.

            Returns:
                list[float]: Parsed numeric values.
            '''

            # Determines whether to use a pipe or a comma as the delimiter
            sep = '|' if '|' in str(s) else ','

            # Splits the string and converts the extracted values into a list of floats
            return [float(v) for v in str(s).split(sep)]

        # Initialises an empty dictionary to store the parsed standard curve data points
        data_points_SC = {}

        # Loops through each row in the standard curves dataframe using itertuples for speed
        for row in standard_curves_df.itertuples():

            # Parses the comma or pipe delimited x-coordinate string into a list
            x = parse_delimited(row.x_csv)

            # Parses the comma or pipe delimited y-coordinate string into a list
            y = parse_delimited(row.y_csv)

            # Evaluates whether the number of x-coordinates matches the number of y-coordinates
            if len(x) != len(y):

                # Prints a warning message identifying the mismatched coordinates
                print(f'Row {row.Index}: mismatched x/y lengths ({len(x)} vs {len(y)})')

            # Extracts and strips the raw time string from the current row
            time_str = str(row.time).strip()

            # Evaluates whether the time string contains hyphens instead of standard colons
            if '-' in time_str and ':' not in time_str:

                # Replaces hyphens with colons to standardise the time format
                time_str = time_str.replace('-', ':')

            # Parses the combined date and time strings into a datetime object
            dt = pd.to_datetime(f'{row.date} {time_str}', format='%Y-%m-%d %H:%M:%S', errors='coerce')

            # Maps the parsed curve arrays and metadata parameters to the curve UUID
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

        # Initialises an empty dictionary to store the cleaned standard curve data points
        clean_data_points_SC = {}

        # Prints the total number of standard curves before the missing value filter is applied
        print(f'Number of curves before removal: {len(data_points_SC)}')

        # Loops through each standard curve UUID and its associated data dictionary
        for k, v in data_points_SC.items():

            # Evaluates whether any missing values exist in the parsed coordinate arrays
            if pd.isnull(v['x']).any() or pd.isnull(v['y']).any():

                # Prints a warning indicating the removal of the curve due to missing values
                print(f'Removing curve: {k} due to NaN values.')

                # Prints the raw coordinate arrays to display the missing values
                print(f"x:\n{v['x']}\ny:\n{v['y']}\n")

            # Handles curves with complete coordinate arrays
            else:

                # Appends the valid curve data to the cleaned dictionary
                clean_data_points_SC[k] = v
                
        # Prints the final count of valid curves remaining after the cleaning process
        print(f'Remaining valid curves: {len(clean_data_points_SC)}')

    # Initialises the stored parameter plots collapsible output widget
    stored_plot_output = collapsible_output('Standard Curve Plots (Stored Parameters)')

    # Executes and displays the stored parameter plots within the collapsible widget
    with stored_plot_output:

        # Loops through each standard curve UUID and its cleaned data dictionary
        for key, data in clean_data_points_SC.items():

            # Generates a dense array of logarithmically spaced concentration points to plot the smooth curve
            x_pts = np.logspace(np.log10(min(data['x'])), np.log10(max(data['x'])), 1000)

            # Evaluates whether the defined fit model requires five parameters
            if data['fit_model'] == '5PL':

                # Calculates the predicted response curve using the five stored parameters
                y_pts = five_pl(x_pts, data['a'], data['d'], data['c'], data['b'], data['e']) 

            # Handles cases where a four-parameter logistic model is specified
            else:

                # Calculates the predicted response curve using the four stored parameters
                y_pts = four_pl(x_pts, data['a'], data['d'], data['c'], data['b'])

            # Initialises a Matplotlib figure for the current curve
            plt.figure(figsize=(10, 6))

            # Plots the raw measured coordinates as a scatter plot
            plt.scatter(data['x'], data['y'], label='Data', color='blue')

            # Plots the calculated curve fit derived from the stored database parameters
            plt.plot(x_pts, y_pts, color='red', label='Stored Fit')

            # Sets the x-axis to a logarithmic scale for concentration
            plt.xscale('log')

            # Sets the title identifying the curve and chip
            plt.title(f"Standard Curve {key} (Chip: {data['chip_id']}) - Stored Values")

            # Sets the x-axis label for concentration
            plt.xlabel('Concentration (ug/ml)')

            # Sets the y-axis label for signal response
            plt.ylabel('Signal')

            # Adds a legend to the plot
            plt.legend()

            # Renders the stored parameter curve plot
            plt.show()

    # Initialises the calculated parameter plots collapsible output widget
    calc_plot_output = collapsible_output('Standard Curve Plots (Calculated Parameters)')

    # Extracts a deduplicated list of all chip identifiers associated with valid standard curves
    unique_chips_SC = list({v['chip_id'] for v in clean_data_points_SC.values()})
    
    # Executes and displays the recalculated parameter plots within the collapsible widget
    with calc_plot_output:

        # Prints the total count of unique chips processed
        print(f'Total Unique Chips: {len(unique_chips_SC)}')

        # Loops through each unique chip identifier
        for chip_id in unique_chips_SC:

            # Loops through each standard curve UUID and its associated data dictionary
            for key, data in clean_data_points_SC.items():

                # Evaluates whether the current curve belongs to the specified chip
                if data['chip_id'] != chip_id:

                    # Skips to the next curve if the chip identifier does not match
                    continue

                # Recalculates the optimal logistic fit parameters based directly on the raw coordinates
                xfit, yfit, _ = fit_curve(data)

                # Initialises a Matplotlib figure for the recalculated curve
                plt.figure(figsize=(10, 6))

                # Plots the raw measured coordinates as a scatter plot
                plt.scatter(data['x'], data['y'], label='Data', color='blue')

                # Plots the recalculated optimal curve fit
                plt.plot(xfit, yfit, color='green', label='Calculated Fit')

                # Sets the x-axis to a logarithmic scale for concentration
                plt.xscale('log')

                # Sets the title identifying the recalculated curve and chip
                plt.title(f'Standard Curve {key} (Chip: {chip_id}) - Calculated Fit')

                # Sets the x-axis label for concentration
                plt.xlabel('Concentration (ug/ml)')

                # Sets the y-axis label for signal response
                plt.ylabel('Signal')

                # Adds a legend to the plot
                plt.legend()

                # Renders the recalculated parameter curve plot
                plt.show()

    # Initialises an empty list to aggregate the recalculated standard curve metrics
    sc_collected_data = []

    # Loops through each standard curve UUID and its cleaned data dictionary
    for key, data in clean_data_points_SC.items():

        # Recalculates the optimal logistic fit parameters to derive the new coefficients
        xfit, yfit, popt = fit_curve(data)

        # Evaluates whether the specified model is five-parameter logistic
        if data['fit_model'] == '5PL':

            # Generates the predicted response values for the original concentrations using the new five-parameter fit
            y_pred = five_pl(np.asarray(data['x']), *popt)

        # Handles cases where the specified model is four-parameter logistic
        else:

            # Generates the predicted response values for the original concentrations using the new four-parameter fit
            y_pred = four_pl(np.asarray(data['x']), *popt)

        # Calculates the residual sum of squares representing the variance unexplained by the fit
        ss_res = np.sum((np.asarray(data['y']) - y_pred) ** 2)

        # Calculates the total sum of squares representing the variance of the data from the mean
        ss_tot = np.sum((np.asarray(data['y']) - np.mean(data['y'])) ** 2)

        # Calculates the R-squared value to assess the goodness of the fit, preventing division by zero errors
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Appends a dictionary containing all curve metadata, raw points, stored parameters, and the newly calculated R-squared score to the collection list
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

    # Converts the aggregated standard curve collection list into a pandas dataframe
    sc_collected_df = pd.DataFrame(sc_collected_data)

    # Returns the cleaned data points dictionary alongside the recalculated metrics dataframe
    return clean_data_points_SC, sc_collected_df

# SC extra data
def calculate_absolute_peak(file_data, zone, median_time_step, t_start):
    '''Finds the absolute peak using a targeted window ~70s after the start flag.

    Args:
        file_data (pd.DataFrame): The complete raw sensorgram dataframe.
        zone (pd.DataFrame): The specific time window data for the current interval.
        median_time_step (float): The calculated median time difference between rows.
        t_start (float): The starting timestamp of the current interval.

    Returns:
        tuple: Index, primary channel value, secondary channel value, and timestamp of the absolute peak.
    '''
    
    # Calculates the target time seventy seconds after the start flag
    target_time = t_start + 70.0

    # Defines the start time for the targeted search window
    search_start_time = target_time - 15.0

    # Defines the end time for the targeted search window
    search_end_time = target_time + 15.0
    
    # Slices the zone data to isolate the specific search window
    search_zone = zone[(zone['time'] >= search_start_time) & (zone['time'] <= search_end_time)]
    
    # Evaluates whether the targeted search window contains any data
    if search_zone.empty:

        # Calculates the midpoint of the entire zone as a fallback
        t_mid = t_start + (zone['time'].max() - t_start) / 2

        # Reassigns the search zone to the first half of the available data
        search_zone = zone[zone['time'] <= t_mid]

    # Calculates the appropriate rolling window size for smoothing based on the median time step
    smooth_window = max(1, int(3.0 / median_time_step))

    # Applies a rolling mean to the primary channel to eliminate micro-noise at the peak
    smoothed_zone = search_zone['channel1'].rolling(window=smooth_window, center=True, min_periods=1).mean()
    
    # Identifies the index of the maximum signal value within the smoothed zone
    absolute_peak_idx = smoothed_zone.idxmax()

    # Extracts the corresponding data row from the parent file using the identified index
    abs_row = file_data.loc[absolute_peak_idx]

    # Returns the index, channel signals, and timestamp for the absolute peak
    return absolute_peak_idx, abs_row['channel1'], abs_row['channel2'], abs_row['time']

def calculate_pre_drop_peak(file_data, zone, absolute_peak_idx, global_drop_size, median_time_step, t_start, t_end, meas_1, meas_2, max_standard_cycle):
    '''Identifies the pre-drop peak strictly within a 60s (+/- 5s) window after the absolute peak.

    Args:
        file_data (pd.DataFrame): The complete raw sensorgram dataframe.
        zone (pd.DataFrame): The specific time window data for the current interval.
        absolute_peak_idx (int): The index of the previously calculated absolute peak.
        global_drop_size (pd.Series): The calculated sustained signal drops across the file.
        median_time_step (float): The calculated median time difference between rows.
        t_start (float): The starting timestamp of the current interval.
        t_end (float): The ending timestamp of the current interval.
        meas_1 (str): The label of the first measurement.
        meas_2 (str): The label of the second measurement.
        max_standard_cycle (float): The maximum duration of a standard interval cycle.

    Returns:
        tuple: Index, primary channel value, secondary channel value, and timestamp of the pre-drop peak.
    '''
    
    # Evaluates if the sequential measurements are identical
    if meas_1 == meas_2:

        # Assigns the full zone end time as the safe drop limit
        t_safe_drop_end = t_end 

    # Evaluates if the subsequent measurement indicates the final stage
    elif meas_2 == 'Final':

        # Assigns a proportional safe drop limit based on the maximum standard cycle
        t_safe_drop_end = t_start + (max_standard_cycle * 0.8)

    # Handles standard transitions between distinct measurements
    else:

        # Calculates a proportional safe drop limit based on the actual zone duration
        t_safe_drop_end = t_start + (t_end - t_start) * 0.8
        
    # Identifies the maximum valid index corresponding to the safe drop end time
    search_limit_idx = zone[zone['time'] <= t_safe_drop_end].index.max()

    # Calculates the number of rows representing a ten-second delay
    delay_rows = int(10.0 / median_time_step)

    # Defines the starting index for the drop search, incorporating the delay
    search_start_idx = min(absolute_peak_idx + delay_rows, search_limit_idx)

    # Slices the global drop size series to isolate the valid search range
    valid_drop_sizes = global_drop_size.loc[search_start_idx : search_limit_idx]
    
    # Evaluates whether the valid drop sizes series contains non-null data
    if not valid_drop_sizes.dropna().empty:

        # Identifies the index corresponding to the largest signal drop
        raw_drop_idx = valid_drop_sizes.idxmax()

        # Extracts the physical timestamp associated with the raw drop index
        raw_drop_time = file_data.loc[raw_drop_idx, 'time']

        # Extracts the physical timestamp of the pre-calculated absolute peak
        abs_peak_time = file_data.loc[absolute_peak_idx, 'time']
        
        # Calculates the strict target time sixty seconds after the absolute peak
        target_time = abs_peak_time + 60.0

        # Defines the start boundary for the targeted search window
        search_start = target_time - 5.0

        # Defines the end boundary for the targeted search window
        search_end = target_time + 5.0
        
        # Restricts the search end boundary to guarantee it precedes the raw drop
        search_end = min(search_end, raw_drop_time - 1.0)
        
        # Evaluates whether the restricted boundaries create an invalid or overlapping window
        if search_start >= search_end:

            # Adjusts the search start boundary to ensure a valid fallback window
            search_start = max(abs_peak_time, raw_drop_time - 6.0)
            
        # Slices the original zone data to isolate the final targeted search window
        search_zone = zone[(zone['time'] >= search_start) & (zone['time'] <= search_end)]
        
        # Checks whether the targeted search zone contains data
        if not search_zone.empty:

            # Calculates the window size required for a gentle one-second smoothing filter
            filter_win = max(1, int(1.0 / median_time_step))

            # Applies a rolling mean to the search zone to prevent selecting micro-noise
            smoothed_zone = file_data.loc[search_zone.index.min() : search_zone.index.max(), 'channel1'].rolling(
                window=filter_win, center=True, min_periods=1).mean()
            
            # Identifies the index of the maximum signal value within the smoothed search zone
            pre_drop_idx = smoothed_zone.idxmax()

        # Handles empty search zones
        else:

            # Defaults to the absolute peak index if the search zone is empty
            pre_drop_idx = absolute_peak_idx

    # Handles cases where no valid drop sizes are found
    else:

        # Defaults to the absolute peak index if drop identification fails
        pre_drop_idx = absolute_peak_idx 

    # Extracts the corresponding data row from the parent file using the pre-drop index
    pre_drop_row = file_data.loc[pre_drop_idx]
    
    # Returns the index, channel signals, and timestamp for the pre-drop peak
    return pre_drop_idx, pre_drop_row['channel1'], pre_drop_row['channel2'], pre_drop_row['time']

def calculate_baselines(file_data, zone, pre_drop_idx, median_time_step):
    '''Calculates Baseline 1 via dynamic shifting and Baseline 2 via pre-slope truncation, extracting the exact point values.

    Args:
        file_data (pd.DataFrame): The complete raw sensorgram dataframe.
        zone (pd.DataFrame): The specific time window data for the current interval.
        pre_drop_idx (int): The calculated index of the pre-drop peak.
        median_time_step (float): The calculated median time difference between rows.

    Returns:
        list[tuple]: A list of tuples containing calculated baseline definitions and parameters.
    '''
    
    # Initialises an empty list to track the calculated baseline records
    baselines = []
    
    # Extracts the physical timestamp corresponding to the pre-drop peak
    t_pre_drop = file_data.loc[pre_drop_idx, 'time']

    # Extracts the final timestamp present within the current zone
    t_zone_end = zone['time'].max()
    
    # Calculates a hard boundary halfway between the pre-drop peak and the zone end
    t_search_limit = t_pre_drop + (t_zone_end - t_pre_drop) / 2.0
    
    # Defines the upper limit for the first baseline search window
    t_b1_limit = min(t_search_limit, t_pre_drop + 30.0)
    
    # Slices the file data to isolate the relevant transition zone
    zone_data = file_data[(file_data['time'] >= t_pre_drop) & (file_data['time'] <= t_search_limit)].copy()
    
    # Evaluates whether the sliced transition zone contains data
    if zone_data.empty:

        # Returns the empty baselines list if no data is present
        return baselines
        
    # Defines the start boundary for the minimum signal search window
    t_min_search_start = t_pre_drop + 10.0

    # Defines the end boundary for the minimum signal search window
    t_min_search_end = min(t_b1_limit, t_pre_drop + 25.0) 
    
    # Slices the file data to isolate the minimum signal search window
    min_search_zone = file_data[(file_data['time'] >= t_min_search_start) & (file_data['time'] <= t_min_search_end)]
    
    # Checks whether the minimum search window contains data
    if not min_search_zone.empty:

        # Identifies the index of the minimum signal value within the search window
        min_idx = min_search_zone['channel1'].idxmin()

        # Extracts the physical timestamp corresponding to the minimum signal
        t_min = file_data.loc[min_idx, 'time']

    # Handles empty minimum search windows
    else:

        # Assigns a default fallback timestamp for the minimum signal
        t_min = t_pre_drop + 15.0
        
    # Slices the file data to isolate the maximum signal search window
    max_search_zone = file_data[(file_data['time'] >= t_min + 2.0) & (file_data['time'] <= t_b1_limit)]
    
    # Evaluates whether the maximum search window contains sufficient data points
    if len(max_search_zone) > 5:

        # Calculates the row count required for a five-second rolling window
        rolling_win_5s = max(1, int(5.0 / median_time_step))
        
        # Calculates the rolling mean across the maximum search window
        roll_mean = max_search_zone['channel1'].rolling(window=rolling_win_5s, center=True, min_periods=1).mean()

        # Calculates the rolling standard deviation across the maximum search window
        roll_std = max_search_zone['channel1'].rolling(window=rolling_win_5s, center=True, min_periods=1).std()
        
        # Calculates the median of the rolling standard deviations
        median_std = roll_std.median()

        # Filters the rolling mean series to isolate candidates with below-median variance
        flat_candidates = roll_mean[roll_std <= median_std]
        
        # Checks whether any flat candidates were identified
        if not flat_candidates.empty:

            # Identifies the index of the maximum signal among the flat candidates
            best_idx = flat_candidates.idxmax()

        # Handles cases where no flat candidates exist
        else:

            # Identifies the index of the absolute maximum signal from the rolling mean
            best_idx = roll_mean.idxmax()
            
        # Extracts the physical timestamp for the first baseline
        t_base1 = file_data.loc[best_idx, 'time']

    # Handles insufficient data in the maximum search window
    else:

        # Assigns a fallback timestamp for the first baseline
        t_base1 = t_min + 5.0
        
    # Calculates the row count for a two-second smoothing window
    smooth_win = max(1, int(2.0 / median_time_step))

    # Applies a rolling mean to smooth the entire transition zone
    smoothed_signal = zone_data['channel1'].rolling(window=smooth_win, center=True, min_periods=1).mean()

    # Calculates the step size corresponding to a two-second interval
    step = max(1, int(2.0 / median_time_step))

    # Calculates and assigns the step differences of the smoothed signal
    zone_data['delta'] = smoothed_signal.diff(step)
    
    # Slices the zone data up to the first baseline limit
    b1_zone = zone_data[zone_data['time'] <= t_b1_limit]

    # Calculates a dynamic edge threshold based on the signal variance
    b1_edge_threshold = max(0.5, b1_zone['delta'].std() * 0.3)
    
    # Loops to iteratively refine the first baseline position
    for _ in range(4): 

        # Defines the start boundary for the baseline check window
        t_start_check = t_base1 - 2.5

        # Defines the end boundary for the baseline check window
        t_end_check = t_base1 + 2.5

        # Slices the zone data to isolate the check window
        check_zone = zone_data[(zone_data['time'] >= t_start_check) & (zone_data['time'] <= t_end_check)]
        
        # Breaks the loop if the check window is empty
        if check_zone.empty:
            break
            
        # Evaluates if the signal is dropping too sharply
        if check_zone['delta'].min() < -b1_edge_threshold:

            # Shifts the baseline position backwards by one second
            t_base1 -= 1.0

        # Evaluates if the signal is rising too sharply
        elif check_zone['delta'].max() > b1_edge_threshold:

            # Shifts the baseline position forwards by one second
            t_base1 += 1.0

        # Handles stable baseline conditions
        else:

            # Breaks the refinement loop
            break 
            
    # Restricts the refined baseline to remain within the allowed limit
    t_base1 = min(t_base1, t_b1_limit - 2.5)
        
    # Assigns the exact point start time for the first baseline
    t_start_win1 = t_base1

    # Assigns the exact point end time for the first baseline
    t_end_win1 = t_base1
    
    # Identifies the index closest to the targeted first baseline time
    exact_idx1 = (file_data['time'] - t_base1).abs().idxmin()

    # Extracts the primary channel signal at the first baseline index
    ch1_val1 = file_data.loc[exact_idx1, 'channel1']

    # Extracts the secondary channel signal at the first baseline index
    ch2_val1 = file_data.loc[exact_idx1, 'channel2']
    
    # Appends the first baseline record to the tracking list
    baselines.append((t_start_win1, t_end_win1, ch1_val1, ch2_val1, '- 1'))

    # Defines the start boundary for the second baseline search window
    t_b2_search_start = t_base1 + 10.0

    # Defines the end boundary for the second baseline search window
    t_b2_search_end = t_zone_end - 3.0 
    
    # Slices the file data to isolate the full search zone for the second baseline
    b2_full_zone = file_data[(file_data['time'] >= t_b2_search_start) & (file_data['time'] <= t_b2_search_end)].copy()
    
    # Calculates the step size corresponding to a two-second interval
    step_size = max(1, int(2.0 / median_time_step))

    # Calculates the row count for a two-second smoothing window
    smooth_win_b2 = max(1, int(2.0 / median_time_step))

    # Calculates the step differences of the smoothed signal within the search zone
    b2_full_zone['delta'] = b2_full_zone['channel1'].rolling(window=smooth_win_b2, center=True, min_periods=1).mean().diff(step_size)
    
    # Calculates a dynamic edge threshold based on the search zone variance
    edge_thresh = max(0.5, b2_full_zone['delta'].std() * 0.3)
    
    # Identifies indices where the signal change exceeds the edge threshold
    valid_edges = b2_full_zone[(b2_full_zone['delta'].abs() > edge_thresh) & (b2_full_zone['time'] > t_b2_search_start + 5.0)]
    
    # Checks whether any valid slope edges were detected
    if not valid_edges.empty:

        # Identifies the timestamp of the first significant slope
        t_next_slope = valid_edges['time'].min()

        # Slices the search zone to exclude the subsequent slope
        b2_zone = b2_full_zone[b2_full_zone['time'] < t_next_slope]

    # Handles cases where no significant slopes are detected
    else:

        # Assigns the full search zone for baseline evaluation
        b2_zone = b2_full_zone
        
    # Evaluates whether the second baseline zone contains sufficient data
    if len(b2_zone) > int(5.0 / median_time_step):

        # Calculates the row count required for a five-second rolling window
        rolling_win_5s = max(1, int(5.0 / median_time_step))
        
        # Calculates the rolling mean across the second baseline zone
        b2_mean = b2_zone['channel1'].rolling(window=rolling_win_5s, center=True, min_periods=1).mean()

        # Calculates the rolling standard deviation across the second baseline zone
        b2_std = b2_zone['channel1'].rolling(window=rolling_win_5s, center=True, min_periods=1).std()
        
        # Defines the maximum allowed signal height relative to the first baseline
        max_allowed_height = ch1_val1 + 5.0 
        
        # Creates a boolean mask to isolate candidates within the height limit
        valid_mask = (b2_mean < max_allowed_height) & b2_std.notna()
        
        # Evaluates whether any valid candidates meet the height criteria
        if valid_mask.any():

            # Identifies the index with the minimum variance among valid candidates
            best_idx_b2 = b2_std[valid_mask].idxmin()

            # Extracts the physical timestamp for the selected second baseline
            t_base2 = file_data.loc[best_idx_b2, 'time']

        # Handles cases where no candidates meet the height criteria
        else:

            # Identifies the index with the minimum variance across the entire zone
            best_idx_b2 = b2_std.idxmin()

            # Extracts the physical timestamp for the selected second baseline
            t_base2 = file_data.loc[best_idx_b2, 'time']

    # Handles insufficient data in the second baseline zone
    else:

        # Assigns a fallback timestamp for the second baseline
        t_base2 = t_b2_search_start + 2.5
        
    # Restricts the second baseline to maintain a minimum separation from the first
    t_base2 = max(t_base2, t_base1 + 10.0)

    # Restricts the second baseline to remain within the valid zone boundary
    t_base2 = min(t_base2, t_zone_end - 2.5)
    
    # Assigns the exact point start time for the second baseline
    t_start_win2 = t_base2

    # Assigns the exact point end time for the second baseline
    t_end_win2 = t_base2
    
    # Identifies the index closest to the targeted second baseline time
    exact_idx2 = (file_data['time'] - t_base2).abs().idxmin()

    # Extracts the primary channel signal at the second baseline index
    ch1_val2 = file_data.loc[exact_idx2, 'channel1']

    # Extracts the secondary channel signal at the second baseline index
    ch2_val2 = file_data.loc[exact_idx2, 'channel2']
    
    # Appends the second baseline record to the tracking list
    baselines.append((t_start_win2, t_end_win2, ch1_val2, ch2_val2, '- 2'))
    
    # Returns the compiled list of baseline records
    return baselines

def calculate_sc_flags(files, save_dir='Standard Curve Files'):
    '''Calculates dynamic baseline and peak flags from sensorgram data, saves them to a CSV, updates the dictionary, and optionally plots the results.

    Args:
        files (dict): Dictionary holding the parsed file data.
        save_dir (str, optional): Root directory where the CSV files should be saved. Defaults to 'Standard Curve Files'.

    Returns:
        dict: The updated files dictionary containing the new baseline flag dataframes.
    '''

    # Loops through each folder in the parsed files dictionary
    for folder in files.keys():

        # Evaluates whether any file within the current folder contains the baseline flags identifier
        if any('baseline_flags' in key for key in files[folder].keys()):

            # Prints a message indicating that baseline flags already exist and the folder will be skipped
            print(f'Baseline flags already exist in folder {folder}. Skipping entire folder.')

            # Proceeds to the next folder in the loop
            continue

        # Initialises a variable to track the previous file name for flag matching
        previous_file = ''

        # Loops through each file within the current folder
        for file in list(files[folder].keys()):
            
            # Evaluates whether the current file contains sensorgram data
            if 'sensorgram' in file:
                
                # Sorts the sensorgram data chronologically and resets its index
                file_data = files[folder][file].sort_values('time').reset_index(drop=True)

                # Sorts the matched flag data chronologically and resets its index
                previous_file_data = files[folder][previous_file].sort_values('time').reset_index(drop=True)

                # Duplicates the primary channel to preserve its raw unnormalised state
                file_data['channel1_raw'] = file_data['channel1']

                # Duplicates the secondary channel to preserve its raw unnormalised state
                file_data['channel2_raw'] = file_data['channel2']
                
                # Extracts the initial signal value for the primary channel
                ch1_initial = file_data['channel1'].iloc[0]

                # Extracts the initial signal value for the secondary channel
                ch2_initial = file_data['channel2'].iloc[0]
                
                # Zero-bases the primary channel using its initial value
                file_data['channel1'] = file_data['channel1'] - ch1_initial

                # Zero-bases the secondary channel using its initial value
                file_data['channel2'] = file_data['channel2'] - ch2_initial

                # Updates the dictionary with the normalised sensorgram data
                files[folder][file] = file_data
                
                # Calculates the median time step to dynamically define row windows
                median_time_step = file_data['time'].diff().median()

                # Calculates the number of rows corresponding to a five-second lookahead window
                lookahead_rows = max(1, int(5 / median_time_step))
                
                # Calculates the number of rows corresponding to a thirty-second total block
                total_block_rows = int(30 / median_time_step) 

                # Calculates the number of rows corresponding to a five-second tail sample offset
                tail_sample_offset = int(5 / median_time_step)
                
                # Initialises an empty list to store the extracted analysis intervals
                intervals = []

                # Loops through the matched flag data to define stage intervals
                for i in range(len(previous_file_data) - 1):

                    # Extracts the start time for the current interval
                    t_start = previous_file_data.iloc[i]['time']

                    # Extracts the end time for the current interval from the subsequent flag
                    t_end = previous_file_data.iloc[i+1]['time']

                    # Extracts the primary measurement label and strips appended suffixes
                    meas_1 = str(previous_file_data.iloc[i]['conc']).split('-')[0]

                    # Extracts the secondary measurement label and strips appended suffixes
                    meas_2 = str(previous_file_data.iloc[i+1]['conc']).split('-')[0]

                    # Appends the defined interval parameters to the tracking list
                    intervals.append((t_start, t_end, meas_1, meas_2))
                    
                # Extracts the final measurement label from the last flag record
                last_meas = str(previous_file_data.iloc[-1]['conc']).split('-')[0]

                # Appends the final open-ended interval to the tracking list
                intervals.append((previous_file_data.iloc[-1]['time'], file_data['time'].max(), last_meas, 'Final'))
                
                # Calculates the total durations for all standard closed intervals
                cycle_lengths = [t_end - t_start for (t_start, t_end, m1, m2) in intervals if m2 != 'Final']

                # Identifies the maximum standard cycle duration or assigns a default fallback
                max_standard_cycle = max(cycle_lengths) if cycle_lengths else 600.0
                
                # Calculates the row count for a five-second averaging window
                avg_win = max(1, int(5.0 / median_time_step))
                
                # Calculates a trailing rolling mean to represent the signal before a drop
                trailing_avg = file_data['channel1'].rolling(window=avg_win, min_periods=1).mean()
                
                # Calculates a leading rolling mean by shifting the trailing average backwards
                leading_avg = trailing_avg.shift(-avg_win)
                
                # Calculates the sustained difference to identify true wash drops
                global_drop_size = trailing_avg - leading_avg
                    
                # Initialises an empty list to store the newly calculated flag records
                saved_file_rows = []
                
                # Extracts the absolute minimum timestamp from the sensorgram data
                t_absolute_first = file_data['time'].min()

                # Assigns the start boundary for the initial baseline window
                t_start_init = t_absolute_first

                # Assigns the end boundary for the initial baseline window
                t_end_init = t_absolute_first + 5.0
                
                # Slices the sensorgram data to isolate the initial baseline window
                init_zone = file_data[(file_data['time'] >= t_start_init) & (file_data['time'] <= t_end_init)]

                # Checks whether the initial baseline zone contains data
                if not init_zone.empty:

                    # Appends the compiled initial baseline record to the tracking list
                    saved_file_rows.append({
                        'Window_Start': t_start_init,
                        'Window_End': t_end_init,
                        'information': 'Initial Baseline',
                        'ch1_avg': init_zone['channel1'].mean(),
                        'ch2_avg': init_zone['channel2'].mean(),
                        'ch1_absolute_peak': np.nan,
                        'ch2_absolute_peak': np.nan,
                        'absolute_peak_time': np.nan,
                        'ch1_peak': np.nan,
                        'ch2_peak': np.nan,
                        'peak_time': np.nan
                    })
    
                # Loops through each defined analysis interval
                for idx, (t_start, t_end, meas_1, meas_2) in enumerate(intervals):

                    # Slices the sensorgram data to isolate the current interval
                    zone = file_data[(file_data['time'] >= t_start) & (file_data['time'] <= t_end)]

                    # Skips the interval if it contains no data
                    if zone.empty:
                        continue
                        
                    # Calculates the absolute peak parameters for the interval
                    absolute_peak_idx, ch1_abs_val, ch2_abs_val, abs_time_val = calculate_absolute_peak(file_data, zone, median_time_step, t_start)

                    # Calculates the pre-drop peak parameters for the interval
                    pre_drop_idx, ch1_peak_val, ch2_peak_val, peak_time_val = calculate_pre_drop_peak(file_data, zone, absolute_peak_idx, global_drop_size, median_time_step, t_start, t_end, meas_1, meas_2, max_standard_cycle)

                    # Calculates the multiple baselines associated with the current interval
                    baselines = calculate_baselines(file_data, zone, pre_drop_idx, median_time_step)
                    
                    # Loops through each calculated baseline
                    for t_start_win, t_end_win, ch1_win_avg, ch2_win_avg, suffix in baselines:

                        # Appends the compiled baseline and peak metrics to the tracking list
                        saved_file_rows.append({
                            'Window_Start': t_start_win,
                            'Window_End': t_end_win,
                            'information': f'Baseline Window {suffix} ({meas_1} / {meas_2})',
                            'ch1_avg': ch1_win_avg,
                            'ch2_avg': ch2_win_avg,
                            'ch1_absolute_peak': ch1_abs_val,
                            'ch2_absolute_peak': ch2_abs_val,
                            'absolute_peak_time': abs_time_val,
                            'ch1_peak': ch1_peak_val,
                            'ch2_peak': ch2_peak_val,
                            'peak_time': peak_time_val
                        })
                    
                # Extracts the absolute maximum timestamp from the sensorgram data
                t_absolute_last = file_data['time'].max()

                # Assigns the start boundary for the final baseline window
                t_start_final = t_absolute_last - 5.0

                # Assigns the end boundary for the final baseline window
                t_end_final = t_absolute_last
                
                # Slices the sensorgram data to isolate the final baseline window
                final_zone = file_data[(file_data['time'] >= t_start_final) & (file_data['time'] <= t_end_final)]
        
                # Appends the compiled final baseline record to the tracking list
                saved_file_rows.append({
                    'Window_Start': t_start_final,
                    'Window_End': t_end_final,
                    'information': 'Final Baseline',
                    'ch1_avg': final_zone['channel1'].mean(),
                    'ch2_avg': final_zone['channel2'].mean(),
                    'ch1_absolute_peak': np.nan,
                    'ch2_absolute_peak': np.nan,
                    'absolute_peak_time': np.nan,
                    'ch1_peak': np.nan,
                    'ch2_peak': np.nan,
                    'peak_time': np.nan
                })

                # Prints a confirmation message indicating the successful calculation of flags
                print(f'Calculated flags for {folder}/{file}')

                # Converts the aggregated flag records into a pandas dataframe
                flag_df = pd.DataFrame(saved_file_rows)

                # Constructs the output filename for the new baseline flags
                file_name = file.split('_')[1] + '_baseline_flags.csv'

                # Constructs the complete file path for saving the CSV
                file_path = os.path.join(save_dir, str(folder), file_name)
                
                # Creates the target directory structure if it does not already exist
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Saves the newly generated baseline flags dataframe to a CSV file
                flag_df.to_csv(file_path, index=False)
        
                # Updates the parsed files dictionary with the new dataframe
                files[folder][file_name.replace('.csv', '')] = flag_df

            # Updates the previous file tracker for the next iteration
            previous_file = file

    # Returns the updated files dictionary containing all generated flags
    return files

def plot_baselines_and_peaks(file_data, previous_file_data, baseline_flags, title):
    '''Generates the detailed Plotly visualisation with tail-sampling windows and peaks.

    Args:
        file_data (pd.DataFrame): The raw sensorgram file data.
        previous_file_data (pd.DataFrame): The previous flag file data.
        baseline_flags (pd.DataFrame): The calculated baseline and peak flags.
        title (str): Title for the generated plot.
    '''

    # Creates a copy of the baseline flags dataframe to preserve the original data
    baseline_flags = baseline_flags.copy()

    # Calculates the midpoint timestamp for each baseline window
    baseline_flags['mid_time'] = (baseline_flags['Window_Start'] + baseline_flags['Window_End']) / 2
            
    # Initialises an empty Plotly figure object
    fig = go.Figure()

    # Adds a line trace representing the raw primary channel sensorgram data
    fig.add_trace(go.Scatter(x=file_data['time'], y=file_data['channel1'], mode='lines', name='Channel 1 Raw Data', line=dict(color='#1f77b4', width=1.5)))

    # Evaluates whether the previous flag data contains valid records
    if previous_file_data is not None and not previous_file_data.empty:

        # Loops through each row in the previous flag data
        for _, flag_row in previous_file_data.iterrows():

            # Adds a dashed vertical line denoting the specific flag timestamp
            fig.add_vline(x=flag_row['time'], line_width=1, line_dash='dash', line_color='black', opacity=1)

            # Adds a text annotation displaying the flag concentration label
            fig.add_annotation(x=flag_row['time'], y=1.02, yref='paper', text=str(flag_row['conc']), showarrow=False, textangle=-45, xanchor='left', yanchor='bottom', font=dict(size=9, color='#0f0f0f'))

    # Adds a scatter trace marking the calculated baseline averages
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

    # Checks whether the pre-drop peak columns exist in the dataframe
    if 'ch1_peak' in baseline_flags.columns and 'peak_time' in baseline_flags.columns:

        # Drops any rows containing missing pre-drop peak values
        valid_peaks = baseline_flags.dropna(subset=['ch1_peak', 'peak_time'])

        # Adds a scatter trace marking the valid pre-drop peaks
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

    # Checks whether the absolute peak columns exist in the dataframe
    if 'ch1_absolute_peak' in baseline_flags.columns and 'absolute_peak_time' in baseline_flags.columns:
        
        # Drops any rows containing missing absolute peak values
        valid_abs_peaks = baseline_flags.dropna(subset=['ch1_absolute_peak', 'absolute_peak_time'])

        # Adds a scatter trace marking the valid absolute peaks
        fig.add_trace(go.Scatter(
            x=valid_abs_peaks['absolute_peak_time'], y=valid_abs_peaks['ch1_absolute_peak'],
            mode='markers', name='Absolute Peak',
            marker=dict(color='magenta', symbol='diamond', size=9, line=dict(color='black', width=1)),
            hoverinfo='text',
            text=[
                f"Absolute Peak Value: {r['ch1_absolute_peak']:.2f} RU<br>Time: {r['absolute_peak_time']:.1f}s" 
                for _, r in valid_abs_peaks.iterrows()
            ]
        ))
    
    # Loops through each row in the calculated baseline flags
    for _, r in baseline_flags.iterrows():

        # Adds a shaded vertical rectangle to visually represent the baseline averaging window
        fig.add_vrect(
            x0=r['Window_Start'], x1=r['Window_End'],
            fillcolor='rgba(44, 160, 44, 0.12)', layer='below', 
            line_width=1, line_color='rgba(44, 160, 44, 0.4)', line_dash='dot'
        )
    
    # Extracts the primary channel average from the initial baseline record
    start_avg_ru = baseline_flags.iloc[0]['ch1_avg']

    # Adds a horizontal reference line for the initial baseline average
    fig.add_hline(y=start_avg_ru, line_width=1.5, line_dash='dash', line_color='purple', annotation_text='Start Average', annotation_position='top left')

    # Extracts the primary channel average from the final baseline record
    final_avg_ru = baseline_flags.iloc[-1]['ch1_avg']

    # Adds a horizontal reference line for the final baseline average
    fig.add_hline(y=final_avg_ru, line_width=1.5, line_dash='dash', line_color='#e67e22', annotation_text='Final Average', annotation_position='bottom left')

    # Configures the overall layout, axis titles, and theme of the Plotly figure
    fig.update_layout(
        title=title + ' (Normalised to Initial Baseline)',
        xaxis_title='Time (Seconds)',
        yaxis_title='Response Units (RU)',
        yaxis=dict(range=[file_data['channel1'][0] - 0.5, file_data['channel1'].max() + 0.5]),
        template='plotly_white',
        hovermode='closest'
    )
    
    # Renders the interactive Plotly figure
    fig.show()

def calculate_and_plot_shift(shift_data, title):
    '''Calculates baseline shifts and plots the twin-axis visualisation.

    Args:
        shift_data (pd.DataFrame): Dataframe containing the baseline shift data.
        title (str): Title for the generated plot.
    '''

    # Creates a filtered copy of the shift data excluding intermediate baseline records
    shift_data = shift_data[~shift_data['information'].astype(str).str.contains('- 1')].reset_index(drop=True)

    # Extracts the initial baseline average to serve as the zero reference
    initial_baseline = shift_data['ch1_avg'].iloc[0]

    # Calculates the absolute shift of each baseline relative to the initial reference
    baseline_shift = shift_data['ch1_avg'] - initial_baseline

    # Calculates the absolute peak response relative to the initial reference
    peak_response = shift_data['ch1_peak'] - initial_baseline

    # Calculates the baseline shift as a percentage of the peak response
    shift_percentage = (baseline_shift / peak_response) * 100
    
    # Assigns the calculated absolute baseline shifts to the dataframe
    shift_data['baseline_shift_raw'] = baseline_shift

    # Assigns the calculated absolute peak responses to the dataframe
    shift_data['peak_response_raw'] = peak_response

    # Assigns the calculated percentage shifts to the dataframe
    shift_data['ch1_shift_pct'] = shift_percentage

    # Defines the list of columns required for the final summary table
    summary_cols = ['information', 'ch1_peak', 'ch1_avg', 'baseline_shift_raw', 'ch1_shift_pct']

    # Creates a focused summary dataframe containing only the required columns
    summary_table = shift_data[summary_cols].copy()
    
    # Standardises the summary table column headers for display
    summary_table.columns = ['Measurement Step', 'Pre-Drop Peak (RU)', 'Absolute Baseline (RU)', 'Baseline Shift (ΔRU)', '% Shift of Peak']

    # Rounds all numerical values in the summary table to three decimal places
    summary_table = summary_table.round(3)
    
    # Initialises a Matplotlib figure with specific dimensions and a primary axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Calculates the midpoint timestamps for plotting on the x-axis
    x_times = (shift_data['Window_Start'] + shift_data['Window_End']) / 2
    
    # Plots the absolute baseline shifts on the primary y-axis
    ax1.plot(x_times, baseline_shift, 'x-', color='b', label='Baseline Shift (Absolute)')

    # Sets the x-axis label for time
    ax1.set_xlabel('Time (s)')

    # Sets the primary y-axis label for absolute response
    ax1.set_ylabel('Response (a.u.)', color='b')

    # Colours the primary y-axis ticks to match the absolute shift plot
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Creates a secondary y-axis sharing the same x-axis
    ax2 = ax1.twinx()

    # Plots the percentage shifts on the secondary y-axis
    ax2.plot(x_times, shift_percentage, 'o--', color='r', alpha=0.6, label='% Shift of Peak')

    # Sets the secondary y-axis label for percentage shift
    ax2.set_ylabel('% of Peak Response', color='r')

    # Colours the secondary y-axis ticks to match the percentage shift plot
    ax2.tick_params(axis='y', labelcolor='r')

    # Sets the main title for the twin-axis shift plot
    plt.title(f'{title} - Baseline-shifted response (initial = {initial_baseline:.2f})')

    # Adds a subtle background grid tied to the primary axis
    ax1.grid(True, alpha=0.3)
    
    # Extracts the legend handles and labels from the primary axis
    lines_1, labels_1 = ax1.get_legend_handles_labels()

    # Extracts the legend handles and labels from the secondary axis
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    # Combines the extracted handles and labels into a single unified legend
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # Renders the twin-axis Matplotlib plot
    plt.show()

    # Renders the formatted summary data table
    display(summary_table)

def analyse_standard_curves_extra(files, save_dir='Standard Curve Files'):
    '''Iterates over the processed files dictionary, generating and displaying all requested outputs neatly inside collapsible UI widgets.

    Args:
        files (dict): Dictionary holding the parsed file data.
        save_dir (str, optional): Root directory where the CSV files should be saved. Defaults to 'Standard Curve Files'.

    Returns:
        dict: The fully processed and updated files dictionary.
    '''

    # Executes the calculation of dynamic standard curve flags and updates the files dictionary
    files = calculate_sc_flags(files, save_dir)

    # Loops through each folder present within the files dictionary
    for folder in files.keys():

        # Prints a header indicating the folder currently being evaluated
        print(f'\nEvaluating Folder: {folder}')
        
        # Initialises a variable to track the previous file name for flag matching
        previous_file = ''

        # Loops through each file within the current folder
        for file in list(files[folder].keys()):

            # Strips extraneous copy prefixes and whitespace from the file name
            clean_filename = file.replace('Copy of ', '').strip()

            # Splits the cleaned file name by underscores to extract identifiers
            parts = clean_filename.split('_')

            # Assigns the extracted chip identifier or a default unknown label
            chip_id = parts[0] if len(parts) > 0 else 'Unknown'

            # Assigns the extracted measurement identifier or a default unknown label
            measurement_id = parts[1] if len(parts) > 1 else 'Unknown'

            # Evaluates whether the current file represents a sensorgram
            if 'sensorgram' in file:

                # Prints a sub-header displaying the current chip and measurement identifiers
                print(f'\n--- Chip ID: {chip_id} | Measurement ID: {measurement_id} ---')

                # Assigns the full sensorgram data dataframe for the current file
                file_data = files[folder][file]

                # Retrieves the corresponding flag data dataframe or defaults to None
                previous_file_data = files[folder].get(previous_file, None)
                
                # Slices the sensorgram dataframe to isolate only the required columns for plotting
                plot_data = file_data[['time', 'channel1', 'channel2']]

                # Initialises and opens a collapsible widget for the basic sensorgram plot
                with collapsible_output(f'Basic Sensorgram: {file}'):

                    # Generates and displays the basic sensorgram plot within the widget
                    plot_sensorgrams({file: plot_data}, 'Standard Curve')
                
                # Evaluates whether matching flag data exists and contains records
                if previous_file_data is not None and not previous_file_data.empty:
                    
                    # Initialises and opens a collapsible widget for the flagged sensorgram plot
                    with collapsible_output(f'Sensorgram with Flags: {file}'):

                        # Extracts the timestamp list from the flag data
                        times = previous_file_data['time'].tolist()

                        # Extracts the concentration labels list from the flag data
                        labels = previous_file_data['conc'].tolist()

                        # Constructs the title for the flagged sensorgram plot
                        title = f'Standard Curve Sensorgram for {file}'

                        # Defines a uniform colour list for the vertical flag markers
                        colours = ['tab:blue'] * len(times)
                        
                        # Generates and displays the flagged sensorgram plot within the widget
                        plot_flags_on_sensorgrams(plot_data, times, labels, title, colours)

                # Constructs the expected filename for the generated baseline flags
                file_name_flags = file.split('_')[1] + '_baseline_flags'

                # Evaluates whether the expected baseline flags file exists in the current folder
                if file_name_flags in files[folder]:

                    # Retrieves the calculated baseline flags dataframe
                    baseline_flags = files[folder][file_name_flags]

                    # Initialises and opens a collapsible widget for the baseline and peak plot
                    with collapsible_output(f'Baseline & Peak Tail-Sampling: {file}'):

                        # Generates and displays the detailed baseline and peak plot within the widget
                        plot_baselines_and_peaks(file_data, previous_file_data, baseline_flags, f'Tail-Sampling Window Method with Peaks - {file}')

                    # Initialises and opens a collapsible widget for the baseline shift analysis
                    with collapsible_output(f'Baseline Shift Analysis: {file}'):

                        # Generates and displays the baseline shift plot and summary table within the widget
                        calculate_and_plot_shift(baseline_flags, file)

            # Updates the previous file tracker for the next iteration
            previous_file = file

    # Returns the final processed files dictionary
    return files
