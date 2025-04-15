import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from typing import List, Optional, Union, Tuple
from matplotlib.axes import Axes
from scipy.stats import probplot
from scipy.stats import t as t_distribution
from scipy.stats import vonmises
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

from matplotlib.patches import Arc, FancyArrowPatch, Patch
from shared_utils.utils.math_utils import rotate_vector
from shared_utils.utils.math_utils import normalize_vector
from shared_utils.utils.math_utils import calculate_unsigned_angle

def draw_circular_arrow(ax, arc_center, radius, starting_direction, arc_angle, is_rotation_clockwise, color, arrowhead_size, arc_line_width, arc_label=None, zorder=1):
    # Calculate the end angle
    start_angle = np.degrees(np.arctan2(starting_direction[1], starting_direction[0]))
    end_angle = start_angle + (arc_angle if is_rotation_clockwise == False else -arc_angle)

    # Draw the arc
    if is_rotation_clockwise:
        arc_start_angle = end_angle
        arc_end_angle = start_angle
    else:
        arc_start_angle = start_angle
        arc_end_angle = end_angle
    arc = Arc(arc_center, width=radius, height=radius, angle=0, theta1=arc_start_angle, theta2=arc_end_angle, color=color, lw=arc_line_width, zorder=zorder)
    ax.add_patch(arc)

    # Calculate the position and direction for the arrowhead at the end of the arc
    arrow_end_angle_rad = np.deg2rad(end_angle)
    arrow_x = arc_center[0] + radius * 0.5 * np.cos(arrow_end_angle_rad)
    arrow_z = arc_center[1] + radius * 0.5 * np.sin(arrow_end_angle_rad)

    # Calculate tangent direction
    tangent_direction = np.array([np.cos(arrow_end_angle_rad), np.sin(arrow_end_angle_rad)])
    tangent_direction = np.array([-tangent_direction[1], tangent_direction[0]]) if is_rotation_clockwise == False else np.array([tangent_direction[1], -tangent_direction[0]])

    # Draw the arrowhead at the end of the arc
    arrow = FancyArrowPatch((arrow_x, arrow_z), (arrow_x + 0.1 * tangent_direction[0], arrow_z + 0.1 * tangent_direction[1]), 
                            color=color, arrowstyle='-|>', mutation_scale=arrowhead_size, label=arc_label, zorder=zorder)
    ax.add_patch(arrow)

# Plotting the trial with CCQ naming convention
def plot_trial(trial_data, color_palette, show_plot=False, tracking_data=None):
    
    starting_marker_size = 8
    walking_head_width = 0.15
    walking_line_width = 2
    tracking_line_width = 2
    tracking_line_alpha = 0.5

    starting_position_color = color_palette[0]

    arrow_encoding_distance_color = color_palette[4]

    enconding_color = color_palette[1]
    angle_encoding_position_color = color_palette[4]

    crystal_marker_size_start = 10

    production_color = color_palette[2]
    crystal_production_marker_size = 16

    arc_arrowhead_size = 20
    arc_line_width = 2

    text_font_size = 14
    text_font_ticks = 12
    text_font_size_title = 16

    fig, ax = plt.subplots(figsize=(10,10))

    # If tracking data is provided, plot it first so it appears in the background
    if tracking_data is not None:
        # Filter tracking data between timeAtReachedReposition and timeAtEndProductionDistance
        mask = (tracking_data['timestamp'] >= trial_data['timeAtReachedReposition']) & \
               (tracking_data['timestamp'] <= trial_data['timeAtEndProductionDistance'])
        filtered_tracking = tracking_data[mask]

         # Create points for the line segments
        points = np.array([filtered_tracking['position_x'], filtered_tracking['position_z']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create a gradient colormap from light gray to dark gray
        n_segments = len(segments)
        cmap = mcolors.LinearSegmentedColormap.from_list("", ["#EEEEEE", "#333333"])
        norm = plt.Normalize(0, n_segments)
        colors = [cmap(norm(i)) for i in range(n_segments)]
        
        # Create the line collection
        lc = LineCollection(segments, colors=colors, linewidth=tracking_line_width, alpha=0.7, zorder=1)
        ax.add_collection(lc)
        
        # Add a single legend entry for the tracking data
        ax.plot([], [], color='gray', linewidth=tracking_line_width, alpha=tracking_line_alpha, label='Tracking Data')

    # plotting the starting position
    ax.plot(trial_data['startingCorner_x'], trial_data['startingCorner_z'], 
            'o', markersize=starting_marker_size, color=starting_position_color, label='Start')
    
    # plotting the walking distance
    dx = trial_data['turningEncodingPosition_x'] - trial_data['startingCorner_x']
    dz = trial_data['turningEncodingPosition_z'] - trial_data['startingCorner_z']
    ax.arrow(trial_data['startingCorner_x'], trial_data['startingCorner_z'], 
             dx, dz, 
             head_width=walking_head_width, lw=walking_line_width, color=arrow_encoding_distance_color, length_includes_head=True, label='Encoding Distance')

    encoding_start = np.array([trial_data['turningEncodingPosition_x'], trial_data['turningEncodingPosition_z']])

    ax.plot(encoding_start[0], encoding_start[1], 
            marker="o", markersize=starting_marker_size, color=angle_encoding_position_color)

    # extending the walking distance
    encoding_direction = np.array([trial_data['encodingDirection_x'], trial_data['encodingDirection_z']])

    additional_walking = encoding_start + encoding_direction
    plt.plot([encoding_start[0], additional_walking[0]], [encoding_start[1], additional_walking[1]], 
             lw=walking_line_width, linestyle='--', color=arrow_encoding_distance_color)

    # plotting the encoding rotation - crystal position
    encoding_angle_direction = rotate_vector(encoding_direction, trial_data['encodingAngle'], trial_data['isEncodingClockwise'])
    crystal_encoding_position = encoding_angle_direction + encoding_start
    
    plt.plot(crystal_encoding_position[0], crystal_encoding_position[1], 
             marker='d', markersize=crystal_marker_size_start, color=enconding_color)    

    # plotting the direction towards the crystal encoding position
    plt.plot([encoding_start[0], crystal_encoding_position[0]], [encoding_start[1], crystal_encoding_position[1]], 
             linestyle=':', lw=walking_line_width, color=enconding_color)
    
    draw_circular_arrow(ax, encoding_start, 1, encoding_direction, trial_data['encodingAngle'], trial_data['isEncodingClockwise'],enconding_color,arc_arrowhead_size, arc_line_width, arc_label='Encoding Angle')

    # plotting where the crystal has been placed    
    placed_crystal = np.array([trial_data['productionDistance_x'],trial_data['productionDistance_z']])

    plt.plot(placed_crystal[0],placed_crystal[1],marker="d",markersize=crystal_production_marker_size,color=production_color)

    production_direction = normalize_vector(np.array([trial_data['productionDistance_x'] - trial_data['turningEncodingPosition_x'], trial_data['productionDistance_z'] - trial_data['turningEncodingPosition_z']]))
    production_angle = calculate_unsigned_angle(encoding_angle_direction, production_direction, trial_data['isProductionClockwise'])

    draw_circular_arrow(ax, encoding_start, 2, encoding_angle_direction, production_angle, trial_data['isProductionClockwise'],production_color,arc_arrowhead_size, arc_line_width, arc_label="Production Angle")

    # plotting the distance towards the placed crystal
    plt.plot([encoding_start[0], placed_crystal[0]], [encoding_start[1], placed_crystal[1]], linestyle=':', lw=walking_line_width, color=production_color, label='Production Distance')
        
    # Set the limits for x-axis and y-axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_aspect('equal', adjustable='box')
    
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xlabel('X Position', fontsize=text_font_size)
    plt.ylabel('Z Position', fontsize=text_font_size)
    plt.xticks(fontsize=text_font_ticks)
    plt.yticks(fontsize=text_font_ticks)

    plt.legend(loc='best', fontsize=text_font_ticks)

    if trial_data['isEncodingClockwise']:
        encoding_dir = "cw"
    else:
        encoding_dir = "ccw"

    if trial_data['isProductionClockwise']:
        production_dir = "cw"    
    else:
        production_dir = "ccw"

    plt.title(f'Trial {trial_data["sequenceNumber"]} - Encoding {encoding_dir} - Production {production_dir}', fontsize=text_font_size_title)

    if(show_plot == True):
        plt.show()

    return fig

def plot_custom_violin(y_data: List[np.array], colors: List[str], mean_color, x_label, y_label, x_categories: List[str], background_color: str = "white", hlines: Optional[List[float]] = None, ylims: List[float] = [0,2], p_value = 1.0, bar_offset = 0.1, ax: Optional[Axes]=None, label_fontsize: int = 18,
    tick_fontsize: int = 15, show_mean_labels: bool = True):
    # Create jittered version of "x" (which is only 0, 1)
    positions = list(range(len(y_data)))
    jitter = 0.04
    x_data = [np.array([i] * len(d)) for i, d in enumerate(y_data)]
    x_jittered = [x + t_distribution(df=6, scale=jitter).rvs(len(x)) for x in x_data]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 12))
    else:
        fig = ax.figure

    # Background color
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    # Add horizontal lines for reference ---------------------------------------------------
    if hlines:
        for h in hlines:
            ax.axhline(h, color="#7F7F7F", ls=(0, (5, 5)), alpha=0.8, zorder=0)

    # Add violin plots ---------------------------------------------------
    for i, data in enumerate(y_data):
        violins = ax.violinplot(
            [data], 
            positions=[positions[i]],
            widths=0.45,
            bw_method="silverman",
            showmeans=False, 
            showmedians=False,
            showextrema=False
        )

        # Customize violins (remove fill, customize line, etc.)
        for pc in violins["bodies"]:
            pc.set_facecolor("none")
            pc.set_edgecolor("#282724")
            pc.set_linewidth(1.4)
            pc.set_alpha(1)

    # Add boxplots ---------------------------------------------------
    medianprops = dict(
        linewidth=4, 
        color="#747473",
        solid_capstyle="butt"
    )
    boxprops = dict(
        linewidth=2, 
        color="#747473"
    )

    ax.boxplot(
        y_data,
        positions=positions, 
        showfliers=False, # Do not show the outliers beyond the caps.
        showcaps=False,   # Do not show the caps
        medianprops=medianprops,
        whiskerprops=boxprops,
        boxprops=boxprops
    )

    for x, y, color in zip(x_jittered, y_data, colors):
        ax.scatter(x, y, s=100, color=color, alpha=0.4)

    # Adding means
    mean_color = mean_color
    means = [y.mean() for y in y_data]
    for i, mean in enumerate(means):
        # Add dot representing the mean
        ax.scatter(i, mean, s=250, color=mean_color, zorder=3)

        if show_mean_labels:
            # Add line connecting mean value and its label
            ax.plot([i, i + 0.25], [mean, mean], ls="dashdot", color="black", zorder=3)

            # Add mean value label.
            ax.text(
                i + 0.25,
                mean,
                r"$\hat{\mu}_{\rm{mean}} = $" + str(round(mean, 2)),
                fontsize=13,
                va="center",
                bbox=dict(
                    facecolor="white",
                    edgecolor="black",
                    boxstyle="round",
                    pad=0.15
                ),
                zorder=10  # to make sure the line is on top
            )

    # Adding p values ----------------------------------
    if(p_value < 0.05):
        max_y = max(max(data) for data in y_data)
        p_bar_height = max_y + bar_offset
        tick_len = 0.02
        ax.plot([0, 0, 1, 1], [p_bar_height - tick_len, p_bar_height, p_bar_height, p_bar_height - tick_len], c="black")


        # Add labels for the p-values
        label_bar = rf"$p$ = {p_value:.2e}"

        pad = 0.02
        ax.text(0.5, p_bar_height + pad, label_bar, fontsize=11, va="bottom", ha="center")

    # Set y-axis limits
    ax.set_ylim(ylims[0], ylims[1])  # Set the limits for the y-axis
    
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    # Customize spines color
    ax.spines["left"].set_color("#b4aea9")
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_color("#b4aea9")
    ax.spines["bottom"].set_linewidth(2)

    # Customize labels and ticks
    ax.tick_params(length=0)
    if hlines:
        ax.set_yticks(hlines)
        ax.set_yticklabels(hlines, size=tick_fontsize)
    ax.set_ylabel(y_label, size=label_fontsize)

    # xlabels accounts for the sample size for each species
    ax.set_xticks(positions)
    ax.set_xticklabels(x_categories, size=tick_fontsize, ha="center", ma="center")
    ax.set_xlabel(x_label, size=label_fontsize)

    return fig, ax

# Visual check for normality
def plot_histogram_qqplot(data, variable_name, group):    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Group {group}', fontsize=16)
    
    sns.histplot(data, kde=True, ax=axs[0])
    axs[0].set_title(f'Histogram of {variable_name}')
    
    probplot(data, dist="norm", plot=axs[1])
    axs[1].set_title(f'Q-Q Plot of {variable_name}')
    
    plt.tight_layout()
    return fig

def plot_correlation_model(data_df, model_r2, real_data, prediction_data, real_data_label, prediction_data_label, color_palette, output_folder, save_individual_plots=True, ylim = [0,360] ):

    grouped = data_df.groupby('subject_id')

    if save_individual_plots == True:
        for subject_id, group in grouped:
            fig, ax = plt.subplots(figsize=(8, 6))
            plt.scatter(group[real_data], group[prediction_data], alpha=0.7)
            plt.xlabel(real_data_label)
            plt.ylabel(prediction_data_label)
            plt.grid(True)
            ax.set_aspect('equal', adjustable='box')  # adjustable='box' ensures the plot box adjusts to the set aspect
            # Save plot
            participant_output_directory = os.path.join(output_folder,f'{subject_id}')
            os.makedirs(participant_output_directory, exist_ok=True)
            plot_filename = os.path.join(participant_output_directory, f'participant_{subject_id}_{prediction_data}.png')
            plt.savefig(plot_filename)
            plt.close()

    cmap = plt.get_cmap('rainbow')
    values = np.linspace(0, 1, len(grouped))
    colors = cmap(values)

    fig, ax = plt.subplots(figsize=(8, 8))
    for i , (subject_id, group) in enumerate(grouped):
        color = colors[i]
        # Add to cumulative plot
        plt.scatter(group[real_data], group[prediction_data], alpha=0.2, color=color)
        
        # Fit a linear regression line
        coeffs = np.polyfit(group[real_data], group[prediction_data], 1)
        poly_eq = np.poly1d(coeffs)
        plt.plot(group[real_data], poly_eq(group[real_data]), color=color, linewidth=1.5, alpha=0.2)

    production_min = data_df[real_data].min()
    production_max = data_df[real_data].max()
    plt.plot([0, production_max], [0, production_max], 'k--', linewidth=3.5, label='(y = x)', alpha = 0.5)

    # Fit overall regression line and calculate confidence intervals
    X = sm.add_constant(data_df[real_data])
    y = data_df[prediction_data]
    ols_model = sm.OLS(y, X)
    model = ols_model.fit()

    production_angle_range = np.linspace(production_min, production_max, 100)
    X_range = sm.add_constant(production_angle_range)
    pred = model.get_prediction(X_range).summary_frame()

    colorline = color_palette[1]
    colorCI = color_palette[0]

    plt.plot(production_angle_range, pred['mean'], color=colorline, linewidth=3.0, alpha=0.85, label='mean regression')
    plt.fill_between(production_angle_range, 
                    pred['mean_ci_lower'], 
                    pred['mean_ci_upper'],
                    color=colorCI, alpha=0.85, label=f'95% ci')

    custom_label = f'mean $R^2 = {model_r2:.2f}$'

    # Finalize cumulative plot
    ax.set_aspect('equal', adjustable='box')  # adjustable='box' ensures the plot box adjusts to the set aspect
    plt.xlim(0,360)
    plt.ylim(ylim)
    plt.xlabel(real_data_label)
    plt.ylabel(prediction_data_label)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color='w', label=custom_label))
    labels.append(custom_label)
    plt.legend(handles=handles, frameon=False)

    plt.grid(False)
    return fig, ax

def plot_model_production_angles(data_df, prediction_data, color_palette, dark_color_palette):
    gray_scale = ["#d3d3d3", "#999999", "#696969", "#1c1c1c"]
    steel_blue = color_palette[1]
    ucla_blue = color_palette[5]
    light_coral = color_palette[4]
    amaranth_purple = color_palette[0]
    orange_wheel = color_palette[6]
    chocolate_cosmos = color_palette[9]
    mint = color_palette[2]
    cerise = color_palette[10]
    cambridge_blue = color_palette[7]
    sunset= color_palette[3]

    # Assigning specific names to the darker versions
    dark_steel_blue = dark_color_palette[1]
    dark_ucla_blue = dark_color_palette[5]
    dark_light_coral = dark_color_palette[4]
    dark_amaranth_purple = dark_color_palette[0]
    dark_orange_wheel = dark_color_palette[6]
    dark_chocolate_cosmos = dark_color_palette[9]
    dark_mint = dark_color_palette[2]
    dark_cerise = dark_color_palette[10]
    dark_atomic_tangerine = dark_color_palette[8]
    dark_sunset = dark_color_palette[3]

    color_palette_dict = {45.0: gray_scale[0], 120.0: gray_scale[1], 315.0: gray_scale[2], 240.0: gray_scale[3]}
    dark_color_palette_dict = {45.0: orange_wheel, 120.0: chocolate_cosmos, 315.0: mint, 240.0: ucla_blue}

    encoding_angles = np.sort(data_df['encodingAngle'].unique())
    fig, axs = plt.subplots(1, len(encoding_angles), subplot_kw={'projection': 'polar'}, figsize=(20, 5))

    binwidth = 10

    for i, angle in enumerate(encoding_angles):

        ax = axs[i]
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(rf'$\theta_{{enc}} = {angle}°$')
        ax.set_ylim(0, 1)

        homing_angles = np.sort(data_df[data_df['encodingAngle'] == angle]['homingAngle'].unique())
        for homing_angle in homing_angles:
            single_encoding_angle_data = data_df[(data_df['encodingAngle'] == angle) & 
                                                        (data_df['homingAngle'] == homing_angle)]

            # We are using the model production angle here
            hist_data = single_encoding_angle_data[prediction_data]
            hist_data = np.deg2rad(hist_data)  # Convert degrees to radians
    
            bins = np.arange(0, 2*np.pi + np.deg2rad(binwidth), np.deg2rad(binwidth))
            hist, edges = np.histogram(hist_data, bins=bins, density=True)  # Calculate density histogram
            
            # Normalize the density by the global maximum density
            hist = hist / hist.max()

            color = color_palette_dict.get(homing_angle)
            dark_color = dark_color_palette_dict.get(homing_angle)

            bars = ax.bar(edges[:-1], hist, width=np.deg2rad(binwidth), align='edge', edgecolor='black', color=color, alpha=0.5)

            # Fit a von Mises distribution to the data
            kappa, loc, scale = vonmises.fit(hist_data, fscale=1)  # Fit the von Mises distribution
            x = np.linspace(0, 2*np.pi, 100)
            fitted_pdf = vonmises.pdf(x, kappa, loc, scale)
            
            # Normalize the fitted PDF to the maximum of the histogram
            fitted_pdf = fitted_pdf / fitted_pdf.max()

            # Plot the fitted von Mises distribution
            ax.plot(x, fitted_pdf, color=color, linestyle='--')

            # Overlay data points
            ax.scatter(hist_data, np.ones_like(hist_data), color=color, alpha=0.4, s=75)

            # Add solid line representing the homing angle
            ax.plot([np.deg2rad(homing_angle), np.deg2rad(homing_angle)], [0, 1], color=dark_color, linestyle='-', linewidth=2.5)
        
    # Create custom legend
    legend_handles = [Patch(color=color, label=f'{angle}°') for angle, color in color_palette_dict.items()]
    plt.legend(handles=legend_handles, title=r'$\theta_{hom}$', loc='upper right', bbox_to_anchor=(1.5, 0.75), frameon=False)
    plt.tight_layout()

    return fig,axs
    