# tuner_plots.py
#
# plotting functions for tuner.py
import re
import torch
import logging
import itertools

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages

from tqdm import tqdm
from math import sqrt, ceil, floor
from typing import *
from omegaconf import DictConfig
from . import tuner_utils
from .tuner_utils import none

log = logging.getLogger(__name__)

def determine_int_axticks(
	series: pd.Series, 
	target_num_ticks: int = 10
) -> List[int]:
	'''
	Determine integer axticks for discrete values
	
		params:
			series (pd.Series)		: a series containing integer values to be plotted along an axis
			target_num_ticks (int)	: the desired number of ticks
		
		returns:
			int_axticks (int)		: a list of ints splitting the series into approximately targe_num_ticks groups
	'''
	if isinstance(series,list):
		series = pd.Series(series)
	
	lowest = series.min()
	highest = series.max()
	
	if (highest - lowest) == target_num_ticks or (highest - lowest) < target_num_ticks:
		return [i for i in range(lowest, highest + 1)]
	
	new_min = target_num_ticks - 1
	new_max = target_num_ticks + 1
	while not highest % target_num_ticks == 0:
		if highest % new_min == 1:
			target_num_ticks = new_min
			break
		else:
			new_min -= 1
			# if we get here, it means highest is a prime and there's no good solution,
			# so we'll just brute force something later
			if new_min == 1:
				break
		
		if highest % new_max == 0:
			target_num_ticks = new_max
			break
		elif not new_max >= highest/2:
			new_max += 1
	
	int_axticks = [int(i) for i in list(range(lowest - 1, highest + 1, int(ceil(highest/target_num_ticks))))]
	int_axticks = [i for i in int_axticks if i in range(lowest, highest)]
	
	if not int_axticks:
		int_axticks = list(set([i for i in series.values]))
	
	return int_axticks		

# main plotting functions
def scatterplot(
	data: pd.DataFrame, x: pd.DataFrame, y: pd.DataFrame, 
	val: str, hue: str, 
	ax: matplotlib.axes.Axes, sem: str = None,
	text: str = None, text_color: str = None,
	text_size: str = None, center_at_origin: bool = False,
	aspect_ratio: str = None, comparison_line: bool = False,
	legend_title: str = None, legend_labels: Dict = None, 
	diffs_plot: bool = False,
	marginal_means: List[str] = None, 
	xlabel: str = None, ylabel: str = None,
	plot_kwargs: Dict = {}, line_kwargs: Dict = {}, 
	text_kwargs: Dict = {}, label_kwargs: Dict = {},
) -> matplotlib.axes.Axes:
	'''
	The main function used for creating scatterplots for tuner objects
	
		params:
			data (pd.DataFrame) 		: a dataframe containing information about the data to plot
			x (pd.DataFrame)			: a dataframe containing information and values to plot on the x-axis.
										  Note that passing a column name will not work, unlike in matplotlib or seaborn
			y (pd.DataFrame)			: a dataframe containing information and values to plot on the y-axis.
										  Note that passing a column name will not work, unlike in matplotlib or seaborn
			val (str)					: the measure to plot on the x and y axes. only one measure may be plotted (i.e., x and y show the same kind of data)
			hue (str)					: a string indicating a column in the dataframes to use for group colors
			ax (matplotlib.axes.Axes)	: an object to plot on
			sem (str)					: name of column containing standard error values in x and y
			text (str)					: name of column containing text to add to the plot for x and y
			text_color (str)			: name of column containing groups to color text differently for
			text_size (str)				: name of column containing groups to size text differently for
			center_at_origin (bool)		: whether to center the plot at the origin
			aspect_ratio (str)			: whether to adjust the aspect ratio of the plot. should be one of 'square', 'eq_square'
										  'eq_square' produces a plot with equal-size x- and y-axes with equal limits
										  'square' produces a plot with visually equal-size x- and y-axes, but with possible different ranges on x and y axes
			comparison_line (bool)		: whether to draw a comparison line (diagonal if diffs_plot is false, at x = 0 if diffs_plot is true)
			legend_title (str)			: what to title the plot legend (if it exists)
			legend_labels (dict)		: a dictionary mapping the values in the dataframes' hue columns to display labels
			diffs_plot (bool)			: whether to create a plot where y = y - x
			marginal_means (list)		: a list of column names, each separate grouping of which to add ticks for the group mean and standard error on the plot's margins for
			xlabel (str)				: the label of the x-axis
			ylabel (str)				: the label of the y-axis
			plot_kwargs (dict)			: arguments passed to sns scatterplot
			line_kwargs (dict)			: arguments passed to ax.plot (used to draw comparison lines)
			text_kwargs (dict)			: arguments passed to ax.text (used when adding text to plots)
			label_kwargs (dict)			: arguments passed to ax.set_xlabel and ax.set_ylabel (when xlabel and ylabel are provided)
		
		returns:
			ax (matplotlib.axes.Axes)	: the ax object after the plot has been created
	'''
						
	data = data.copy()
	x = x.copy()
	y = y.copy()
	
	# seaborn can't plot tensors, so make sure everything is float
	for col in [val, sem]:
		for df in [x, y]:
			if col in df:
				df[col] = df[col].astype(float)
	
	if diffs_plot:
		y = y.copy()
		y[val] = y[val] - x[val]
	
	sns.scatterplot(data=data, x=x[val], y=y[val], hue=hue, ax=ax, **plot_kwargs)
	
	if sem is not None:
		# this makes sure the colors of the points match the color of the errorbars
		collections = ax.collections[1:].copy()
		
		for (_, x_group), (_, y_group), collection in zip(x.groupby(hue), y.groupby(hue), collections):
			ax.errorbar(x=x_group[val], xerr=x_group[sem], y=y_group[val], yerr=y_group[sem], color=collection._original_edgecolor, ls='none')
	
	(xllim, xulim), (yllim, yulim) = get_set_plot_limits(ax, center_at_origin, aspect_ratio, diffs_plot)
	
	if comparison_line:
		yline = (0,0) if diffs_plot else (yllim,yulim)
		ax.plot((xllim, xulim), yline, linestyle='--', color='k', scalex=False, scaley=False, zorder=0, alpha=.3, **line_kwargs)
	
	if aspect_ratio is not None and 'square' in aspect_ratio:
		ax.set_aspect(1./ax.get_data_ratio(), adjustable='box')
	
	if text in data.columns:
		v_adjust = (yulim - yllim)/100
		
		set_unique_text_colors(data, text_color)
		
		set_unique_text_sizes(data, text_size)
		
		for line in data.index:
			# this uses the user specified color/size if it exists, otherwise use the one we just set
			tmp_text_kwargs = {'color': data.loc[line]['__tmp__color__'], 'size': data.loc[line]['__tmp__size__'], **text_kwargs}
			ax.text(
				x.loc[line, val], y.loc[line, val] - v_adjust, 
				data.loc[line].token.replace(chr(288), ''), **tmp_text_kwargs
			)
	
	set_legend_title(ax, legend_title)
	
	set_legend_labels(ax, legend_labels)
	
	if marginal_means:
		colors = [collection._original_edgecolor for collection in ax.collections[1:]]
		
		# if we've added the errorbars, it repeats the original colors twice (once for each axis the error bar is shown on) 
		# we just want to pass the unique colors
		colors = sorted(set(colors), key=colors.index)
		add_marginal_mean_ticks_to_plot(x=x, y=y, val=val, ax=ax, groups=marginal_means, colors=colors)
	
	if xlabel is not None:
		ax.set_xlabel(xlabel, **label_kwargs)
	
	if ylabel is not None:
		ax.set_ylabel(ylabel, **label_kwargs)
	
	return ax

def add_marginal_mean_ticks_to_plot(
	x: pd.DataFrame, 
	y: pd.DataFrame, 
	val: str, 
	ax: matplotlib.axes.Axes,
	colors: List[Tuple[float]] = None,
	groups: List[str] = None,
) -> matplotlib.axes.Axes:
	'''
	Adds ticks to the margins of the plot for means and standard devations for groups
	
		params:
			x (pd.DataFrame)			: the data plotted on the x-axis
			y (pd.DataFrame)			: the data plotted on the y-axis
			val (str)					: which value to plot group means for
			ax (matplotlib.axes.Axes)	: the ax object to add mean ticks to
			colors (list)				: a list of colors (expressed as float tuples) corresponding to groups grouped by color in the plot
			groups (list)				: a list of columns in x and y. for each combination of unique values in all columns, a separate mean tick and se will be adde
		
		returns:
			ax (matplotlib.axes.Axes)	: the ax object with the mean and standard error ticks added to the margins
	'''
	# here we add ticks to show the mean and standard errors along each axis
	groups = [groups] if isinstance(groups, str) else groups
	
	if colors is None:
		# if no colors are passed, default to black
		colors = [(0.,0.,0.)]
	
	if groups is not None and all(all(group in data.columns for group in groups) for data in [x, y]):
		# this makes sure that the colors used in the plot and the colors used for the groups in the plot and the colors used for marginal mean ticks match
		groups 		= sorted(groups, key = lambda group: x[group].unique().size == len(colors))
		
		# repeat the colors as needed for each additional group
		for group in groups[:-1]:
			colors  *= x[group].unique().size
		
		group_means = [data.groupby(groups)[val].agg({'mean', 'sem'}) for data in [x,y]]
	else:
		group_means = [data[val].agg('mean', 'sem') for data in [x,y]]
	
	xllim, xulim 	= ax.get_xlim()
	yllim, yulim 	= ax.get_ylim()
	xtick_range 	= (xulim - xllim)/30
	ytick_range 	= (yulim - yllim)/30
	
	line_kwargs 	= dict(linestyle='-', zorder=0, scalex=False, scaley=False, alpha=.3)
	
	for axis, group_mean, llim, tick_range in zip(['x', 'y'], group_means, [yllim, xllim], [xtick_range, ytick_range]):
		for (groupname, group), color in zip(group_mean.groupby(groups), colors):
			
			line_kwargs.update(dict(color=color))
			
			x_loc 		= (group.loc[groupname, 'mean'], group.loc[groupname, 'mean'])
			y_loc 		= (llim, llim + tick_range)
			xsem_loc 	= (x_loc - group.loc[groupname, 'sem'], x_loc + group.loc[groupname, 'sem'])
			ysem_loc 	= (llim + tick_range/2, llim + tick_range/2)
			
			if axis == 'y':
				x_loc, y_loc = y_loc, x_loc
				xsem_loc, ysem_loc = ysem_loc, xsem_loc
			
			ax.plot(x_loc, y_loc, **line_kwargs)
			ax.plot(xsem_loc, ysem_loc, linewidth=0.75, **line_kwargs)
	
	return ax


# setters
def get_set_plot_limits(
	ax: matplotlib.axes.Axes, 
	center_at_origin: bool = False,
	aspect_ratio: str = '', 
	diffs_plot: bool = False
) -> Tuple:
	'''
	Sets and returns the plot limits for the appropriate plot type
	
		params:
			ax (matplotlib.axes.Axes)	: the ax object to get/set limits for
			center_at_origin (bool)		: whether to center the plot at the origin
			aspect_ratio (str)			: what aspect ratio to use. currently, only 'eq_square' is implemented
										  'eq_square' produces a plot with x and y axis that cover the same range
			diffs_plot (str)			: whether this is a plot where y = y - x
		
		returns:
		 	limits (tuple)				: a tuple with ((x lower lim, x upper lim), (y lower lim), (y upper lim))
		 								  according to the passed parameters
	'''
	if center_at_origin:
		xulim = max([*np.abs(ax.get_xlim()), *np.abs(ax.get_ylim())])
		xulim += (ax.get_xlim()[1] - ax.get_xlim()[0])/32
		yulim = xulim
		xllim, yllim = -xulim, -yulim
	elif aspect_ratio == 'eq_square' and not diffs_plot:
		xulim = max([*ax.get_xlim(), *ax.get_ylim()])
		xulim += (ax.get_xlim()[1] - ax.get_xlim()[0])/32
		xllim = min([*ax.get_xlim(), *ax.get_ylim()])
		xllim -= (ax.get_xlim()[1] - ax.get_xlim()[0])/32
		yulim = xulim
		yllim = xllim
	else:
		xllim, xulim = ax.get_xlim()
		xllim -= (ax.get_xlim()[1] - ax.get_xlim()[0])/32
		xulim += (ax.get_xlim()[1] - ax.get_xlim()[0])/32
		yllim, yulim = ax.get_ylim()
		yllim -= (ax.get_ylim()[1] - ax.get_ylim()[0])/32
		yulim += (ax.get_ylim()[1] - ax.get_ylim()[0])/32
	
	if diffs_plot:
		yulim = max(np.abs(ax.get_ylim()))
		yulim += (ax.get_ylim()[1] - ax.get_ylim()[0])/32
		yllim = -yulim
	
	ax.set_xlim([xllim, xulim])
	ax.set_ylim([yllim, yulim])
	
	return (xllim, xulim), (yllim, yulim)

def set_unique_text_colors(
	df: pd.DataFrame, 
	text_color: Union[str, Dict]	
) -> None:
	'''
	Set unique color values to use for groups
	
		params:
			df (pd.DataFrame)		: a dataframe to add color information to for each group
			text_color (str or dict): if str, a column in the dataframe to generate unique text colors for
									  if dict, must contain one entry of the form 'colname': str, where str
									  is the colname in the dataframe to add color information for. other entries
									  map entries in colname to the text color desired for them
	'''
	def colors_generator(default_value: Tuple[float], n: int) -> Tuple[float]:
		'''
		Generate a color palette with n unique colors
		
			params:
				n (int): the number of unique colors to generate
		'''
		yield from sns.color_palette('bright', n_colors=n)
	
	set_unique_values(
		df=df,
		value_name='color',
		value_str_or_dict=text_color,
		default_value=(0.,0.,0.),
		values_generator=colors_generator,
	)

def set_unique_text_sizes(
	df: pd.DataFrame, 
	text_size: Union[str, Dict]
) -> None:
	'''
	Set unique text sizes to use for groups
	
		params:
			df (pd.DataFrame)		: a dataframe to add text size information to for each group
			text_size (str or dict)	: if str, a column in the dataframe to generate unique text colors for
									  if dict, must contain one entry of the form 'colname': str, where str
									  is the colname in the dataframe to add size information for. other entries
									  map entries in colname to the text size desired for them
	'''
	def sizes_generator(n):
		yield from range(6, (6+n)*2, 2)
	
	set_unique_values(
		df=df, 
		value_name='size',
		value_str_or_dict=text_size,
		default_value=6,
		values_generator=sizes_generator
	)

def set_unique_values(
	df: pd.DataFrame,
	value_name: str,
	value_str_or_dict: Union[str, Dict] = None,
	default_value: 'any' = None,
	values_generator: Generator = None
) -> None:
	'''
	Adds information about unique grouping values to a dataframe for plotting
	
		params:
			df (pd.DataFrame)				: the dataframe to add values to
			value_name (str)				: the kind of value being added (e.g., color, size)
			value_str_or_dict (str or dict) : if str, a column in the dataframe to generate unique values for
											  if dict, must contain one entry of the form 'colname': str, where str
											  is the colname in the dataframe to add value information for. other entries
											  map entries in colname to the values desired for them
			default_value (any)				: what the default/first value to add
			values_generator (Generator)	: a generator that yields unique values of the appropriate type
	'''
			
	default_value = [default_value] if not isinstance(default_value,list) else default_value
	
	if isinstance(value_str_or_dict, str) or value_str_or_dict is None:
		if value_str_or_dict in df.columns:
			n_values = df[value_str_or_dict].unique().size - 1
			default_value.append(list(values_generator(n_values)))
			
			default_value = dict(zip(df[value_name].unique(), default_value))
			df[f'__tmp__{value_name}__'] = [default_value[group] for group in df[value_str_or_dict]]
		else:
			df[f'__tmp__{value_name}__'] = [default_value[0] for line in df.index]
	elif isinstance(value_str_or_dict, dict):
		if all(k in df[value_str_or_dict['colname']].unique() for k in value_str_or_dict if k != 'colname'):
			df[f'__tmp__{value_name}__'] = [value_str_or_dict[k] for k in df[value_str_or_dict['colname']]]
		else:
			raise ValueError(f'Not all keys in {value_str_or_dict} were found in the df columns!')

def set_legend_title(
	ax: matplotlib.axes.Axes,
	legend_title: str = None
) -> None:
	'''
	Sets the plot legend title
	
		params:
			ax (matplotlib.axes.Axes)	: the plot to set the legend title for
			legend_title (str)			: what to set the legend title to
	'''
	if legend_title is not None:
		if legend_title != '':
			try:
				ax.get_legend().set_title(legend_title)
			except AttributeError:
				log.warning('A legend title was provided but no legend exists.')
				pass
		elif legend_title == '':
			# if the legend title is '', we want to delete the whole thing
			try:
				# this ensures the title exists
				# if it doesn't, an error is thrown and we'll exit without overwriting the wrong thing
				_ = ax.get_legend().get_title()
				handles, labels = ax.get_legend_handles_labels()
				ax.legend(handles=handles, labels=labels)				
			except AttributeError:
				pass

def set_legend_labels(
	ax: matplotlib.axes.Axes,
	legend_labels: Dict = None
) -> None:
	'''
	Set the plots legend labels
	
		params:
			ax (matplotlib.axes.Axes)	: the plot to set legend labels for
			legend_labels (Dict)		: a dict mapping the default legend labels (taken from colnames in the data)
										  to the desired display labels
	'''
	if legend_labels is not None:
		for text in ax.get_legend().get_texts():
			try:
				text.set_text(legend_labels[text.get_text()])
			# this happens if the text isn't in the mapping, or if there is no legend to add the labels to
			# we just ignore it
			except (KeyError, AttributeError):
				pass


# used during tuning/evaluation
def get_plot_title(
	df: pd.DataFrame,
	metric: str = ''
) -> str:
	'''
	Get a plot title for tuner plots
	
		params:
			df (pd.DataFrame)	: a dataframe containing information about the experiment
			metric (str)		: the metric for which a plot title is being created
		
		returns:
			title (str)			: a plot title with information from df and metric
	'''
	title = tuner_utils.multiplator(df.model_name, multstr="Multiple models'")
	title += f' {metric}'
	if 'eval_epoch' in df.columns:
		title += (' @ epoch ' + str(df.eval_epoch.unique()[0]) + '/') if df.eval_epoch.unique().size == 1 else ', epochs: '
		title += str(tuner_utils.multiplator(df.total_epochs))
	elif 'epoch' in df.columns:
		title += f', epochs: {df.epoch.max()}'
	
	if 'epoch_criteria' in df.columns:
		title += f' ({tuner_utils.multiplator(df.epoch_criteria, multstr="multiple criteria").replace("_", " ")})'
	
	title += f'\nmin epochs: {tuner_utils.multiplator(df.min_epochs)}, '
	title += f'max epochs: {tuner_utils.multiplator(df.max_epochs)}'
	title += f', patience: {tuner_utils.multiplator(df.patience)}'
	title += f' (\u0394={tuner_utils.multiplator(df.delta)})'
	title += '\ntuning: ' + tuner_utils.multiplator(df.tuning).replace('_', ' ')
	title += ', masking: ' if all(df.masked) else ' unmasked' if none(df.masked) else ', '
	title += tuner_utils.multiplator(df.masked_tuning_style) if any(df.masked == 'multiple') or any(df.masked) else ''
	title += ', ' + ('no punctuation' if all(df.strip_punct) else "with punctuation" if none(df.strip_punct) else 'multiple punctuation')
	title += f', {tuner_utils.multiplator(df.unfreezing)} unfreezing' if df.unfreezing.unique().size > 1 else ''
	if df.unfreezing.unique().size == 1 and df.unfreezing.unique()[0] == 'gradual':
		title += f' ({tuner_utils.multiplator(df.unfreezing_epochs_per_layer)})'
	
	title += ', mask args' if all(~np.isnan(df.mask_args)) else ''
	title += f', lr={tuner_utils.multiplator(df.lr)}'
	title += '\n'
	
	if 'args_group' in df.columns:
		title += f'args group: {tuner_utils.multiplator(df.args_group)}\n'
	
	return title

def create_metrics_plots(
	metrics: pd.DataFrame, 
	ignore_for_ylims: List[str], 
	dont_plot_separately: List[str]
) -> None:
	'''
	Plots metrics collected during fine-tuning
	
		params:
			metrics (pd.DataFrame)		: a dataframe containing information about metrics to plot
			ignore_for_ylims (list)		: label values to ignore when determining which metrics are alike for computing identical y-axes
			dont_plot_separately (list)	: metrics to not plot on separate plots (used to plot individual word data together with means in newverb expts)
	'''
	def get_metrics_plot_title(df: pd.DataFrame, metric: str) -> str:
		'''
		Get a title for the metrics plot
		
			params:
				df (pd.DataFrame)	: a dataframe containing metrics information
				metric (str)		: which metric is being plotted
			
			returns:
				title (str)			: the title for the plot
		'''
		def format_dataset_max_min(label: str, ser: pd.Series, r: int = 2) -> str:
			'''
			Get a string with the maximum (if applicable) and minimum values in the series
			Note that when r = 0, we are plotting patience, and so we don't include the maximum
			
				params:
					label (str)		: the name of the dataset
					ser (pd.Series)	: a series containing values for a metric in the dataset
					r (int)			: how much to round the max and min values
				
				returns:
					max_min (str)	: a formatted string containing information about the maximum (if applicable) and minimum values in ser
			'''
			subtitle = f'{label}: '
			
			format_str = f'{{m:.{r}f}}'
			
			# r == 0 when we are plotting patience,
			# if we are plotting patience, we don't plot the max since it's redundant
			if r != 0:
				subtitle += f'max @ {ser.idxmax()}: {format_str.format(m=ser.max())}, '
			
			subtitle += f'min @ {ser.idxmin()}: {format_str.format(m=ser.min())}\n'
		
			return subtitle
		
		title = f'{get_plot_title(df, metric)}\n'
		
		df = metrics.copy()
		if metric == 'remaining patience':
			df = df[df.metric.isin([metric, 'remaining patience overall'])]
		else:
			df = df[df.metric == metric]
		
		mean_dev = df[(~df.dataset.str.endswith('(train)')) & (df.dataset != 'overall')].copy()
		
		# if we only have one dev set, we don't need to report means or overall patience (since this is equivalent to the single dev set)
		num_dev_sets = mean_dev.dataset.unique().size
		
		if num_dev_sets > 1:
			if metric != 'remaining patience':
				mean_dev = mean_dev[['epoch', 'value']].groupby('epoch').value.agg('mean')
				title += format_dataset_max_min(label='mean dev', ser=mean_dev)
			else:
				overall = df[df.metric == 'remaining patience overall'][['epoch', 'value']].set_index('epoch')
				overall = overall.squeeze()
				title += format_dataset_max_min(label='overall', ser=overall, r=0)
		
		# drop the remaining patience overall to iterate through the dev sets
		df = df[df.metric == metric]
		
		for dataset in df.dataset.unique():
			dataset_values = df[(df.dataset == dataset) & (df.metric == metric)][['epoch', 'value']].set_index('epoch')
			dataset_values = dataset_values.squeeze()
			
			if not dataset_values.empty:
				kwargs = dict(label=dataset, ser=dataset_values)
				if metric == 'remaining patience':
					kwargs.update(dict(r=0))
				
				title += format_dataset_max_min(**kwargs)
		
		return title
	
	def get_like_metrics(
		metric: str, 
		ignore_strs: List[str], 
		all_metrics: Union[List[str],np.ndarray]
	) -> List[str]:
		'''
		Get the other metrics which are like metric but for different tokens so that we can
		set the axis limits to a common value. This is so we can compare the metrics visually
		for each token more easily
		
			params:
				metric (str)			: the metric to find like metrics for
				ignore_strs (str)		: which strings to ignore when determining whether metrics are alike
				all_metrics (list-like)	: a list of all the metrics to compare to metric
			
			returns:
				like_metrics (list)		: a list of the metrics like metric in all_metrics, when ignoring the strs in ignore_strs
		'''
		like_metrics = []
		for m in all_metrics:
			if not m == metric:
				m1 = metric
				m2 = m
				
				# we need to filter out cases where one of the ignore tokens occurs inside the other
				# which leads to issues when the smaller one gets replaced first. instead, we just
				# filter to the largest token to avoid the issue
				m1_ignore_strs = [t for t in ignore_strs if t in m1]
				m1_ignore_strs = [t for t in m1_ignore_strs if not any(t in ignore_str and len(ignore_str) > len(t) for ignore_str in m1_ignore_strs)]
				
				m2_ignore_strs = [t for t in ignore_strs if t in m2]
				m2_ignore_strs = [t for t in m2_ignore_strs if not any(t in ignore_str and len(ignore_str) > len(t) for ignore_str in m2_ignore_strs)]
				
				for token in m1_ignore_strs:
					m1 = m1.replace(token.upper(), '').replace(token.lower(), '') # do this to deal with both cased and uncased models
				
				for token in m2_ignore_strs:
					m2 = m2.replace(token.upper(), '').replace(token.lower(), '') # do this to deal with both cased and uncased models
				
				m1 = m1.strip()
				m2 = m2.strip()
				
				if m1 == m2:
					like_metrics.append(m)
		
		return like_metrics
	
	def setup_metrics_plot(metric: str, ignore_for_ylims: List[str]) -> Tuple:
		'''
		Initializes a metrics plot
		
			params:
				metric (str)											: which metric is being plotted?
				ignore_for_ylims (list)									: a list of strings to ignore when determining shared y-axis limits
			
			returns:
				df (pd.DataFrame)										: a dataframe with the information needed for plotting this metric
				palette (list)											: list of colors to use when plotting (we need one more color than datasets for the mean line)
				fig, ax (matplotlib.figure.Figure, matplotlib.axes.Axes): matplotlib plot objects
		'''
		df = metrics.copy()
		
		# we do this so that the metrics that are alike are plotted on the same scale
		like_metrics = get_like_metrics(metric=metric, ignore_strs=ignore_for_ylims, all_metrics=df.metric.unique())
		df = metrics[(metrics.metric.isin([metric, *like_metrics])) & (~metrics.value.isnull())].reset_index(drop=True)
		ulim = df.value.max()
		llim = df.value.min()
		
		adj = max(np.abs(ulim - llim)/40, 0.05)
		
		fig, ax = plt.subplots(1)
		fig.set_size_inches(9, 7)
		ax.set_ylim(llim - adj, ulim + adj)
		
		# do this manually so we don't recycle colors
		num_datasets = len(df[df.dataset != 'overall'].dataset.unique())+1 # add one for mean, which also gets used for overall
		palette = sns.color_palette(n_colors=num_datasets) if num_datasets <= 10 else sns.color_palette('hls', num_datasets) # if we have more than 10 dev sets, don't repeat colors
		sns.set_palette(palette)
		
		return df, palette, fig, ax
	
	if metrics.epoch.unique().size <= 1:
		log.warning('Not enough data to create line plots for metrics. Try fine-tuning for >1 epoch.')
		return
	
	metrics = metrics.copy()
	
	# for legend titles
	metrics.dataset = [dataset.replace('_', ' ') for dataset in metrics.dataset]
	
	# when plotting metrics for newverb experiments, we don't want separate plots for the individual arguments
	all_metrics = [m for m in metrics.metric.unique() if not 'overall' in m and not m in dont_plot_separately]
	
	xticks = determine_int_axticks(metrics.epoch)
	
	with PdfPages('metrics.pdf') as pdf:
		for metric in all_metrics:
			
			df, palette, fig, ax = setup_metrics_plot(metric=metric, ignore_for_ylims=ignore_for_ylims)
			
			common_kwargs = dict(
				x='epoch',
				y='value',
				ax=ax,
				hue='dataset',
				style='dataset_type',
				legend='full'
			)
			
			plot_cols = ['epoch', 'metric', 'value', 'dataset', 'dataset_type']
			
			dev_sets_df = df[(~df.dataset.str.endswith('(train)')) & (df.dataset != 'overall')].copy().reset_index(drop=True)
			num_dev_sets = dev_sets_df.dataset.unique().size
			
			if metric == 'remaining patience':
				if num_dev_sets > 1:
					overall = df[df.metric == 'remaining patience overall'][plot_cols].reset_index(drop=True)
					plot_data = pd.concat([df, overall])
				else:
					plot_data = df[df.metric == metric]
				
				yticks = determine_int_axticks(pd.concat([plot_data.value.astype(int), pd.Series(0)]))
				# ensure that we always get a 0 for remaining patience at the bottom
				plt.yticks(list(set([0] + yticks)))
			else:
				plot_data = df[df.metric == metric][plot_cols].copy().reset_index(drop=True)
			
			sns.lineplot(data=plot_data, **common_kwargs)
			
			# if we have more than one dev set, plot the mean + sd
			if num_dev_sets > 1:
				sns.lineplot(data=dev_sets_df[dev_sets_df.metric == metric], x='epoch', y='value', ax=ax, color=palette[-1], ci=68)
				ax.lines[-1].set_linestyle(':')
			
			dont_plot_sep_like_metrics = [m for m in dont_plot_separately if m in get_like_metrics(metric, ignore_strs=ignore_for_ylims, all_metrics=metrics.metric.unique())]
			dont_plot_sep_like_metrics = [m for m in dont_plot_sep_like_metrics if re.sub(r'\[(.*)\].*', '[\\1]', metric) in m]
			
			if any(dont_plot_sep_like_metrics):
				# this occurs when we're doing a newverb exp and we want to plot the individual tokens in addition to the overall mean
				tokens_df = df[df.metric.isin(dont_plot_sep_like_metrics)][plot_cols]
				
				if not tokens_df.empty:
					tokens_df['token'] = [re.sub(r'^([^\s]+).*', '\\1', m) for m in tokens_df.metric]
					
					v_adjust = (ax.get_ylim()[1] - ax.get_ylim()[0])/100
					common_kwargs.update(dict(linewidth=.5, legend=False, alpha=.3))
					text_kwargs = dict(size=6, horizontalalignment='center', verticalalignment='top', color='black', zorder=15, alpha=.3)
					
					for (token, dataset), token_dataset_df in tokens_df.groupby(['token', 'dataset']):
						token_dataset_df = token_dataset_df[~token_dataset_df.value.isnull()].reset_index(drop=True)
						sns.lineplot(data=token_dataset_df, **common_kwargs)
						
						x_text_pos = floor(token_dataset_df.epoch.max() * .8)
						y_text_pos = token_dataset_df[token_dataset_df.epoch == x_text_pos].value - v_adjust
						ax.text(x_text_pos, y_text_pos, token, **text_kwargs)
			
			ax.set_ylabel(metric)
			
			# remove redundant information from the legend
			# we can't do this directly using the get/set_title() methods due to some bad design in seaborn
			# (it treats titles incorrectly as subgroup labels)
			handles_labels = list(zip(*ax.get_legend_handles_labels()))
			handles_labels = [(handle, label) for handle, label in handles_labels if label not in ['dataset', 'dataset_type', 'train', 'dev', 'overall']]
			handles, labels = zip(*handles_labels)
			ax.legend(handles=handles, labels=labels, fontsize=9)
			
			plt.xticks(xticks)
			
			title = get_metrics_plot_title(df, metric)
			
			fig.suptitle(title)
			fig.tight_layout()
			
			# little hack here for when we have an extra line in the label in the new verb experiments
			fig.subplots_adjust(top=0.825-(num_dev_sets + (1 if 'args_group' in metrics.columns else 0))*.03125)
			
			pdf.savefig()
			plt.close()
			del fig

def create_cossims_plot(cossims: pd.DataFrame) -> None:
	'''
	Plot information about cosine similarities comparing predicted arguments to target groups
	
		params:
			cossims (pd.DataFrame): a dataframe containing cosine similarity data to plot
	'''
	def setup_cossims_plot(cossims: pd.DataFrame) -> Tuple:
		'''
		Initalize a cosine similarity plot
		
			params:
				cossims (pd.DataFrame)									: a dataframe containing cosine similarity data to plot
			
			returns:
				cossims (pd.DataFrame)									: the dataframe filtered to plot-relevant information
				cossim (str)											: the name of the column containing cosine similarity data (cossim, or cossim_mean)
				sem (str)												: the name of the column containing cosine similarity standard error data (cossim_sem)
				pairs (tuple)											: tuple of pairs of predicted arguments to compare
				fig, ax (matplotlib.figure.Figure, matplotlib.axes.Axes): matplotlib plot objects
		'''
		cossims = cossims[~cossims.target_group.str.endswith('most similar')].copy().reset_index(drop=True)
		if cossims.empty:
			log.info('No target groups were provided for cosine similarities. No comparison plots for cosine similarities can be created.')
			return
		
		if cossims.predicted_arg.unique().size <= 1:
			log.info(f'One or fewer predicted arguments were provided for cosine similarities ({cossims.target_group.unique()[0]}). No comparison plots for cosine similarities can be created.')
			return
		
		cossims, (cossim, sem), pairs = tuner_utils.get_data_for_pairwise_comparisons(cossims, cossims=True)
		
		fig, ax = plt.subplots(len(pairs), 2)
		ax = ax.reshape(len(pairs), 2)
		fig.set_size_inches(12.5, (6*len(pairs))+(0.6*cossims.predicted_arg.unique().size)+(0.6*cossims.target_group.unique().size)+0.25)
		
		return cossims, cossim, sem, pairs, fig, ax
	
	def get_cossims_plot_title(cossims: pd.DataFrame, cossim: str) -> str:
		'''
		Get a title for a cosine similarity plot
		
			params:
				cossims (pd.DataFrame)	: a dataframe containing information about the plot
				cossim (str)			: the name of the column containing cosine similarities (cossim or cossim_mean)
			
			returns:
				title (str)				: a title for the cosine similarity plot
		'''
		metric = 'cosine similarities to '
		metric += tuner_utils.multiplator(cossims.eval_data, multstr=f"{cossims.eval_data.unique().size} eval sets'")
		metric += f' target group tokens'
		
		title = get_plot_title(cossims, metric)
		
		if 'target_group_label' in cossims.columns:
			group_labels = cossims[['target_group', 'target_group_label']].drop_duplicates()
			group_labels = group_labels.groupby('target_group').apply(lambda x: x.to_dict(orient='records')[0]['target_group_label']).to_dict()
		else:
			group_labels = cossims.target_group.unique()
			group_labels = {g: g for g in group_labels}
		
		# add info about the averages
		for col1, col2 in itertools.permutations(['target_group', 'predicted_arg']):
			if cossims[col1].unique().size > 1:
				for group, df in cossims.groupby(col1):
					means = df.groupby(col2)[cossim].agg({'mean', 'sem', 'std', 'size'})
					out_group_means = means.loc[[i for i in means.index if not i == group]]
					for arg in out_group_means.index:
						mean1 	= means['mean'][group]
						sem1 	= means['sem'][group]
						std1 	= means['std'][group]
						size1 	= means['size'][group]
						
						mean2 	= out_group_means['mean'][arg]
						sem2 	= out_group_means['sem'][arg]
						std2 	= out_group_means['std'][arg]
						size2 	= out_group_means['size'][arg]
						
						label1 	= group_labels[group]
						if col1 == 'target_group':
							label2 = label1
						elif col1 == 'predicted_arg':
							label2 = group_labels[arg]
						
						diff_means = mean1 - mean2
						sem_diff_means = sqrt(((std1**2)/size1) + ((std2**2)/size2))
						
						title += (
							f'\nMean cosine similarity of {group} to {label1} \u2212 {arg} to {label2} targets: ' +
							f'{mean1:.4f} (\u00b1{sem1:.4f}) - {mean2:.4f} (\u00b1{sem2:.4f}) = ' +
							f'{diff_means:.4f} (\u00b1{sem_diff_means:.4f})'
						).replace('-', '\u2212') 
				
				title += '\n'
				
		return title
	
	cossims, cossim, sem, pairs, fig, ax = setup_cossims_plot(cossims)
	
	for i, pair in enumerate(pairs):
		in_token, out_token = tuner_utils.get_single_pair_data(cossims, pair, pair_col='predicted_arg', group='target_group_label')
		in_arg, out_arg = in_token.predicted_arg.unique()[0], out_token.predicted_arg.unique()[0]
		
		plot_args = dict(
			data=in_token, x=in_token, y=out_token, 
			val=cossim, sem=sem, ax=ax[i][0], 
			hue='target_group_label',
			legend_title='target group',
			aspect_ratio='eq_square',
			comparison_line=True,
			center_at_origin=False,
			marginal_means=['model_name','target_group_label'],
			xlabel=f'{in_arg} cosine similarity',
			ylabel=f'{out_arg} cosine similarity',
			plot_kwargs=dict(zorder=5, linewidth=0),
		)
		
		if cossims.model_name.unique().size == 1:
			plot_args.update(dict(
				text='token', 
				marginal_means=['target_group_label'],
				text_kwargs=dict(
					size=6, 
					horizontalalignment='center', 
					verticalalignment='top'
				)
			))
		
		scatterplot(**plot_args)
		
		# diffs plot, to show the extent to which the out group token is more similar to the target group tokens than the desired token
		plot_args.update(dict(
			ax=ax[i][1], diffs_plot=True,
			xlabel=f'{in_arg} cosine similarity',
			ylabel=f'{out_arg} - {in_arg} cosine similarity'.replace('-', '\u2212'),
		))
		
		scatterplot(**plot_args)
		
		# if we are plotting for multiple models, add names to the means of each model for each subplot
		# (we only want to do this for cossims plots, since they're the only ones that show the separate groups clearly)
		if cossims.model_name.unique().size > 1:
			model_means = cossims.groupby(['model_name', 'predicted_arg'])[cossim].agg('mean')
			model_means = model_means.reset_index().pivot_table(index='model_name', columns='predicted_arg')
			for plot_type, axis in zip(['xy', 'diffs'], ax[i]):
				for model_name, group in model_means.groupby('model_name'):
					x_pos = group.loc[model_name,cossim].loc[in_arg]
					y_pos = group.loc[model_name,cossim].loc[out_arg]
					
					if plot_type == 'diffs':
						y_pos -= x_pos
					
					axis.text(
						x_pos, y_pos, model_name, size=10, horizontalalignment='center', 
						verticalalignment='center', color='black', zorder=15, alpha=0.65, 
						fontweight='bold', path_effects=[pe.withStroke(linewidth=2, foreground='white')]
					)
	
	suptitle = get_cossims_plot_title(cossims, cossim)
	
	fig.suptitle(suptitle)
	fig.tight_layout()
	
	filename = tuner_utils.get_file_prefix(cossims) + '-cossims-plot.pdf'
	plt.savefig(filename)
	plt.close()
	del fig

def create_tsnes_plots(
	tsnes: pd.DataFrame, 
	components: List[str] = ['tsne1', 'tsne2']
) -> None:
	'''
	Plot 2 tsne components
	
		params:
			tsnes (pd.DataFrame)	: a dataframe containing tsnes to plot
			components (list)		: which 2 components in tsnes to plot
	'''
	def get_tsne_plot_title(df: pd.DataFrame, tsne_type: str = '') -> str:
		'''
		Get a title for the tsne plot
		
			params:
				df (pd.DataFrame)	: a dataframe containing information about tsnes
				tsne_type (str)		: what kind of tokens are being plotted on the tsnes (first n or target groups)
			
			returns:
				title (str)			: a title for the tsnes plot
		'''
		metric = f't-SNEs of {tsne_type} + novel token(s) (filtered)'
		title = get_plot_title(df, metric)
		return title
	
	file_prefix = tuner_utils.get_file_prefix(tsnes)
	
	tsnes = tsnes.copy()
	tsnes.token = tsnes.token.str.replace(chr(288), '')
	tsnes = tsnes.sort_values(by=['tsne_type', 'target_group_label', 'token'], key=lambda col: col.str.replace(r'^novel token$', '0', regex=True)).reset_index(drop=True)
	
	if len(components) > 2:
		log.warning('Only two t-SNE components can be plotted. The first two components provided will be used.')
		components = components[:2]
	
	# this is so we can use the custom plotting function
	# that expects different dfs for x and y with plotted points having identical column labels
	tsnes = tsnes.melt(id_vars=[c for c in tsnes.columns if not c in components], value_vars=components)
	
	xlabel = components[0].replace('tsne', 't-SNE ')
	ylabel = components[1].replace('tsne', 't-SNE ')
	
	with PdfPages(f'{file_prefix}-tsne-plots.pdf') as pdf:
		for tsne_type, df in tsnes.groupby('tsne_type'):
			fig, ax = plt.subplots(1)
			fig.set_size_inches(12, 10)
			
			tsne1, tsne2 = tuner_utils.get_single_pair_data(df, components, 'token', 'variable')
			
			scatterplot(
				data=tsne1, x=tsne1, y=tsne2, ax=ax,
				val='value', hue='target_group_label',
				legend_title='target group', text='token',
				text_size=dict(colname='token_category', existing=6, novel=10),
				xlabel=xlabel, ylabel=ylabel,
				text_kwargs=dict(horizontalalignment='center', verticalalignment='top')
			)
			
			title = get_tsne_plot_title(df, tsne_type=tsne_type)
			
			fig.suptitle(title)
			fig.tight_layout()
			
			pdf.savefig()
			plt.close()
			del fig	

def create_odds_ratios_plots(
	summary: pd.DataFrame,
	eval_cfg: DictConfig,
	plot_diffs: bool = False, 
	**kwargs: Dict
) -> None:
	'''
	Create plots of odds ratios or differences of odds ratios
	
		params:
			summary (pd.DataFrame)	: a summary containing information about odds ratios
			eval_cfg (DictConfig)	: a configuration for the experiment being plotted
			plot_diffs (bool)		: whether to plot improvements in odds ratios or just odds ratios
			**kwargs (Dict)			: passed to odds plot and then scatterplot
	'''
	
	def setup_odds_ratios_plot(data: pd.DataFrame, ratio_name: str, position_num: str) -> Tuple:
		'''
		Initialize an odds ratios plot
		
			params:
				data (pd.DataFrame)										: a dataframe containing information about the odds ratios to be plotted
				ratio_name (str)										: the name of the column containing the labels of the odds ratios
				position_num (str)										: the name of the column containing the labels of the odds ratios expressed in terms of argument positions
			
			returns:
				ratio_names_positions (list)							: a list of the unique pairs of ratio_names and positions in the data
				fig, ax (matplotlib.figure.Figure, matplotlib.axes.Axes): matplotlib plot objects
		'''
		# get number of linear positions (if there's only one position, we can't make plots by linear position)
		ratio_names_positions = data[[ratio_name, position_num]].drop_duplicates().reset_index(drop=True).to_records(index=False).tolist()
		ratio_names_positions = sorted(ratio_names_positions, key=lambda x: int(re.sub(r'position ([0-9]+)((\/.*)|$)', '\\1', x[1])))
		
		# this is true if plots by linear order will differ from plots by gf/role,
		# which means we want to construct a bigger canvas so we can plot them
		if len(ratio_names_positions) > 1 and not all(x_data[position_num] == y_data[position_num]):
			fig, ax = plt.subplots(2, 2)
			fig.set_size_inches(11, 13.45)
			ax = [axes for axeses in ax for axes in axeses] # much faster than using tuner_utils.flatten
		else:
			fig, ax = plt.subplots(1, 2)
			fig.set_size_inches(11, 9)
		
		return ratio_names_positions, fig, ax	
	
	def oddsplot(
		x: pd.DataFrame, 
		y: pd.DataFrame, 
		val: str, 
		sem: str,
		ax: matplotlib.axes.Axes,  
		exp_type: str,
		xlabel: str,
		ylabel: str,
		diffs_plot: bool = False, 
		pos_plot: bool = False,
		plot_kwargs: Dict = {}, line_kwargs: Dict = {},
		text_kwargs: Dict = {}, label_kwargs: Dict = dict(fontsize=8),
	) -> None:
		'''
		Main function for plotting odds ratios. Essentially does some transformations based on settings and calls scatterplot
		
			params:
				x (pd.DataFrame)			: the data containing odds ratios to plot on the x axis
				y (pd.DataFrame)			: the data containing odds ratios to plot on the y axis
				val (str)					: the name of the column containing the odds ratios to plot
				sem (str)					: the name of the column containing standard errors for val
				ax (matplotlib.axes.Axes)	: the object to create the plot on
				exp_type (str)				: the type of experiment for which odds ratios are being plotted
				xlabel (str)				: the x axis label
				ylabel (stn)				: the y axis label
				diffs_plot (bool)			: whether ax should be a plot of y = y - x
				pos_plot (bool)				: whether ax is a plot of linear order odds ratios as opposed to thematic role odds ratios
				plot_kwargs (dict)			: passed to scatterplot
				line_kwargs (dict)			: passed to scatterplot
				text_kwargs (dict)			: passed to scatterplot
				label_kwargs (dict)			: passed to scatterplot (default reduces fontsize)
		'''
		if diffs_plot:
			ylabel = f'Over{ylabel[0].lower() + ylabel[1:]}'
		
		center_at_origin = not diffs_plot
		
		labels = dict.fromkeys(x.ratio_name.unique())
		
		if pos_plot or (not pos_plot and exp_type == 'newarg'):
			legend_col = 'position_ratio_name' if pos_plot else 'role_position'
			for label in labels:
				addl_label = x[x.ratio_name == label][legend_col].unique()[0].split('/')[0]
				formatted_label = f'{label.split("/")[0]} args' if exp_type == 'newverb' else label
				labels[label] = f'{formatted_label} in {addl_label}' + (' position' if legend_col == 'role_position' else '')
		else:
			labels = {label: f'{label} position for {label.split("/")[0]} arguments' for label in labels}
		
		scatterplot(
			data=x, x=x, y=y,
			val=val, sem=sem, hue='ratio_name', ax=ax,
			xlabel=xlabel, ylabel=ylabel,
			comparison_line=True,
			center_at_origin=center_at_origin,
			diffs_plot=diffs_plot,
			aspect_ratio='eq_square',
			legend_title='',
			legend_labels=labels,
			plot_kwargs=plot_kwargs, line_kwargs=line_kwargs, 
			text_kwargs=text_kwargs, label_kwargs=label_kwargs,
		)
		
		return ax
	
	def get_odds_ratios_plot_title(
		summary: pd.DataFrame, 
		eval_cfg: DictConfig, 
		pair: Tuple[str], 
		x_data: pd.DataFrame, 
		y_data: pd.DataFrame, 
		odds_ratio: str
	) -> str:
		'''
		Get a title containing accuracy information for the odds ratios plot
		
			params:
				summary (pd.DataFrame)	: the data being plotted
				eval_cfg (DictConfig)	: the configuration of the experiment/evaluation
				pair (tuple)			: the pair for which odds ratios are being plotted
				x_data (pd.DataFrame)	: the data being plotted on the x axis
				y_data (pd.DataFrame)	: the data being plotted on the y axis
				odds_ratio (str)		: the name of the column in summary/x_data/y_data containing the odds ratios being plotted
		'''
		title 		= re.sub(r"\'\s(.*?)", f"' {', '.join(pair)} ", eval_cfg.data.description.replace('tuples', 'pairs')) + '\n'
		metric 		= 'odds ratios' if odds_ratio in ['odds_ratio' or 'odds_ratio_mean'] else 'odds ratios improvements'
		
		title 		+= get_plot_title(df=summary, metric=metric)
		
		pair_acc 	= [{**tuner_utils.get_accuracy_measures(x_data, y_data, odds_ratio), 'arg_type': 'any'}]
		for ratio_name in x_data.ratio_name.unique():
			pair_acc.append({
				**tuner_utils.get_accuracy_measures(
					x_data[x_data.ratio_name == ratio_name], 
					y_data[y_data.ratio_name == ratio_name], 
					odds_ratio
				), 
				'arg_type': ratio_name.split('/')[0]
			})
		
		if all('arg_type' in acc for acc in pair_acc):
			pair_acc = sorted(
				pair_acc, 
				key=lambda acc: '0' \
								if acc['arg_type'] == 'any' \
								else str(tuner_utils.GF_ORDER.index(f'[{acc["arg_type"].split("/")[0]}]') + 1) if f'[{acc["arg_type"]}]' in tuner_utils.GF_ORDER \
								else acc['arg_type']
			)
		
		subtitle = ''
		for acc in pair_acc:
			arg = acc['arg_type']
			prefix = 'overall' if arg == 'any' else arg
			perc_correct_str = (
				'\n' + prefix + ' acc' + 
				f', X\u2227Y: {round(acc["both_correct"], 2)}' +				# x and y
				f', X\u22bdY: {round(acc["both_incorrect"], 2)}' + 				# x nor y
				f', X\u00acY: {round(acc["ref_correct_gen_incorrect"], 2)}' + 	# x not y
				f', Y\u00acX: {round(acc["ref_incorrect_gen_correct"], 2)}' + 	# y not x
				f', Y|X: {round(acc["gen_given_ref"], 2)}' +	 				# x given y
				f', MSE: {round(acc["specificity_(MSE)"], 2)}' +		 		# mean squared error
				f' (\u00B1{round(acc["specificity_se"], 2)})'		 			# sem of mse
			)
			
			if arg == 'any':
				gen_given_ref_o_r = torch.tensor(y_data[y_data.index.isin(x_data[x_data[odds_ratio] > 0].index)][odds_ratio].tolist())
			else:
				gen_given_ref_o_r = torch.tensor(y_data[y_data.index.isin(x_data[(x_data[odds_ratio] > 0) & (x_data.ratio_name.str.startswith(f'{arg}'))].index)][odds_ratio].tolist())
			
			std, mean 				= torch.std_mean(gen_given_ref_o_r)
			perc_correct_str 		+= f', \u03BC Y|X: {mean:.2f}'
			if gen_given_ref_o_r.nelement() > 0:
				se 					= std/sqrt(len(gen_given_ref_o_r))
				perc_correct_str 	+= f' (\u00B1{se:.2f})'
			
			subtitle += perc_correct_str
		
		x_sentence_ex 	= tuner_utils.get_sentence_label(x_data)
		y_sentence_ex 	= tuner_utils.get_sentence_label(y_data)
		
		subtitle 		+= '\n\nX: ' + x_sentence_ex
		subtitle 		+= '\nY: ' + y_sentence_ex
		
		title 			+= subtitle
		
		return title
	
	def get_linear_order_plot_data(
		x_data: pd.DataFrame, 
		y_data: pd.DataFrame,
		odds_ratio_col: str, 
		ratio_names_positions: Tuple
	) -> Tuple[pd.DataFrame,str,str]:
		'''
		Get data used for constructing plots of odds ratios by linear order instead of by thematic role
		
			params:
				x_data (pd.DataFrame)			: the data to be plotted on the x axis
				y_data (pd.DataFrame)			: the data to be plotted on the y axis
				odds_ratio_col (str)			: which column contains the odds ratios to be plotted
				ratios_names_positions (tuple0)	: pairs of unique odds ratios of thematic roles and how they compare to odds ratios of positions
			
			returns:
				y_pos_data (pd.DataFrame)		: y_data formatted for the linear order plot (odds ratios are flipped if needed)
				xlabel (str)					: an x axis label containing information about how linear order corresponds to expected token positions
				ylabel (str)					: a y axis label containing information about how linear order corresponds to expected token positions
		'''
		y_pos_data = y_data.copy()
		xlabel = []
		ylabel = []
		
		for ratio_name, position in ratio_names_positions:
			x_idx = np.where(x_data.position_ratio_name == position)[0]
			y_idx = np.where(y_data.position_ratio_name == position)[0]
			
			x_expected = x_data.loc[x_idx, 'ratio_name'].unique()[0].split('/')[0]
			y_expected = y_data.loc[y_idx, 'ratio_name'].unique()[0].split('/')[0]
			
			x_position_label = x_data.loc[x_idx, 'position_ratio_name'].unique()[0].split('/')[0]
			y_position_label = y_data.loc[y_idx, 'position_ratio_name'].unique()[0].split('/')[0]
			
			xlabel.append(f'Expected {x_expected} in {x_position_label}')
			ylabel.append(f'Expected {y_expected} in {y_position_label}')
			
			# Flip the sign if the expected ratio isn't the same for x and y to get the correct values
			if not x_expected == y_expected:
				y_pos_data.loc[y_idx, odds_ratio_col] = -y_pos_data.loc[y_idx, odds_ratio_col]
		
		xlabel = '\n' + '\n'.join(sorted(xlabel, key=lambda item: int(re.sub(r'.*([0-9]+)$', '\\1', item)))) + '\n'
		ylabel = '\n' + '\n'.join(sorted(ylabel, key=lambda item: int(re.sub(r'.*([0-9]+)$', '\\1', item))))
		
		return y_pos_data, xlabel, ylabel
	
	ax_label = 'Confidence' if not plot_diffs else 'Improvement'
	arg_type = 'predicted_arg' if eval_cfg.data.exp_type == 'newarg' else 'arg_type' if eval_cfg.data.exp_type == 'newverb' else None
	
	summary, (odds_ratio, odds_ratio_sem), paired_sentence_types = tuner_utils.get_data_for_pairwise_comparisons(summary, eval_cfg=eval_cfg, diffs=plot_diffs)
	
	filename = tuner_utils.get_file_prefix(summary) + '-odds_ratios' + ('_diffs' if plot_diffs else '') + '-plots.pdf'
	
	with PdfPages(filename) as pdf:
		for pair in tqdm(paired_sentence_types):
			x_data, y_data 					= tuner_utils.get_single_pair_data(summary, pair, 'ratio_name')
			ratio_names_positions, fig, ax 	= setup_odds_ratios_plot(x_data, 'ratio_name', 'position_ratio_name')
			
			xlabel = f'{ax_label} in {pair[0]} sentences'
			ylabel = f'{ax_label} in {pair[1]} sentences'
			
			common_args = dict(
				x=x_data,
				y=y_data,
				val=odds_ratio, 
				sem=odds_ratio_sem,
				exp_type=eval_cfg.data.exp_type,
				xlabel=xlabel,
				ylabel=ylabel,
				**kwargs
			)
			
			oddsplot(ax=ax[0], **common_args)
			oddsplot(ax=ax[1], diffs_plot=True, **common_args)
			
			if len(ax) > 2:
				y_pos_data, pos_xlabel, pos_ylabel = get_linear_order_plot_data(x_data, y_data, odds_ratio, ratio_names_positions)
				
				pos_xlabel = xlabel + pos_xlabel
				pos_ylabel = ylabel + pos_ylabel
				
				common_args.update({
					'y': y_pos_data,
					'xlabel': pos_xlabel,
					'ylabel': pos_ylabel,
					'pos_plot': True
				})
				
				oddsplot(ax=ax[2], **common_args)
				oddsplot(ax=ax[3], diffs_plot=True, **common_args)
			
			title = get_odds_ratios_plot_title(summary, eval_cfg, pair, x_data, y_data, odds_ratio)
			
			fig.suptitle(title)
			fig.tight_layout()
			pdf.savefig()
			plt.close('all')
			del fig

def graph_results(
	summary: Dict, 
	eval_cfg: DictConfig
) -> None:
	'''
	Plot aconf and entropy (needs to be updated)
	
		params:
			summary (dict)			: a dictionary containing results information
			eval_cfg (DictConfig)	: the configuration of the experiment
	'''
	dataset = str(eval_cfg.data.name).split('.')[0]
	
	fig, axs = plt.subplots(2, 2, sharey='row', sharex='row', tight_layout=True)
	
	theme_entr = [x.item() for x in summary['theme']['entropy']]
	recip_entr = [x.item() for x in summary['recipient']['entropy']]
	
	inan = summary['theme']['animacy_conf']
	anim = summary['recipient']['animacy_conf']
	
	# Entropy Plots
	axs[0][0].hist(theme_entr)
	axs[0][0].axvline(np.mean(theme_entr), color='r')
	axs[0][0].set_title('entropy [theme]')
	
	axs[0][1].hist(recip_entr)
	axs[0][1].axvline(np.mean(recip_entr), color='r')
	axs[0][1].set_title('entropy [recipient]')
	
	# Animacy Plots
	
	axs[1][0].hist(inan)
	axs[1][0].axvline(np.mean(inan), color='r')
	axs[1][0].set_title('animacy confidence [theme]')
	
	axs[1][1].hist(anim)
	axs[1][1].axvline(np.mean(anim), color='r')
	axs[1][1].set_title('animacy confidence [recipient]')
	
	fig.suptitle(f"{eval_cfg.data.description}")
	
	plt.savefig(f"{dataset}.png")
	
	with open(f"{dataset}-scores.npy", "wb") as f:
		np.save(f, np.array(theme_entr))
		np.save(f, np.array(recip_entr))
		np.save(f, np.array(inan))
		np.save(f, np.array(anim))