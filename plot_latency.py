#!/usr/bin/env python
# coding: utf-8

# ## Generating histogram, CDF and HDR plots from histrogram data in .csv format
# * also generates sequence plot from sequence data in .csv format
# 
# ### Input format
# * histogram data in csv format with two columns
# * latency (in nanosecond)
# * occurence
# * e.g. as generated by MoonGen
# * example:
# ```
# 1663,1
# 1668,22
# 1669,76
# 1674,13
# 1675,930
# 1680,73
# 1681,449
# ```
# 
# ### Features
# * histogram, normalized histogram, CDF and HDR generation
# * optinal sequence plot generation
# * figures created in figures/*.tex
# * externalized data into data/*.tsv
# * TUMcolors supported
# * makefile to generate pdfs
# * same structure as expected by I8 thesis template
# * latency is converted to microsecond
# * histogram data is binned to microsecond resolution
# 
# ## You should not have to edit any of the following cells besides the last one
# * However you might want to tweak some plots manually
# 
# ## errors
# * if you get tex capacity exceeded when trying to compile the figures you have too many data points
#  * solution: less bins, by rounding more (e.g. 10 or 100 microsecond resolution)
#  * change: in to_microsecond change the dividend (from 1000 to 10000 or 100000)
#  * result: not microsecond resolution/bins but 10 or 100 microsecond
#  * dont forget to either update all axis labels or convert back to microsecond after binning

# In[ ]:


import os
import sys
import math
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from glob import glob
rprint=print
from pprint import pprint


# In[ ]:


# NOTE: tumcolors only work with python 3.6 and newer
from util.tumcolor import tumcolor_cycler
from util.i8_tikzplotlib import get_tikz_code, save_plt
from util.loop_plot import _plot_loop
from util.i8_model import add_model, add_error_plot,                           calculate_2nd_derivative, get_k_maxima, add_derivative_plot


# In[ ]:


# for command line invocation
def run_from_cli():
    import argparse

    parser = argparse.ArgumentParser(description='Generating plots from histogram data')
    parser.add_argument('basepath', metavar='BASEPATH', type=str,
                        help='Base path for all experiments')
    parser.add_argument('--histogram-filename', metavar='HIST_FILENAME', type=str, default='histogram.csv',
                        help='name of the histogram data file, wildcard possible')
    parser.add_argument('--sequence-filename', metavar='SEQ_FILENAME', type=str, default='',
                        help='name of the sequence data file, wildcard possible')
    parser.add_argument('--name', type=str, default='',
                        help='suffix for generated files, e.g. hdr-NAME.tex')
    parser.add_argument('path', metavar='PATH', type=str, nargs='+',
                        help='path to one or more csv file(s), will be RESULTS/<path>/HIST_FILENAME')
    parser.add_argument('--label', metavar='LABEL', type=str, action='append',
                        help='Nicer name for experiments')
    parser.add_argument('--round-ms-digits', metavar='ROUND', type=int, default=3,
                        help='Round to ROUND ms digits for binning')
    parser.add_argument('--histogram-bar-width', metavar='BAR_WIDTH', type=float, default=0.005,
                        help='Width for histogram bars')
    parser.add_argument('--loop-log-scale', action='store_true',
                        help='Log scale for loop plot x axis')

    args = parser.parse_args()
    if args.label and not len(args.label) == len(args.path):
        raise argparse.ArgumentTypeError('Must provide a label for either no or all paths')
        
    experiments = []
    if args.label:
        experiments = list(zip(args.path, args.label))
    else:
        experiments = args.path
        
    plot(experiments,
         basepath=args.basepath,
         histogram_file=args.histogram_filename,
         sequence_file=args.sequence_filename,
         name=args.name,
         round_ms_digits=args.round_ms_digits,
         histogram_bar_width=args.histogram_bar_width,
         
         loop_log_scale=args.loop_log_scale,
    )
        
    sys.exit()


# In[ ]:


def read_2c_csv(exp):
    data = dict()
    with open(exp) as infile:
        for line in infile:
            lat, occ = line.strip().split(',')
            data[int(lat)] = int(occ)
    return data


# In[ ]:


def to_microsecond(data, keys=True, values=False):
    if keys and values:
        return {k / 1000: v / 1000 for k, v in data.items()}
    if keys:
        return {k / 1000: v for k, v in data.items()}
    if values:
        return {k: v / 1000 for k, v in data.items()}
    
def to_ms_bins(data, round_ms_digits=3):
    binned = {}
    for k, v in data.items():
        rounded = round(k, round_ms_digits)
        if rounded not in binned:
            binned[rounded] = v
        else:
            binned[rounded] += v
    return binned

def to_expanded(data):
    expanded = []
    for val, occ in data.items():
        expanded += [val] * occ
    return expanded

def normalize(data):
    total = sum(data.values())
    percs = {k: (v/total) for k, v in data.items()}
    return percs

def accumulate(data):
    global curr
    curr = 0
    def acc(val): # just for the list comprehension
        global curr
        curr += val
        return curr
    return {k: acc(v) for k, v in sorted(data.items())}
    
def to_hdr(data):
    # treat negative (>1.0) and exact 1.0 values and very high values for v
    MAX_ACCURACY = 1000000000
    return {k: 1/(1-v) for k, v in data.items() if not (1-v) == 0.0 and not 1/(1-v) < 0 and not 1/(1-v) > MAX_ACCURACY}


# In[ ]:


def extract_hist_data(paths, basepath='/', histogram_file='histogram.csv', round_ms_digits=3, **kwargs):
    data = {}
    if not isinstance(paths, list):
        paths = [paths]
        
    print('Using {} digits for ms histogram'.format(round_ms_digits))
        
    for path in paths:
        name = None
        if not isinstance(path, tuple):
            name = path.replace('_', '-') # tex friendly path
        else:
            name = path[1]
            path = path[0]
            
        extended_path = os.path.join(basepath, path)
        experiment = os.path.join(extended_path, histogram_file)
        rprint('Processing ' + extended_path)
        
        subexperiments = glob(experiment)
        update_name = False
        base_name = name
        if len(subexperiments) > 1:
            update_name = True
        
        for exp in subexperiments:
            # replace everything that is not wildcard
            if not (basepath == '.' or basepath == '..'):
                histo = exp.replace(basepath, '')
            histo = histo.replace(path, '')
            histo = histo.replace(histogram_file, '')
            histo = histo.replace('//', '/')
            histo = histo[:-1]
            
            rprint('Subexperiment ' + histo)
            if update_name:
                name = base_name + histo
                
            # load data
            try:
                raw_data = read_2c_csv(exp)
            except FileNotFoundError as exce:
                rprint('Skipping - {}'.format(exce), file=sys.stderr)
                continue
                
            # different processing steps
            ms_data = to_microsecond(raw_data)
            hist_data = to_ms_bins(ms_data, round_ms_digits=round_ms_digits)
            box_data = to_expanded(ms_data)
            normalized_data = normalize(hist_data)
            accumulated_data = accumulate(normalized_data)
            hdr_data = to_hdr(accumulated_data)
            
            
            # store data
            data[name] = {}
            data[name]['hist'] = hist_data
            data[name]['hist_norm'] = normalized_data
            data[name]['cdf'] = accumulated_data
            data[name]['hdr'] = hdr_data
            data[name]['box'] = box_data

    return data

def extract_sequence_data(paths, basepath='/', sequence_file='sequence.csv', **kwargs):
    data = {}
    if not isinstance(paths, list):
        paths = [paths]

    for path in paths:
        name = None
        if not isinstance(path, tuple):
            name = path.replace('_', '-') # tex friendly path
        else:
            name = path[1]
            path = path[0]
            
        extended_path = os.path.join(basepath, path)
        experiment = os.path.join(extended_path, sequence_file)
        rprint('Processing ' + extended_path)
        
        subexperiments = glob(experiment)
        update_name = False
        base_name = name
        if len(subexperiments) > 1:
            update_name = True
        
        for exp in subexperiments:
            # remove basepath and filename from what we will use as label
            histo = exp.replace(basepath, '')
            histo = histo.replace(sequence_file, '')
            
            rprint('Subexperiment ' + histo)
            if update_name:
                name = base_name + histo
        
            # load data
            try:
                raw_data = read_2c_csv(exp)
            except FileNotFoundError as exce:
                rprint('Skipping - {}'.format(exce), file=sys.stderr)
                continue
            
            # different processing steps
            seq_data = to_microsecond(raw_data, keys=False, values=True)
            
            
            # store data
            data[name] = {}
            data[name]['seq'] = seq_data

    return data


# In[ ]:


def get_sorted_values(xs, ys, sort_by='xs'):
    # necessary for python <3.6
    if sort_by == 'xs':
        sort_by = 0
    else:
        sort_by = 1
    tup = zip(xs, ys)
    tup = sorted(tup, key=lambda x: x[sort_by])
    xs = [x for x,_ in tup]
    ys = [y for _,y in tup]
    return xs, ys

def plot_sequence(data, name='', **kwargs):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.set_prop_cycle(tumcolor_cycler)
    
    max_value = 0
    min_value = 1000000
    for exp, data in sorted(data.items()):
        hist = data['seq']
        xs = list(hist.keys())
        ys = list(hist.values())
        xs, ys = get_sorted_values(xs, ys)
        max_value=max(max_value, max(xs))
        min_value=min(min_value, min(xs))
        ax.plot(xs, ys, marker='o', markersize=1, linestyle='', label=exp)

    plt.ylim(bottom=0)
                
    ax.grid()
    ax.set(ylabel='Latency [$\mu$s]',
           xlabel='Number [-]')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.xlim(left=min_value)
    plt.xlim(right=max_value)
    
    save_plt('sequence', name=name)
    plt.show()


# In[ ]:


def plot_hist(data, name='', key='hist', ymax=None, ylabel='Occurence [-]',
              historgram_bar_width=0.005, **kwargs):
    fig, ax = plt.subplots(figsize=(9,6))
    ax.set_prop_cycle(tumcolor_cycler)
    
    print('Gathering data for histograms. If this takes too long, adjust round_ms_digits')
    
    max_value = 0
    data_points = 0
    for exp, data in sorted(data.items()):
        hist = data[key]
        xs = list(hist.keys())
        if key == 'hist':
            factor = 1
        else:
            # assume normalized
            factor = 100
        ys = [factor * val for val in hist.values()]
        if not ys:
            continue
        tup = zip(xs, ys)
        tup = sorted(tup, key=lambda x: x[0])
        xs, ys = get_sorted_values(xs, ys)
        data_points += len(ys)
        max_value=max(max_value, max(ys))
        ax.bar(xs, ys, width=historgram_bar_width, label=exp)

    print('Total amount of histogram of data points: {}'.format(data_points))
    
    if not ymax:
        ymax = max_value
    plt.ylim(bottom=0)
    plt.ylim(top=ymax)
                
    ax.grid()
    ax.set(ylabel=ylabel,
           xlabel='Latency [$\mu$s]')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.xlim(left=0)
    
    save_plt(key, name=name)
    plt.show()


# In[ ]:


def plot_cdf(data, name='', **kwargs):
    fig, ax = plt.subplots(figsize=(9,6))
    ax.set_prop_cycle(tumcolor_cycler)
    
    for exp, data in sorted(data.items()):
        cdf = data['cdf']
        xs = list(cdf.keys())
        ys = [100 * val for val in cdf.values()]
        xs, ys = get_sorted_values(xs, ys)
        ax.plot(xs, ys, label=exp)


    plt.ylim(bottom=0)
    plt.ylim(top=100)
                
    ax.grid()
    ax.set(ylabel='CDF [\%]',
           xlabel='Latency [$\mu$s]')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.xlim(left=0)
    
    save_plt('cdf', name=name)
    plt.show()


# In[ ]:


def plot_hdr(data, name='', **kwargs):
    fig, ax = plt.subplots(figsize=(9,6))
    ax.set_prop_cycle(tumcolor_cycler)
    
    max_value = 0
    min_value = 10000000000
    data_points = 0
    for exp, data in sorted(data.items()):
        hdr = data['hdr']
        xs = list(hdr.values())
        ys = list(hdr.keys())
        if not ys:
            continue
        xs, ys = get_sorted_values(xs, ys)
        data_points += len(ys)
        max_value=max(max_value, max(ys))
        min_value=min(min_value, min(ys))
        ax.plot(xs, ys, label=exp)
              
    print('Total amount of HDR data points: {}'.format(data_points))
            
    # automatically determine min/max based on min/max values log10
    log_max = pow(10, math.ceil(math.log10(max_value)))
    log_min = pow(10, math.floor(math.log10(min_value)))
    plt.ylim(bottom=log_min)
    plt.ylim(top=log_max)
                
    ax.grid()
    ax.set(xlabel='Percentile [\%] (log)',
           ylabel='Latency [$\mu$s] (log)')
    ax.set_xscale('log', subsx=[])
    ax.set_yscale('log')
    ticks = [1, 2, 10, 100, 1000, 10000, 100000, 1000000]
    labels = ["$0$", "$50$", "$90$", "$99$", "$99.9$", "$99.99$", "$99.999$", "$99.9999$"]
    plt.xticks(ticks, labels)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlim(left=1)
    # TODO determine xlim right

    save_plt('hdr', name=name)
    plt.show()


# In[ ]:


def plot_box(data, name='', **kwargs):
    fig, ax = plt.subplots(figsize=(9,6))
    ax.set_prop_cycle(tumcolor_cycler)
    
    boxes = []
    labels = []
    for exp, data in sorted(data.items()):
        values = data['box']
        boxes.append(values)
        labels.append(exp)
    ax.boxplot(boxes, showfliers=True, whis=1.5, labels=labels, patch_artist=True,
               medianprops=dict(color='TUMOrange'),
               boxprops=dict(facecolor='TUMWhite', color='TUMBlack'),
               
            )
            
    plt.ylim(bottom=0)
    plt.xticks(ticks=range(1, len(labels) + 1), labels=labels)
    plt.xlim(left=0.5, right=len(labels) + 0.5)
                
    ax.grid()
    ax.set(xlabel='',
           ylabel='Latency [$\mu$s]')

    save_plt('box', name=name)
    plt.show()


# In[ ]:


def get_nth_percentile(data, percentile):
    entries = len(data)
    if entries < get_accuracy(percentile):
        raise IndexError
    index = entries * percentile / 100
    index = int(index)
    return data[index-1]

def get_accuracy(f):
    # number of entries required by percentile
    # 99.99 -> 9999
    s = str(f)
    parts = s.split('.')
    if len(parts) == 1:
        return f
    total = len(parts[1])
    res = f * (10**total)
    return res

def plot_loop(name, content, mapping, hist_data, key=None,
              additional_plot_exports=None, only_direction=None, only_core_id=None,
              loop_log_scale=False, **kwargs):
    if not additional_plot_exports:
        additional_plot_exports = []
    if not key:
        key = [50]
    
    fig, ax = plt.subplots(figsize=(9,6))
    ax.set_prop_cycle(tumcolor_cycler)
    
    axis_label = None
    xss = {}
    yss = {}
    mapped = {}
    
    # gather data based on mapping
    for exp, run, type, label in content:
        axis_label = list(type.keys())[0]
        if label not in xss:
            xss[label] = []
            yss[label] = {}
        xss[label].append(list(type.values())[0])
        try:
            mapped[label] = hist_data[mapping[exp][run]]
        except KeyError as exce:
            continue
        else:
            data = mapped[label]
            for percentile in key:
                perc = 0
                try:
                    perc = get_nth_percentile(data['box'], percentile)
                except IndexError:
                    pass
                if not percentile in yss[label]:
                    yss[label][percentile] = []
                yss[label][percentile].append(perc)
        
    x_data = []
    y_data = []
    labels = []
    max_y_value = -1
    max_x_value = 0
    min_x_value = 1000000
    for exp, data in sorted(mapped.items()):
        for percentile in sorted(key):
            ys = yss[exp][percentile]
            xs = xss[exp]
            zipped = list(zip(xs, ys))
            zipped.sort(key=lambda tup: tup[0])
            xs, ys = zip(*zipped)
            x_data.append(xs)
            y_data.append(ys)
            labels.append(label)
            
            max_y_value = max(max_y_value, max(ys))
            max_x_value = max(max_x_value, max(xs))
            min_x_value = min(max_x_value, min(xs))
            
            ax.plot(xs, ys, marker='x', label = '{} {}%ile'.format(exp, percentile))
            
    if kwargs.get('model_parameters'):
        modelp = kwargs.get('model_parameters')
        # add model now that we made all configs in case of abort
        error_plot = []
        derivatives = []
        maximas = []
            
        num_models = len(key) * modelp.get('num_graphs', 1)
        for i in range(0, num_models):
            if not len(x_data) >= i + 1:
                break
            end = 0
            if modelp.get('end'):
                num = len(x_data[i])
                end = num - modelp.get('end')
        
            # calculate derivative
            derivative = calculate_2nd_derivative(x_data[i], y_data[i], log=loop_log_scale)
            maxima = get_k_maxima(derivative[2], k=modelp.get('maxima_k', 10),
                                  j=modelp.get('maxima_j', 0), max_index=end)
            maximas.append(maxima)
            derivative.append(maxima)
            derivatives.append((x_data[i], derivative, i))
            
            error_plot += add_model(x_data, y_data, ax, modelp,
                                    maximum=max_y_value * 1.2, graph=i, end=end, maxima=maxima)

    plt.ylim(bottom=0)
    plt.xlim(left=min_x_value)
    plt.xlim(right=max_x_value)
    if loop_log_scale:
        plt.xscale('log')
                
    ax.grid()
    ax.set(ylabel='Latency [$\mu$s]',
           xlabel=axis_label)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    long_key = '_'.join([str(p) for p in key])
    
    save_plt('loop_{}'.format(long_key), name=name)
    for ape in additional_plot_exports:
        rprint('Additional export as {}'.format(ape))
        savefig('figures/{}_loop_{}.{}'.format(name, long_key, ape), format=ape)
    plt.show()
    
    if kwargs.get('model_parameters'):
        add_error_plot(plt, error_plot, name, key=key, log_scale=loop_log_scale, axis_label=axis_label,
                       additional_plot_exports=additional_plot_exports)
        
        add_derivative_plot(plt, derivatives, name, key=key, log_scale=loop_log_scale, axis_label=axis_label,
                            additional_plot_exports=additional_plot_exports)


# In[ ]:


def _plot_sequence(paths, **kwargs):
    print('------------- plotting sequence data ------------')
    seq_data = extract_sequence_data(paths, **kwargs)
    if not seq_data:
        rprint('No sequence data found', file=sys.stderr)
    else:
        plot_sequence(seq_data, **kwargs)
        
def _plot_default_histogram(hist_data, histogram_plots=True, **kwargs):
    print('------------ plotting default histogram data ----------')
    # different plot types for histogram data
    if histogram_plots:
        plot_hist(hist_data, **kwargs)
        plot_hist(hist_data, key='hist_norm', ylabel='Occurence [\%]', **kwargs)
    else:
        print('skipping histograms')
    plot_box(hist_data, **kwargs)
    plot_cdf(hist_data, **kwargs)
    plot_hdr(hist_data, **kwargs)


# In[ ]:


def plot(paths, **kwargs):
    
    if kwargs.get('sequence_file'):
        _plot_sequence(paths, **kwargs)
    
    if kwargs.get('histogram_file'):
        # histogram data
        hist_data = extract_hist_data(paths, **kwargs)
        if not hist_data:
            rprint('No histogram data found', file=sys.stderr)
            return
        
        if kwargs.get('default_plots'):
            _plot_default_histogram(hist_data, **kwargs)
            
    if not kwargs.get('percentiles'):
        print('you need to define the percentiles of interest as list of lists')
        return

    if kwargs.get('loop_file'):
        _plot_loop(paths, hist_data, plot_loop, metrics=kwargs.get('percentiles'), **kwargs)


# In[ ]:


# this will only be triggered if invoked from command-line
if not sys.argv[0].endswith('ipykernel_launcher.py'):
    run_from_cli()


# # Make your edits in the cell below

# In[ ]:


# configuration
configuration = None
# templated from generator script
#$configuration
if not configuration:
    # default for testing
    configuration = {'convert_to_cycles': 2.0,
 'dut': 'cesis',
 'experiment_name': 'meta_field_writes',
 'latency_rates': [0.1, 0.5, 0.7],
 'loadgen': 'nida',
 'log_scale': '',
 'loop_plot_per': ['packet_size'],
 'loop_x_axis': 'meta_field_writes',
 'model_end': 0,
 'model_parts': [1, 2],
 'model_start': 0,
 'only_core_id': 1,
 'perf_stat_events': [['r08d1', 'L1_cache_misses'],
                      ['r10d1', 'L2_cache_misses'],
                      ['r20d1', 'L3_cache_misses']],
 'repetitions': 3,
 'result_dir_file': '/home/scholzd/component_benchmarking/experiments/p4_tapas/other/meta_field_writes/result_directory.txt',
 'target': 'p4_t4p4s'}
    
    
result_dir = None
with open(configuration['result_dir_file'], 'r') as fh:
    # one per row, filter duplicates and empty last line
    result_dirs = sorted(list(set(fh.read().split('\n')[:-1])))
    
import socket
hostname = socket.gethostname()
print('Results on node ' + hostname)
    
print('Found result directories')
pprint(result_dirs)

print('Using configuration:')
pprint(configuration)


# In[ ]:


result_dir = result_dirs[-1]
print('Result data in:')
print(result_dir)

RESULTS=os.path.join(result_dir, configuration['loadgen'])
LOOP_FILENAME = '*_measurement_run*.loop'

model_parameters = {
    'parts': configuration['model_parts'],
    'bruteforce': [2], 
    'end': configuration['model_end'],
    'start': configuration['model_start'],
    
    'num_graphs': 1,
    'show_best_x': 1,
    
    'maxima_k': 10,
    'maxima_j': 0,
    
    'gamma': 1e-05,
    'comp_method': 'sMAPE',
    'epsilon': 0.005,
    'epsilon_rel': 0.005,
}

for latency_rate in configuration['latency_rates']:
    HISTOGRAM_FILENAME = 'histogram-{}_run*.csv'.format(latency_rate)
    plot([
          ('', configuration['experiment_name']),
         ],
         basepath=RESULTS,
         name='{}_{}'.format(configuration['target'], latency_rate),
         histogram_file=HISTOGRAM_FILENAME,
        
         percentiles=[[0, 50], [25, 75], [100]],
         default_plots=False,
         histogram_plots=False,
         round_ms_digits=1,
        
         convert_to_cycles=configuration['convert_to_cycles'],
        
         loop_file=LOOP_FILENAME,
         loop_x_axis=configuration['loop_x_axis'],
         loop_plot_per=configuration['loop_plot_per'],
         loop_log_scale=configuration['log_scale'],
        
         model_parameters=model_parameters,
    )


# In[ ]:




