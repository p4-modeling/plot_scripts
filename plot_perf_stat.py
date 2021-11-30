#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import csv
from pprint import pprint #as print
rprint=print
#from pprint import pprint as print
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig


# In[ ]:


# NOTE: tumcolors only work with python 3.6 and newer
from util.tumcolor import tumcolor_cycler
from util.i8_tikzplotlib import save_plt, latex_save
from util.loop_plot import _plot_loop
from util.i8_model import add_model, convert_to_cpu_cycles, add_error_plot


# In[ ]:


def read_perf_stat(exp, perf_stat_file, repeat=1):
    max_value = 0
    max_idx = 0
    data = {}
    for rep in range(1, repeat + 1):
        exp = exp.replace('stat_1', 'stat_{}'.format(rep))
        with open(exp) as infile:
            content = csv.DictReader(infile, delimiter=',',
                                     fieldnames=['count', '-1', 'event', 'duration',
                                                 'percentage', '-2', '-3'])
            for line in content:
                event = line['event']
                # to seconds
                factor = int(line['duration']) / 1000000000
                count = int(line['count'])
                # to 1/s
                normalized = count / factor
                if not event in data:
                    data[event] = []
                data[event].append(normalized)
                
    # average for all values and repetitions
    result = {}
    for event, counts in data.items():
        result[event] = sum(counts) / len(counts)
                
    return result


# In[ ]:


def add_values(data, prefix, func):
    for cid, data2 in data.items():
        for direction, data3 in data2.items():
            for item in MOONGEN_DATA_OUTPUT:
                data[cid][direction][prefix + '_' + item] = func(data3[item])
                
def add_progression_x_value(progression_mapping_function, exp, data):
    for cid, data2 in data.items():
        for direction, data3 in data2.items():
            data[cid][direction]['x_value'] = progression_mapping_function(exp, data)
                
def add_packet_loss(data, only_core_id=1, only_direction='rx'):
    global found_loss
    global counter
    sent = 0
    recv = 0
    for cid, data2 in data.items():
        for direction, data3 in data2.items():
            if direction == 'tx':
                sent = data3['max_mpps']
            else:
                recv = data3['max_mpps']
    loss = sent - recv
    data[only_core_id][only_direction]['loss'] = loss


# In[ ]:


def extract_perf_stat_data(paths, basepath='/', perf_stat_file='perf_stat.stderr',
                           repeat=1):
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
        #rprint('Processing ' + extended_path)
        experiment = os.path.join(extended_path, perf_stat_file)
        #rprint(experiment)
        
        subexperiments = glob(experiment)
        update_name = False
        base_name = name
        if len(subexperiments) > 1:
            update_name = True
        
        for exp in sorted(subexperiments):
            # replace everything that is not wildcard
            if not (basepath == '.' or basepath == '..'):
                histo = exp.replace(basepath, '')
            histo = histo.replace(path, '')
            histo = histo.replace(perf_stat_file, '')
            histo = histo.replace('//', '/')
            histo = histo[:-1]
            
            #rprint('Subexperiment ' + histo)
            if update_name:
                name = base_name + histo
                
            # load data
            try:
                raw_data = read_perf_stat(exp, perf_stat_file, repeat=repeat)
            except FileNotFoundError as exce:
                rprint('Skipping {} - {}'.format(histo, exce), file=sys.stderr)
                continue
                
            # store data
            data[name] = {}
            data[name]['stat'] = raw_data

    return data


# In[ ]:


def plot_events(data, name='', additional_plot_exports=None):
    if not additional_plot_exports:
        additional_plot_exports = []
        
    fig, ax = plt.subplots(figsize=(9,6))
    ax.set_prop_cycle(tumcolor_cycler)
    
    max_x_value = 0
    grouped = {}
    for exp, data in sorted(data.items()):
        for event, data in data['stat'].items():
            if not event in grouped:
                grouped[event] = []
            grouped[event].append(data)
            
    
    for event, data in sorted(grouped.items()):
        xs = range(len(data))
        ys = data
        
        max_x_value = len(data) - 1
        
        label = event
        ax.plot(xs, ys, label=label)

    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.xlim(right=max_x_value)
                
    ax.grid()
    ax.set(ylabel='Events [-]',
           xlabel='Experiment [-]')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    save_plt("events", name=name)
    for ape in additional_plot_exports:
        rprint('Additional export as {}'.format(ape))
        savefig('figures/{}_{}.{}'.format(name, "events", ape), format=ape)
    plt.show()


# In[ ]:


def plot_loop(name, content, mapping, perf_data, key='event', additional_plot_exports=None,
              log_scale=False,
              only_direction="", only_core_id="", convert_to_cycles="",
              model_parts=[], model_bruteforce=[], model_start=0, model_end=0,
              model_gamma=1e-05, model_comp_method='sMAPE',
              model_epsilon=0.005, model_epsilon_rel=0.05,
              model_show_best_x=1, model_num_graphs=1,
              model_maxima_k=8, model_maxima_j=0):
    if not additional_plot_exports:
        additional_plot_exports = []
        
    fig, ax = plt.subplots(figsize=(9,6))
    ax.set_prop_cycle(tumcolor_cycler)
    
    event = key[0]
    event_clear = key[1]
    
    axis_label = None
    xss = {}
    yss = {}
    mapped = {}
    
    num_models = 2
    
    # gather data based on mapping
    for exp, run, type, label in content:
        axis_label = list(type.keys())[0]
        if label not in xss:
            xss[label] = []
            yss[label] = {}
        xss[label].append(list(type.values())[0])
        try:
            mapped[label] = perf_data[mapping[exp][run]]
        except KeyError as exce:
            continue
        else:
            mapped[label] = mapped[label]['stat']
            y = mapped[label][event]
            full = event
            try:
                yss[label][full].append(y)
            except KeyError:
                yss[label][full] = [y]
                    
    max_x_value = 0
    min_x_value = 1000000
    for exp, data in sorted(mapped.items()):
        label = exp
        full = event
        ys = yss[exp][full]
        xs = xss[exp]
        zipped = list(zip(xs, ys))
        zipped.sort(key=lambda tup: tup[0])
        xs, ys = zip(*zipped)
        
        max_x_value = max(max_x_value, max(xs))
        min_x_value = min(max_x_value, min(xs))
        
        ax.plot(xs, ys, marker='x', label=latex_save(label))
    
    # process x axis label
    axis_label = ' '.join([part.capitalize() for part in axis_label.split('_')])
    
    # axis limits
    plt.ylim(bottom=0)
    plt.xlim(left=min_x_value)
    plt.xlim(right=max_x_value)
    if log_scale:
        plt.xscale('log')
        axis_label += ' [log]'
                
    ax.grid()
    ax.set(ylabel='Events [-]',
           xlabel=axis_label)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    save_plt('loop_event_{}'.format(event_clear), name=name)
    for ape in additional_plot_exports:
        rprint('Additional export as {}'.format(ape))
        savefig('figures/{}_loop_event_{}.{}'.format(name, event_clear, ape), format=ape)
    plt.show()


# In[ ]:


def plot(paths, name=None, repeat=0,
         perf_stat_file=None,
         default_plots=True, additional_plot_exports=None, metrics=None,
         loop_file=None, loop_x_axis=None, loop_plot_per=None,  loop_log_scale=False,
         **kwargs):
    
    # extract throughput data
    perf_data = extract_perf_stat_data(paths, perf_stat_file=perf_stat_file,
                                       repeat=repeat,
                                       **kwargs)
    
    if not perf_data:
        rprint('No perf stat data found', file=sys.stderr)
        return
    
    if default_plots:
        plot_events(perf_data, name, additional_plot_exports)
    
    if not metrics:
        print('you need to define the metrics of interest as tuple (METRIC, CLEAR_NAME)')
        return
        
    if (loop_file and not loop_x_axis) or (loop_x_axis and not loop_file):
        raise RuntimeError('must define loop_file AND loop_x_axis if using loop variables')
    if loop_file:
        _plot_loop(paths, name, perf_data, loop_file, loop_x_axis, loop_plot_per, metrics, plot_loop,
                   additional_plot_exports, "", "", loop_log_scale, "",
                   **kwargs)


# # Make your edits in the cell below

# In[ ]:


def argument_parser():
    import argparse

    parser = argparse.ArgumentParser(description='run throughput script')
    parser.add_argument('-l', '--local-data', action='store_true',
                        help='Run with local data')
    
    args = parser.parse_args()
    return args


# In[ ]:


# argument parser
args = None
if not sys.argv[0].endswith('ipykernel_launcher.py'):
    args = argument_parser()
 
local_data = args and args.local_data or False

# configuration
configuration = None
# templated from generator script
#$configuration
if not configuration:
    # default for testing
    configuration = {'convert_to_cycles': 'from_loop',
 'dut': 'cesis',
 'experiment_name': 'cpu_frequency',
 'latency_rates': [0.1, 0.5, 0.7],
 'loadgen': 'nida',
 'log_scale': '',
 'loop_plot_per': ['packet_size'],
 'loop_x_axis': 'cpu_frequency',
 'model_end': 0,
 'model_parts': [1, 2],
 'model_start': 0,
 'only_core_id': 1,
 'perf_stat_events': [['r08d1', 'L1_cache_misses'],
                      ['r10d1', 'L2_cache_misses'],
                      ['r20d1', 'L3_cache_misses']],
 'repetitions': 3,
 'result_dir_file': '/home/scholzd/component_benchmarking/experiments/p4_tapas/baseline/cpu_frequency/result_directory.txt',
 'target': 'p4_t4p4s'}
    
print('Using local data: ' + str(local_data))
# in this case, modify result_dir_file and result_dirs with local path in data repository
if local_data:
    configuration['result_dir_file'] = '../result_directory.txt'
    
result_dir = None
with open(configuration['result_dir_file'], 'r') as fh:
    # one per row, filter duplicates and empty last line
    result_dirs = sorted(list(set(fh.read().split('\n')[:-1])))
    

if local_data:
    result_dirs = [os.path.join('../data/', result_dir.split('/')[-1]) for result_dir in result_dirs]
    
if not local_data:
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

RESULTS=os.path.join(result_dir, configuration['dut'])
PERF_STAT_FILENAME = 'perf_stat_1_run*.stderr'
LOOP_FILENAME = '*_measurement_run*.loop'

plot([
      ('', configuration['experiment_name']),
     ],
     basepath=RESULTS,
     name=configuration['target'],
     perf_stat_file=PERF_STAT_FILENAME,
    
     metrics=configuration['perf_stat_events'],
     default_plots=False,
     repeat=configuration['repetitions'],
     
     loop_file=LOOP_FILENAME,
     loop_x_axis=configuration['loop_x_axis'],
     loop_plot_per=configuration['loop_plot_per'],
     loop_log_scale=configuration['log_scale'],
)


# In[ ]:




