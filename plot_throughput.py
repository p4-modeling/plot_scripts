#!/usr/bin/env python
# coding: utf-8

# ## Generating throughput and packet rate plots from MoonGen data
# 
# ### Input format
# * MoonGen stdout or csv (tx and rx split to different files)
# * for stdout: line every second containing the RX or TX data per core
# ```
# ...
# [Packets counted] RX: 0.10 Mpps, 51 Mbit/s (67 Mbit/s with framing)
# [Device: id=0] TX: 0.10 Mpps, 51 Mbit/s (67 Mbit/s with framing)
# [Packets counted] RX: 0.10 Mpps, 51 Mbit/s (67 Mbit/s with framing)
# [Device: id=0] TX: 0.10 Mpps, 51 Mbit/s (67 Mbit/s with framing)
# ...
# ```
# * for csv respectively
# 
# ### Features
# * three types of plotting
#     * plots data for all individual files
#         * througput (with and without framing), packet rate
#     * plots progression over all individual files
#         * min, max, avg of the above
#         * requires definition of mapping function
#         * mapping function maps result of file to x axis value
#     * plots loop experiment
#         * define the order of loop variables
# * figures created in figures/*.tex
# * externalized data into data/*.tsv
# * TUMcolors supported
# * makefile to generate pdfs
# * same structure as expected by I8 thesis template
# 
# ## You should not have to edit any of the following cells besides the last one
# * However you might want to tweak some plots manually
# 
# ## errors
# * if you get tex capacity exceeded when trying to compile the figures you have too many data points
#     * usually happens for the first plot type as it generates a slope per measurement run
# 

# In[ ]:


import os
import sys
import csv
from pprint import pprint
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
from util.i8_model import add_model, convert_to_cpu_cycles, add_error_plot,                           calculate_2nd_derivative, get_k_maxima, add_derivative_plot


# In[ ]:


MOONGEN_DATA_OUTPUT = ['mpps', 'mbit', 'mbitcrc']

class ParsingError(Exception):
    pass

def read_moongen_stdout(exp, strip):
    data = dict()
    valid_file = dict()
    with open(exp) as infile:
        for line in infile:
            # filter unwanted lines
            if not (('[Packets counted]' in line or '[Device: id=' in line) and ('RX' in line or 'TX' in line)):
                continue
                
            # check if we reached the last lines containing the summary
            summary = False
            if 'StdDev' in line and 'total' in line:
                summary = True
                
            cid = 0
            direction = 'rx'
            mpps = 0 
            mbit = 0
            mbitcrc = 0
            
            parts = line.split('] ')            
            # get ID
            if parts[0].endswith('Packets counted'):
                #TODO does this make sense?
                cid = 0
            else:
                cid = int(parts[0].split('=')[-1])
                
            # get direction
            parts = parts[1].split(' ')
            if parts[0].startswith('RX'):
                direction = 'rx'
            elif parts[0].startswith('TX'):
                direction = 'tx'
            else:
                raise ValueError('Unable to parse direction: {}'.format(line))
            
            # prepare structure
            if not cid in data:
                data[cid] = dict()
            if not direction in data[cid]:
                data[cid][direction] = dict()
            for item in MOONGEN_DATA_OUTPUT:
                if not item in data[cid][direction]:
                    data[cid][direction][item] = list()
                
            # get other data
            if not summary:
                mpps = float(parts[1])
                mbit = float(parts[3])
                mbitcrc = float(parts[5][1:])
                
                data[cid][direction]['mpps'].append(mpps)
                data[cid][direction]['mbit'].append(mbit)
                data[cid][direction]['mbitcrc'].append(mbitcrc)
                valid_file[direction] = True
            else:
                mpps = float(parts[1])
                mbit = float(parts[5])
                mbitcrc = float(parts[9][1:])
                
                data[cid][direction]['avg_mg_mpps'] = mpps
                data[cid][direction]['avg_mg_mbit'] = mbit
                data[cid][direction]['avg_mg_mbitcrc'] = mbitcrc
                
                # strip from head and tail of data
                if strip:
                    data[cid][direction]['mpps'] = data[cid][direction]['mpps'][strip:-(strip+1)]
                    data[cid][direction]['mbit'] = data[cid][direction]['mbit'][strip:-(strip+1)]
                    data[cid][direction]['mbitcrc'] = data[cid][direction]['mbitcrc'][strip:-(strip+1)]
                
                # add self calculated averages with skips as default
                data[cid][direction]['avg_mpps'] = np.mean(data[cid][direction]['mpps'])
                data[cid][direction]['avg_mbit'] = np.mean(data[cid][direction]['mbit'])
                data[cid][direction]['avg_mbitcrc'] = np.mean(data[cid][direction]['mbitcrc'])
                
                valid_file[direction + '_summary'] = True
        if not len(valid_file.keys()) == 4:
            raise ParsingError('Invalid file: {}'.format(valid_file))
                
    return data

def read_moongen_csv(exp, throughput_file, strip, repeat=1, only_core_id=1, only_direction='rx'):
    
    # split location, replace tx with rx and vice versa
    parts = [p for p in throughput_file.split('*') if 'rx' in p or 'tx' in p]
    if not len(parts) == 1:
        print('unknown throughput_file format, must include exactly one of tx or rx in name')
    throughput_file = parts[0]
        
    
    if 'tx' in throughput_file:
        throughput_file_2 = throughput_file.replace('tx', 'rx')
    elif 'rx' in throughput_file:
        throughput_file_2 = throughput_file.replace('rx', 'tx')
    exps = [exp, exp.replace(throughput_file, throughput_file_2)]
    
    max_value = 0
    max_idx = 0
    datas = []
    for rep in range(1, repeat + 1):
        data = {}
        for exp in exps:
            exp = exp[:-1] + str(rep)
            with open(exp) as infile:
                content = csv.DictReader(infile, delimiter=',')
                for line in content:
                    # no core id, use device id
                    cid = int(line['Device'].split('=')[1])
                    direction = line['Direction'].lower()
                    mpps = float(line['PacketRate'])
                    mbit = float(line['Mbit'])
                    mbitcrc = float(line['MbitWithFraming'])
                    
                    if not cid in data:
                        data[cid] = dict()
                    if not direction in data[cid]:
                        data[cid][direction] = dict()
                    for item in MOONGEN_DATA_OUTPUT:
                        if not item in data[cid][direction]:
                            data[cid][direction][item] = list()
                            
                    data[cid][direction]['mpps'].append(mpps)
                    data[cid][direction]['mbit'].append(mbit)
                    data[cid][direction]['mbitcrc'].append(mbitcrc)
        
        # add average values
        for cid, dat in data.items():
            for direction, dat in dat.items():
                for t in MOONGEN_DATA_OUTPUT:
                    data[cid][direction]['avg_' + t] = sum(dat[t]) / len(dat[t])
                    # also update max value for later
                    if cid == only_core_id and direction == only_direction and t == 'mpps':
                        if data[cid][direction]['avg_' + t] > max_value:
                            max_value = data[cid][direction]['avg_' + t]
                            max_idx = rep
    
                # strip from head and tail of data
                if strip:
                    data[cid][direction]['mpps'] = data[cid][direction]['mpps'][strip:-(strip+1)]
                    data[cid][direction]['mbit'] = data[cid][direction]['mbit'][strip:-(strip+1)]
                    data[cid][direction]['mbitcrc'] = data[cid][direction]['mbitcrc'][strip:-(strip+1)]
        datas.append(data)
                
    # only return repetition of maximum value
    return datas[max_idx - 1]


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


def extract_tp_data(paths, basepath='/', throughput_file='histogram.csv', throughput_location='stdout',
                    throughput_strip=0, progression_mapping_function=None,
                    repeat=1, only_direction='rx', only_core_id=1, **args):
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
        rprint('Processing ' + extended_path)
        experiment = os.path.join(extended_path, throughput_file + '_1')
        
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
            histo = histo.replace(throughput_file, '')
            histo = histo.replace('//', '/')
            histo = histo[:-1]
            
            #rprint('Subexperiment ' + histo)
            if update_name:
                name = base_name + histo
                
            # load data
            try:
                if throughput_location == 'stdout':
                    raw_data = read_moongen_stdout(exp, throughput_strip)
                elif throughput_location == 'split':
                    raw_data = read_moongen_csv(exp, throughput_file, throughput_strip,
                                                repeat=repeat, only_direction=only_direction,
                                                only_core_id=only_core_id)
                else:
                    print('unknown throughput_location')
                    return
            except (FileNotFoundError, ParsingError) as exce:
                rprint('Skipping {} - {}'.format(histo, exce), file=sys.stderr)
                continue
                
            # different processing steps
            add_values(raw_data, 'max', max)
            add_values(raw_data, 'min', min)
            add_packet_loss(raw_data, only_direction=only_direction, only_core_id=only_core_id)
            if progression_mapping_function:
                add_progression_x_value(progression_mapping_function, exp, raw_data)
            
            # store data
            data[name] = {}
            data[name]['tp'] = raw_data

    return data


# In[ ]:


def plot_rate(data, name='', key='mbit', ylabel='Throughput [Mbit/s]', additional_plot_exports=None,
              only_direction=None, only_core_id=None):
    if not additional_plot_exports:
        additional_plot_exports = []
        
    fig, ax = plt.subplots(figsize=(9,6))
    ax.set_prop_cycle(tumcolor_cycler)
    
    max_x_value = 0
    for exp, data in sorted(data.items()):
        data = data['tp']
        for cid, data in data.items():
            if only_core_id and not cid == only_core_id:
                continue
            for direction, data in data.items():
                if only_direction and not direction == only_direction:
                    continue
                data = data[key]
                
                xs = range(len(data))
                ys = data
                
                max_x_value = len(data) - 1
                
                label = exp
                if not only_core_id:
                    label += ' - c' + str(cid)
                if not only_direction:
                    label += ' - ' + direction
                ax.plot(xs, ys, label=label)

    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.xlim(right=max_x_value)
                
    ax.grid()
    ax.set(ylabel=ylabel,
           xlabel='Experiment Duration [s]')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    save_plt(key, name=name)
    for ape in additional_plot_exports:
        rprint('Additional export as {}'.format(ape))
        savefig('figures/{}_{}.{}'.format(name, key, ape), format=ape)
    plt.show()


# In[ ]:


def plot_loop(name, content, mapping, tp_data, key='max_mbit',
              additional_plot_exports=None, only_direction=None, only_core_id=None,
              loop_log_scale=False, convert_to_cycles=False, **kwargs):
    if not additional_plot_exports:
        additional_plot_exports = []
        
    fig, ax = plt.subplots(figsize=(9,6))
    ax.set_prop_cycle(tumcolor_cycler)
    
    axis_label = None
    xss = {}
    yss = {}
    loss = {}
    mapped = {}
    
    # gather data based on mapping
    for exp, run, type, label in content:
        axis_label = list(type.keys())[0]
        if label not in xss:
            xss[label] = []
            yss[label] = {}
            loss[label] = {}
        xss[label].append(list(type.values())[0])
        try:
            mapped[label] = tp_data[mapping[exp][run]]
        except KeyError as exce:
            continue
        else:
            mapped[label] = mapped[label]['tp']
            for cid, data in mapped[label].items():
                for direction, data in data.items():
                    y = data[key]
                    full = '{}-{}-{}'.format(label, cid, direction)
                    try:
                        yss[label][full].append(y)
                    except KeyError:
                        yss[label][full] = [y]
                    try:
                        lo = data['loss']
                        try:
                            loss[label][full].append(lo)
                        except KeyError:
                            loss[label][full] = [lo]
                    except KeyError:
                        pass
        
    x_data = []
    y_data = []
    loss_data = []
    labels = []
    max_x_value = 0
    min_x_value = 1000000
    for exp, data in sorted(mapped.items()):
        for cid, data in sorted(data.items()):
            if only_core_id and not cid == only_core_id:
                continue
            for direction, data in sorted(data.items()):
                if only_direction and not direction == only_direction:
                    continue
                full = '{}-{}-{}'.format(exp, cid, direction)
                label = exp
                if not only_core_id:
                    label += '-' + str(cid)
                if not only_direction:
                    label += '-' + direction
                ys = yss[exp][full]
                xs = xss[exp]
                lo = loss[exp][full]
                zipped = list(zip(xs, ys, lo))
                zipped.sort(key=lambda tup: tup[0])
                xs, ys, lo = zip(*zipped)
                x_data.append(xs)
                y_data.append(ys)
                loss_data.append(lo)
                labels.append(label)
                
                max_x_value = max(max_x_value, max(xs))
                min_x_value = min(max_x_value, min(xs))
                
                ax.plot(xs, ys, marker='x', label=latex_save(label))
    
    # process x axis label
    axis_label = ' '.join([part.capitalize() for part in axis_label.split('_')])
    
    # axis limits
    plt.ylim(bottom=0)
    plt.xlim(left=min_x_value)
    plt.xlim(right=max_x_value)
    if loop_log_scale:
        plt.xscale('log')
        axis_label += ' [log]'
                
    ax.grid()
    ax.set(ylabel=METRIC_TO_LABEL[key],
           xlabel=axis_label)
    
    if kwargs.get('model_parameters'):
        modelp = kwargs.get('model_parameters')
        # add model now that we made all configs in case of abort
        error_plot = []
        derivatives = []
        maximas = []
        for i in range(0, modelp.get('num_graphs', 1)):
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
                                    maximum=14.88, graph=i, end=end, maxima=maxima)
        
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    save_plt('loop_{}'.format(key), name=name)
    for ape in additional_plot_exports:
        rprint('Additional export as {}'.format(ape))
        savefig('figures/{}_loop_{}.{}'.format(name, key, ape), format=ape)
    plt.show()
    
    if kwargs.get('model_parameters', False):
        add_error_plot(plt, error_plot, name, key=key, log_scale=loop_log_scale, axis_label=axis_label,
                       additional_plot_exports=additional_plot_exports)
        
        add_derivative_plot(plt, derivatives, name, key=key, log_scale=loop_log_scale, axis_label=axis_label,
                            additional_plot_exports=additional_plot_exports)
    
    # also model the cpu cycles per packet
    if not convert_to_cycles:
        return
        
    fig, ax = plt.subplots(figsize=(9,6))
    ax.set_prop_cycle(tumcolor_cycler)
    
    cycle_xs = []
    cycle_ys = []
    max_x_value = 0
    min_x_value = 1000000
    for xs, ys, loss, label in zip(x_data, y_data, loss_data, labels):
        cores = int(label.split('cpu_cores-')[-1])
        current_freq = None
        if convert_to_cycles == 'from_loop':
            current_freq = list(xs)
        else:
            current_freq = float(convert_to_cycles)
        ys = convert_to_cpu_cycles(ys, current_freq, loss,
                                   loss_level=0.1, filter_surrounding=4, cores=cores)
                
        # remove -1s
        reduced = [(x, y) for x, y in zip(xs, ys) if y > -1]
        if not reduced:
            continue
        xs, ys = list(zip(*reduced))
        
        cycle_xs.append(xs)
        cycle_ys.append(ys)
        
        max_x_value = max(max_x_value, max(xs))
        min_x_value = min(min_x_value, min(xs))
                
        ax.plot(xs, ys, marker='x', label=latex_save(label))
    
    plt.ylim(bottom=0)
    plt.xlim(left=min_x_value)
    plt.xlim(right=max_x_value)
    if loop_log_scale:
        plt.xscale('log')
    
    ax.grid()
    ax.set(ylabel="Cycles per Packet [-]",
           xlabel=axis_label)
        
    if kwargs.get('model_parameters'):
        modelp = kwargs.get('model_parameters')
        # add model now that we made all configs in case of abort
        error_plot = []
        derivatives = []
        for i in range(0, modelp.get('num_graphs', 1)):
            if len(cycle_ys) <= i:
                continue
            if not len(cycle_xs) >= i + 1:
                break
            end = 0
            if modelp.get('end'):
                num = len(cycle_xs[i])
                end = num - modelp.get('end')
        
            # calculate derivative
            derivative = calculate_2nd_derivative(cycle_xs[i], cycle_ys[i], log=loop_log_scale)
            derivative.append(maxima)
            derivatives.append((cycle_xs[i], derivative, i))
            maxima = get_k_maxima(derivative[2], k=modelp.get('maxima_k', 10),
                                  j=modelp.get('maxima_j', 0), max_index=end)
            maximas[i] += maxima
            maxima = sorted(list(set(maximas[i])))
            
            # do not apply model start and end again this time
            error_plot += add_model(cycle_xs, cycle_ys, ax, modelp,
                                    maximum=2* max(cycle_ys[i]), graph=i, end=end, maxima=maxima)
        
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    save_plt('loop_cycles_{}'.format(key), name=name)
    for ape in additional_plot_exports:
        rprint('Additional export as {}'.format(ape))
        savefig('figures/{}_loop_cycles_{}.{}'.format(name, key, ape), format=ape)
    plt.show()
    
    if kwargs.get('model_parameters'):
        add_error_plot(plt, error_plot, name, name_extra='cycles', key=key, log_scale=loop_log_scale, axis_label=axis_label,
                       additional_plot_exports=additional_plot_exports)
        
        add_derivative_plot(plt, derivatives, name, name_extra='cycles', key=key,
                            log_scale=loop_log_scale, axis_label=axis_label,
                            additional_plot_exports=additional_plot_exports)


# In[ ]:


def _plot_individual_runs(tp_data, name, ape, only_direction, only_core_id):
    print('---------- regular plots of individual measurements -----------------------')
    print('this will include every single measurement run and might take long to generate the graph')
    print('set default_plots=False if it takes too long')
    # different plot types for moongen throughput data
    plot_rate(tp_data, name, only_direction=only_direction, only_core_id=only_core_id, key='mbit', ylabel='Throughput [Mbit/s]', additional_plot_exports=ape)
    plot_rate(tp_data, name, only_direction=only_direction, only_core_id=only_core_id, key='mbitcrc', ylabel='Throughput (with Framing) [Mbit/s]', additional_plot_exports=ape)
    plot_rate(tp_data, name, only_direction=only_direction, only_core_id=only_core_id, key='mpps', ylabel='Packet Rate [Mpps]', additional_plot_exports=ape)
    
def _plot_progression(tp_data, name, progression_x_label, metrics, ape, only_direction):
    print('------------ progression plots ----------------')
    for metric in metrics:
        plot_progression(tp_data, name, key=metric, ylabel=METRIC_TO_LABEL[metric], xlabel=progression_x_label,
                         additional_plot_exports=ape, only_direction=only_direction)


# In[ ]:


def plot(paths, **kwargs):
    
    # extract throughput data
    tp_data = extract_tp_data(paths, **kwargs)
    
    if not tp_data:
        rprint('No throughput data found', file=sys.stderr)
        return
    
    if kwargs.get('default_plots'):
        _plot_individual_runs(tp_data, **kwargs)
    
    if not kwargs.get('metrics'):
        print('you need to define the metrics of interest (METRIC_TO_LABEL.keys())')
        return
        
    if kwargs.get('loop_file'):
        _plot_loop(paths, tp_data, plot_loop, **kwargs)


# In[ ]:


METRIC_TO_LABEL = {
    'max_mbit'   : 'Throughput [Mbit/s]',
    'max_mbitcrc': 'Throughput (with Framing) [Mbit/s]',
    'max_mpps'   : 'Packet Rate [Mpps]',
    'avg_mbit'   : 'Average Throughput [Mbit/s]',
    'avg_mbitcrc': 'Average Throughput (with Framing) [Mbit/s]',
    'avg_mpps'   : 'Average Packet Rate [Mpps]',
    'min_mbit'   : 'Minimum Throughput [Mbit/s]',
    'min_mbitcrc': 'Minimum Throughput (with Framing) [Mbit/s]',
    'min_mpps'   : 'Minimum Packet Rate [Mpps]',
    'avg_mg_mbit'   : 'Average Throughput [Mbit/s]',
    'avg_mg_mbitcrc': 'Average Throughput (with Framing) [Mbit/s]',
    'avg_mg_mpps'   : 'Average Packet Rate [Mpps]',
}


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
    configuration = {'convert_to_cycles': 2.0,
 'dut': 'cesis',
 'experiment_name': 'number_entries',
 'latency_rates': [0.1, 0.5, 0.7],
 'loadgen': 'nida',
 'log_scale': 'True',
 'loop_plot_per': ['packet_size'],
 'loop_x_axis': 'table_entries',
 'model_end': 0,
 'model_parts': [3],
 'model_start': 0,
 'only_core_id': 1,
 'perf_stat_events': [['r08d1', 'L1_cache_misses'],
                      ['r10d1', 'L2_cache_misses'],
                      ['r20d1', 'L3_cache_misses']],
 'repetitions': 3,
 'result_dir_file': '/home/scholzd/component_benchmarking/experiments/p4_tapas/mat/table_entries/result_directory.txt',
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

RESULTS=os.path.join(result_dir, configuration['loadgen'])
THROUGHPUT_FILENAME = 'throughput-max-tx_run*.csv'
METRICS = ['max_mpps']
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

plot(
    [
        ('', configuration['experiment_name']),
    ],
    basepath=RESULTS,
    name=configuration['target'],
    throughput_file=THROUGHPUT_FILENAME,
    throughput_location='split',
    throughput_strip=1,
    
    metrics=METRICS,
    default_plots=False,
    only_direction='rx',
    only_core_id=configuration['only_core_id'],
    repeat=configuration['repetitions'],
    
    convert_to_cycles=configuration['convert_to_cycles'],
    
    loop_file=LOOP_FILENAME,
    loop_x_axis=configuration['loop_x_axis'],
    loop_plot_per=configuration['loop_plot_per'],
    loop_log_scale=configuration['log_scale'],
    
    model_parameters=model_parameters,
)


# In[ ]:




