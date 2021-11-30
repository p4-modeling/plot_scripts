#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import os
from glob import glob
rprint=print
from pprint import pprint as print


# In[ ]:


def read_loopfile(loopfile):
    data = None
    with open(loopfile) as infile:
        try:
            data = json.load(infile)
        except json.JSONDecodeError:
            # for old posd files, delete
            infile.seek(0)
            content = infile.read()
            content = content.replace("'", '"')
            data = json.loads(content)
    return data


# In[ ]:


def extract_loop_data(paths, loop_file='loopfile', basepath='/', **kwargs):
    data = {}
    if not isinstance(paths, list):
        paths = [paths]

    found = False
    for path in paths:
        name = None
        if not isinstance(path, tuple):
            name = path.replace('_', '-') # tex friendly path
        else:
            name = path[1]
            path = path[0]
        
        data[name] = {}
        
        extended_path = os.path.join(basepath, path)
        loopfile = os.path.join(extended_path, loop_file)
        
        loopfiles = glob(loopfile)
        if not loopfiles:
            # previously used experiment_run for loop files
            loopfile = loopfile.replace('measurement', 'experiment')
            loopfiles = glob(loopfile)
        rprint('Processing {} loopfiles '.format(len(loopfiles)) + loopfile)
        for loop in sorted(loopfiles):
            #print('Loopfile ' + loop)
            run = int(loop.split('_run')[1].split('.loop')[0])
                
            # load data
            try:
                raw_data = read_loopfile(loop)
            except FileNotFoundError as exce:
                rprint('Skipping {} - {}'.format(loop, exce), file=sys.stderr)
                continue
            data[name][run] = raw_data
            found = True
    if not found:
        print('WARNING: no loop data found')

    return data


# In[ ]:


def _plot_loop(paths, tp_data, function, name='default', loop_x_axis='', loop_plot_per='', metrics='',
               **kwargs):
    if (kwargs.get('loop_file') and not loop_x_axis)             or (loop_x_axis and not kwargs.get('loop_file')):
        raise RuntimeError('must define loop_file AND loop_x_axis if using loop variables')
    
    print('---------------- plotting using loop variables ----------------------')
    loop_data = extract_loop_data(paths, **kwargs)
    
    # group data by loop params
    groups = {}
    if not loop_plot_per:
        key = 'all_in_one'
        for exp, l_data in loop_data.items():
            for run, data in l_data.items():
                tup = (exp, run, data, exp)
                try:
                    groups[key].append(tup)
                except KeyError:
                    groups[key] = [tup]
    else:
        # first key
        key = loop_plot_per[0]
        keys = [key]
        for exp, l_data in loop_data.items():
            for run, data in l_data.items():
                value = str(data[key])
                del data[key]
                value = '{}-{}-{}'.format(exp, key, value)
                tup = (exp, run, data, value)
                try:
                    groups[value].append(tup)
                except KeyError:
                    groups[value] = [tup]
                    
        # if we have more keys sort the new subgroups
        for key in loop_plot_per[1:-1]:
            new_groups = {}
            for old_name, group in groups.items():
                for item in group:
                    test, run, rest, label = item
                    current = rest[key]
                    del rest[key]
                    new_name = '{}-{}-{}'.format(old_name, key, current)
                    if new_name not in new_groups:
                        new_groups[new_name] = []
                    new_groups[new_name].append((test, run, rest, new_name))
            groups = new_groups
        
    # in the groups concat everything that is not x axis label
    for group, content in groups.items():
        for idx, tup in enumerate(content):
            values = tup[2]
            label = tup[3]
            for key, value in values.items():
                if not key == loop_x_axis:
                    label = '{}-{}-{}'.format(label, key, value)
            new_values = {}
            new_values[loop_x_axis] = values[loop_x_axis]
            content[idx] = (tup[0], tup[1], new_values, label)
                
    # names to {exp: {number: name}}
    mapping = {}
    for key in tp_data.keys():
        exp = '/'.join(key.split('/')[:-1])
        try:
            number = int(key.split('_run')[1].split('.')[0])
        except IndexError:
            print('No run found in data')
            return
        if exp not in mapping:
            mapping[exp] = {}
        mapping[exp][number] = key        
        
    # plot groups
    for key, content in groups.items():
        plotname = key
        if name:
            plotname = '{}_{}'.format(name, key)
        for metric in metrics:
            function(plotname, content, mapping, tp_data, key=metric, **kwargs)

