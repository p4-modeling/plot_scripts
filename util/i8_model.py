#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import warnings
import functools
import string
import scipy.optimize
from scipy.optimize import OptimizeWarning
from scipy.stats import wasserstein_distance
warnings.simplefilter("error", OptimizeWarning)

import numpy as np
np.seterr(all='ignore')

from tqdm.auto import tqdm
from multiprocessing import Pool

import import_ipynb
try:
    try:
        from i8_tikzplotlib import latex_save, save_plt
        from util.tumcolor import tumcolor_cycler
    except ModuleNotFoundError:
        from util.i8_tikzplotlib import latex_save, save_plt
        from util.tumcolor import tumcolor_cycler
except NameError:
    # ModuleNotFoundError was added with python3.6, prior versions throw NameError here
    from util.i8_tikzplotlib import latex_save, save_plt
    from util.tumcolor import tumcolor_cycler

def convert_to_cpu_cycles(packet_rate_vector, cpu_freq, loss, cores=1, loss_level=0.01, filter_surrounding=4):
    # packet_rate_vector is in Mpps (1000 * 1000)
    # cpu_freq is in Gbit/s (1000 * 1000 * 1000)
    # as a result only multiply with 1000
    if not isinstance(cpu_freq, list):
        cpu_freq = [cpu_freq] * len(packet_rate_vector)
    cycles = [freq * 1000 * cores for freq in cpu_freq]
    
    # we can only convert this if the RX rate is below 14.88 (or in other words: loss > loss_level)
    cycles_per_packet = [cycl / val if (not val == 0 and lo > loss_level) else -1
                         for val, lo, cycl in zip(packet_rate_vector, loss, cycles)]
    
    # apply filter: from -n to + n we require at least n + 1 non -1 items
    num = len(cycles_per_packet)
    vals = []
    for idx, val in enumerate(cycles_per_packet):
        if val == -1:
            vals.append(-1)
            continue
        left = max(0, idx - filter_surrounding)
        right = min(num, idx + filter_surrounding + 1)
        vals_sel = cycles_per_packet[left:right]
        
        available = len(vals_sel)
        should_be = 2 * filter_surrounding + 1
        
        found = len([d for d in vals_sel if d > -1])
        if found >= filter_surrounding + 1 or (available < should_be and found >= filter_surrounding - 1):
            vals.append(val)
        else:
            vals.append(-1)
            
    return vals

def add_error_plot(plt, error_plot, name, name_extra='', key='', log_scale=False, axis_label='',
                   additional_plot_exports=None):
    if not error_plot:
        print('No error plot data')
        return
    fig, ax = plt.subplots(figsize=(9,6))
    ax.set_prop_cycle(tumcolor_cycler)
    
    min_x_value = 10000000
    max_x_value = 0
    for (args, kwargs) in error_plot:
        if not args[0]:
            print('No x error plot data')
            continue
            
        min_x_value = min(min_x_value, min(args[0]))
        max_x_value = max(max_x_value, max(args[0]))
        
        ax.plot(*args, **kwargs)
    
    plt.ylim(bottom=0)
    plt.xlim(left=min_x_value)
    plt.xlim(right=max_x_value)
    if log_scale:
        plt.xscale('log')
    
    ax.grid()
    ax.set(ylabel="Relative Error [%]",
           xlabel=axis_label)
        
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    save_plt('loop_{}error_{}'.format(name_extra + '_' if name_extra else '', key), name=name)
    for ape in additional_plot_exports:
        rprint('Additional export as {}'.format(ape))
        savefig('figures/{}_loop_error_{}.{}'.format(name, key, ape), format=ape)
    plt.show()
    

def add_model(x_data, y_data, ax, model_parameters, maximum=14.88, end=0, graph=0, maxima=False):
    start = model_parameters.get('start', 0)
    only_best_x = model_parameters.get('only_best_x', 1)
    model_parts = model_parameters.get('parts', [])
    model_bruteforce = model_parameters.get('bruteforce', [])
    gamma = model_parameters.get('gamma', 1e-05)
    comp_method = model_parameters.get('comp_method', 'sMAPE')
    epsilon = model_parameters.get('epsilon', 0.005)
    epsilon_rel = model_parameters.get('epsilon_rel', 0.05)
    
    if end == 0:
        end = len(x_data[graph])
    if maxima and [m for m in maxima if m >= end]:
        print('too large maximum contained, cutting: {}'.format(maxima))
        maxima = [m for m in maxima if m < end]
        print(maxima)
    x_use = x_data[graph][start:end]
    y_use = y_data[graph][start:end]
    
    max_processes = max(1, os.cpu_count() - 2)
    
    marker = 'D'
    
    error_plot = []
    
    if not model_parts:
        model_parts = [1]
        
    print('\n*** Creating models with {} parts'.format(', '.join([str(m) for m in model_parts])))
    
    if 1 in model_parts:
        # try with just one function
        print('\n*** Creating {} model(s) for graph {} ***'.format(only_best_x, graph))
        fittings = create_model(x_use, y_use, maximum=maximum, gamma=gamma, comp_method=comp_method,
                                epsilon=epsilon, epsilon_rel=epsilon_rel)
        num_models = len(fittings)
        for i, factory in enumerate(sorted(fittings)):
            if i >= only_best_x:
                continue
            fname = factory.get_name()
            errors = factory.get_errors()
            error = factory.get_error()
            params = factory.get_params()
            func = factory.get_function()
            print(factory.print_full())
            label = latex_save('{} {}'.format(graph, factory))
            
            kwargs = {'marker': marker, 'label': label}
            ax.plot(x_use, func(x_use), **kwargs)
            error_plot.append(([x_use, errors], kwargs))
        
    # try with n splits
    # uses maxima if not enabled
    for n in range(2, 6 + 1):
        if not n in model_parts:
            continue
            
        print('\n*** Preparing {} split model next ***'.format(n))
        if not maxima:
            break
        function_map = [
            None,
            None,
            calculate_2_split,
            calculate_3_split,
            calculate_4_split,
            calculate_5_split,
            calculate_6_split]
        split_map = [
            False,
            False,
            2 in model_bruteforce,
            3 in model_bruteforce,
            4 in model_bruteforce,
            5 in model_bruteforce,
            6 in model_bruteforce,
        ]
        
        best_fittings = BestNStore(only_best_x)
        smallest_error = 1
        fitting = []
        _x_use = [None] * 5
        idx = []
        
        if not split_map[n]:
            # use maxima
            if len(maxima) < n - 1:
                print("Not enough maxima for {} split".format(n))
                continue
            runs = maxima
            data = [[_idx, end, x_use, y_use, maximum, maxima, gamma, comp_method, epsilon, epsilon_rel, only_best_x] for _idx in maxima]
        else:
            # use brute force
            runs = range(0, end - 1)
            data = [[_idx, end, x_use, y_use, maximum, False, gamma, comp_method, epsilon, epsilon_rel, only_best_x] for _idx in runs]
        result = None
        print('\n*** Creating {}-split model for graph {} ({})***'.format(
            n, graph, '{} maxima'.format(len(maxima)) if not split_map[n] else 'brute force'))
        
        # parallelize
        results = []
        with Pool(max_processes) as p:
            iterable = p.imap(function_map[n], data)
            for _ in tqdm(runs):
                result = next(iterable)
                results.append(result)
        # newline after tqdm in CLI
        print()
            
        # gather results in one store
        for fitstore in results:
            best_fittings.merge(fitstore)
        
        if not best_fittings.empty():
            for cfitting in best_fittings.get():
                # now plot with best split
                print('{}'.format(cfitting.print_full()))
                x_vals = []
                y_vals = []
                for i, (domain, fittinglist) in enumerate(cfitting.get_partial_fittings()):
                    # in case we perform debugging we might get more than one best fitting
                    # plot all of the fittings, but only use best fitting for calculations/error
                    if not isinstance(fittinglist, list):
                        fittinglist = [fittinglist]
                    for k, fitting in enumerate(fittinglist):
                        func = fitting.get_function()
                        if k == 0:
                            num = i + 1
                            x_vals += domain
                            y_vals += fitting.get_errors()
                        ax.plot(domain, func(domain), marker=marker,
                                label=latex_save('{} ${}_{}$-split {}'.format(graph, n, num, fitting)))
                error_plot.append(([x_vals, y_vals],
                                   {'marker': marker, 
                                    'label': latex_save('{} {}-split (d={:.2f}%)'.format(
                                        graph, n, cfitting.get_error() * 100))}))
        else:
            print('Unable to generate split {} model'.format(n))
    return error_plot


def relative_error(ys, ys_app):
    errors = []
    absolutes = []
    for y, a in zip(ys, ys_app):
        absolute = abs(y - a)
        absolutes.append(absolute)
        if y == 0:
            relative = 1
        else:
            relative = absolute / abs(y)
        errors.append(relative)
        #print('{} {} {} {}'.format(y, a, absolute, relative))
    error = sum(errors) / len(errors)
    #print(error)
    return error, [err * 100 for err in errors]

def relative_error_sym(ys, ys_app):
    errors = []
    absolutes = []
    bottoms = []
    for y, a in zip(ys, ys_app):
        absolute = abs(y - a)
        absolutes.append(absolute)
        bottom = abs(y) + abs(a)
        bottoms.append(bottom)
        if y == 0:
            relative = 1
        else:
            relative = absolute / abs(y)
        errors.append(relative)
    
    bottomsum = sum(bottoms)
    if bottomsum == 0:
        bottomsum = -1
    error = sum(absolutes) / bottomsum
    return error, [err * 100 for err in errors]

@functools.total_ordering
class Fitting:
    # uses MAPE and AIC-based function ranks
    # alternative metrics can be used, e.g. EMD
    # write as: plug in different metrics
    def __init__(self, name, func, formatted, x, y, psi, params,
                 gamma=1e-05, comp_method='sMAPE', epsilon=0.005, epsilon_rel=0.05, maximum=14.88):
        
        # previously static values
        self.gamma= gamma
        self.comp_method = comp_method
        self.epsilon = epsilon
        self.epsilon_rel = epsilon_rel
        
        self.name = name
        self.params = self.process_params(params)
        self.func = func
        self.formatted = formatted
        self.maximum = maximum
        self.psi = psi
        self.error = None
        self.errors = None
        self.x = x
        self.y = y
        self.fitted = None
        self.fitted_unmod = None
        
        # if fitting is incomplete, this will throw an exception
        self.get_fitted()
        
    def process_params(self, params):
        ret = []
        for param in params:
            if abs(param) < self.gamma:
                if param > 0:
                    param = self.gamma
                else:
                    param = -self.gamma
            ret.append(param)
        return ret
        
    def get_function(self):
        def comp_func(x):
            #print(" - ".join([str(param) for param in self.params]))
            return np.maximum(0, np.minimum(self.maximum, self.func(x, *self.params)))
        return np.vectorize(comp_func)
    
    def get_function_unmod(self):
        # any errors should be calculated with the unlimited fitting
        def comp_func(x):
            #print(" - ".join([str(param) for param in self.params]))
            return self.func(x, *self.params)
        return np.vectorize(comp_func)
    
    def get_fitted(self):
        if self.fitted is None:
            self.fitted = self.get_function()(self.x)
        return self.fitted
    
    def get_fitted_unmod(self):
        if self.fitted_unmod is None:
            self.fitted_unmod = self.get_function_unmod()(self.x)
        return self.fitted_unmod
    
    def get_name(self):
        return self.name
    
    def get_formatted(self):
        return self.formatted
    
    def get_params(self):
        return self.params
    
    def get_rank(self):
        return self.psi
    
    def get_emd(self):
        # for distributions
        if not self.errors:
            # calculate MAPE
            error = wasserstein_distance(self.y, self.get_fitted_unmod())
            self.error = error
            self.errors = []
        return self.error
    
    def get_mape(self):
        if not self.errors:
            # calculate MAPE
            error, errors = relative_error(self.y, self.get_fitted_unmod())
            self.error = error
            self.errors = errors
        return self.error
    
    def get_smape(self):
        if not self.errors:
            # calculate MAPE
            error, errors = relative_error_sym(self.y, self.get_fitted_unmod())
            self.error = error
            self.errors = errors
        return self.error
    
    def get_error(self):
        if self.comp_method == 'MAPE':
            return self.get_mape()
        elif self.comp_method == 'sMAPE':
            return self.get_smape()
        elif self.comp_method == 'EMD':
            return self.get_emd()
        else:
            raise RuntimeError('unknown error compare method: ' + str(self.comp_method))
        
    def get_errors(self):
        if not self.errors:
            self.get_error()
        return self.errors
    
    def get_margin(self):
        return self.get_error() * self.epsilon_rel
    
    def _compare(self):
        return self.get_error()
    
    def __lt__(self, other):
        margin = max(min(self.get_margin(), other.get_margin()), self.epsilon)
        
        if abs(self._compare() - other._compare()) < margin:
            # within range, use rank
            if self.get_rank() == other.get_rank():
                # same rank, use error
                return self._compare() < other._compare()
            # use rank
            return self.get_rank() < other.get_rank()
        # otherwise just ose error
        return self._compare() < other._compare()
    
    def __str__(self, full=False):
        #sformat = "{} {} with $\eta={:.3f}%$, $\psi={}, emd={:.3f}$"
        sformat = "{} {} $\eta={:.3f}%$ ({}, {:.3f}, {}, {}), $\psi={}$"
        if full:
            #sformat.replace(":.3f", "")
            sformat += "\n      ("
            sformat += ", ".join(["{}={:.5f}".format(string.ascii_lowercase[i], param)
                                  for i, param in enumerate(self.get_params())])
            sformat += ")"
        return sformat.format(self.get_name(),
                              self.get_formatted(),
                              self.get_error() * 100,
                              self.comp_method,
                              self.get_margin(),
                              self.epsilon,
                              self.epsilon_rel,
                              self.get_rank(),
                              self.get_emd())
    
    def print_full(self):
        return self.__str__(full=True)
    
def constant(x, a):
    return a
    
def inverse_constant(x, a, b):
    return a / constant(x, b)

def linear(x, a, b):
    return a * x + b

def inverse_linear(x, a, b, c):
    return a / linear(x, b, c)

def inverse_linear_plus(x, a, b, c, d):
    return a / linear(x, b, c) + d

def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c

def inverse_quadratic(x, a, b, c, d):
    return a / quadratic(x, b, c, d)

def cubic(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

def inverse_cubic(x, a, b, c, d, e):
    return a / cubic(x, b, c, d, e)

def poly_4(x, a, b, c, d, e):
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e

def inverse_poly_4(x, a, b, c, d, e, f):
    return a / poly_4(x, b, c, d, e, f)

def poly_5(x, a, b, c, d, e, f):
    return a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + e * x + f

def inverse_poly_5(x, a, b, c, d, e, f, g):
    return a / poly_5(x, b, c, d, e, f, g)

def exponential(x, a, b, c):
    return a * (b ** x) + c

def inverse_exponential(x, a, b, c, d):
    return a / exponential(x, b, c, d)

def exponential_full(x, a, b, c, d, e):
    return a * (b ** (c * x + d)) + e

def inverse_exponential_full(x, a, b, c, d, e, f):
    return a / exponential_full(x, b, c, d, e, f)

def exponential_np(x, a, b, c):
    return a * np.exp(b * x) + c

def inverse_exponential_np(x, a, b, c, d):
    return a / exponential_np(x, b, c, d)

def exponential_np_full(x, a, b, c, d):
    return a * np.exp(b * x + c) + d

def inverse_exponential_np_full(x, a, b, c, d, e):
    return a / exponential_np_full(x, b, c, d, e)

def special_inverse_exponential_np(x, a, b, c, d, e):
    return a * (b / (c + np.exp(x + d))) + e

def exponential_neg_np(x, a, b, c):
    return a * np.exp(-b * x) + c

def inverse_exponential_neg_np(x, a, b, c, d):
    return a / exponential_neg_np(x, b, c, d)

def log_np(x, a, b, c):
    return a * np.log(b*x) + c

def inverse_log_np(x, a, b, c, d):
    return a / log_np(x, b, c, d)

def log_np_full(x, a, b, c, d):
    return a * np.log(b*x + c) + d

def inverse_log_np_full(x, a, b, c, d, e):
    return a / log_np_full(x, b, c, d, e)

def special_inverse_times_log_np(x, a, b, c):
    return a / (x * np.log(x + b) + c)

def logten_np(x, a, b, c):
    return a * np.log10(b*x) + c

def inverse_logten_np(x, a, b, c, d):
    return a / logten_np(x, b, c, d)

def logtwo_np(x, a, b, c):
    return a * np.log2(b*x) + c

def inverse_logtwo_np(x, a, b, c, d):
    return a / logtwo_np(x, b, c, d)

def gaussian(x, a, b, c):
    return a * np.exp(-((x - b) ** 2) / 2 * (c ** 2))
    
def create_model(x, y, maximum=14.88, verbose=False,
                 gamma=1e-05, comp_method='sMAPE', epsilon=0.005, epsilon_rel=0.05):
    
    functions = {
        #'exp': (exponential, 3, "$a * (b^x) + c$"),
        #'inv_exp': (inverse_exponential, 4, "$a / (b * (c^x) + d)$"),
        'exp_full': (exponential_full, 5, "$a * (b^(cx + d)) + e$"),
        'inv_exp_full': (inverse_exponential_full, 6, "$a / (b * (c^(dx + e)) + f)$"),
        #'exp_np': (exponential_np, 10, "$a * (e^{bx}) + c$"),
        #'inv_exp_np': (inverse_exponential_np, 10.1, "$a / (b * (e^{cx}) + d)$"),
        #'exp_np_full': (exponential_np_full, 11, "$a * (e^{bx + c}) + d$"),
        #'inv_exp_np_full': (inverse_exponential_np_full, 11.1, "$a / (b * (e^{cx + d}) + e)$"),
        #'spe_inv_exp_np': (special_inverse_exponential_np, 10.2, "$a * (b / (c + e^{x + d})) + e$"),
        #'exp_neg_np': (exponential_neg_np, 10, "$a * (e^{-bx}) + c$"),
        #'inv_exp_neg_np': (inverse_exponential_neg_np, 10.1, "$a / (b * (e^{-cx}) + d)$"),
        
        'log_np': (log_np, 3, "$a * log(bx) + c$"),
        'inv_log_np': (inverse_log_np, 4, "$a / (b * log(cx) + d)$"),
        'log_np_full': (log_np_full, 4, "$a * log(bx + c) + d$"),
        'inv_log_np_full': (inverse_log_np_full, 5, "$a / (b * log(cx + d) + e)$"),
        #'spe_inv_times_log_np': (special_inverse_times_log_np, 10.2, "$a / (x * log(x + b) + c)$"),
        'logten_np': (logten_np, 3, "$a * log10(bx) + c$"),
        'inv_logten_np': (inverse_logten_np, 4, "$a / (b * log10(cx) + d)$"),
        'logtwo_np': (logtwo_np, 3, "$a * log2(bx) + c$"),
        'inv_logtwo_np': (inverse_logtwo_np, 4, "$a / (b * log2(cx) + d)$"),
        
        'gaussian': (gaussian, 3, "$a * e^(-(x - b)^2 / 2 * c^2)$"),
        
        'poly_5': (poly_5, 6, "$ax^5 + bx^4 + cx^3 + dx^2 + ex + f$"),
        'inv_poly_5': (inverse_poly_5, 7, "$a / (bx^5 + cx^4 + dx^3 + ex^2 + fx + g)$"),
        'poly_4': (poly_4, 5, "$ax^4 + bx^3 + cx^2 + dx + e$"),
        'inv_poly_4': (inverse_poly_4, 6, "$a / (bx^4 + cx^3 + dx^2 + ex + f)$"),
        'cubic': (cubic, 4, "$ax^3 + bx^2 + cx + d$"),
        'inv_cubic': (inverse_cubic, 5, "$a / (bx^3 + cx^2 + dx + e)$"),
        'quad': (quadratic, 3, "$ax^2 +bx + c$"),
        'inv_quad': (inverse_quadratic, 4, "$a/(bx^2 + cx + d)$"),
        'lin': (linear, 2, "$ax + b$"),
        'inv_lin': (inverse_linear, 3, "$(a / (bx + c))$"),
        #'inv_lin_plus': (inverse_linear_plus, 1.2, "$a / (bx + c) + d"),
        'const': (constant, 1, "$a$"),
        'inv_const': (inverse_constant, 2, "$a/b$"),
    }
    
    fittings = []
    
    for name, (func, rank, formatted) in functions.items():
        if verbose:
            print('Trying ' + name)
        params = []
        pcov = []
        try:
            params, pcov = scipy.optimize.curve_fit(func, x, y)
        except RuntimeError:
            if verbose:
                print('Unable to find fitting for {}'.format(name))
            continue
        except RuntimeWarning:
            if verbose:
                print('RuntimeWarning for {}'.format(name))
            continue
        except TypeError as exce:
            if verbose:
                print('Unable to find fitting for {}: {}'.format(name, exce))
            continue
        except ValueError as exce:
            if verbose:
                print(exce)
            continue
        except scipy.optimize.OptimizeWarning as exce:
            if verbose:
                print('OptimizeWarning for {}: {}'.format(name, exce))
            # this is ok, might be salvagable
            if len(y) == 1:
                params = [y[0]]
        
        #print(["%.5f"%x for x in params])
        #perr = np.sqrt(np.diag(pcov))
        #print(["%.5f"%x for x in perr])
        #if any([True for val in perr if np.isinf(val)]):
            #pass
            #print('infinite error, skipping')
            #continue
        
        try:
            fitting = Fitting(name, func, formatted, x, y, rank, params, maximum=maximum,
                              gamma=gamma, comp_method=comp_method, epsilon=epsilon, epsilon_rel=epsilon_rel)
        except TypeError:
            if verbose:
                print("missing params, skip")
            continue
        fittings.append(fitting)
    return fittings

@functools.total_ordering
class CombinedFitting:
    
    def __init__(self, degree, split_points, domains=[], partial_fittings=[], comp_method='sMAPE', epsilon=0.005, epsilon_rel=0.05):
        # number of fittings
        self.degree = degree
        # the splitting points/function domains
        self.split_points = split_points
        self.domains = []
        # the calculated best fittings for each domain
        self.p_fittings = []
        
        self.error = -1
        self.rank = -1
        
        # previous static
        self.comp_method = comp_method
        self.epsilon = epsilon
        self.epsilon_rel = epsilon_rel
        
        if not len(domains) == len(partial_fittings):
            raise RuntimeError("uneven number of domains - partial fittings")
        
        for dom, fit in zip(domains, partial_fittings):
            self.add_partial_fitting(dom, fit)
        
    def add_partial_fitting(self, domain, fitting):
        if self.is_complete():
            raise RuntimeError('trying to add more fittings than expected degree')
        # array [xmin, ..., xmax]
        if not domain:
            # dont add empty domains
            return
        self.domains.append(domain)
        # array [Fitting1, ... Fittingn]
        # choose the best one for this domain
        # use the second line for debugging what the top x individual fittings were
        try:
            self.p_fittings.append(sorted(fitting)[0])
            #self.p_fittings.append(sorted(fitting[:1]))
        except IndexError:
            # no fitting was found
            return
                      
    def get_partial_fittings(self):
        return zip(self.domains, self.p_fittings)
        
    def is_complete(self):
        return len(self.domains) == self.degree
        
    def get_error(self):
        # check fitting is complete
        if not self.is_complete():
            return -1
        
        if self.error >= 0:
            return self.error
        
        # combined error
        sum_error = 0
        sum_range = 0
        for dom, fit in zip(self.domains, self.p_fittings):
            if isinstance(fit, list):
                fit = fit[0]
            r = len(dom)
            sum_range += r
            sum_error += fit.get_error() * r
        try:
            self.error = sum_error / sum_range
        except ZeroDivisionError:
            print(self.domains)
            print(self.p_fittings)
            raise
        return self.error
    
    def get_rank(self):
        if self.rank >= 0:
            return self.rank
    
        # combined rank
        sum_rank = 0
        sum_range = 0
        for dom, fit in zip(self.domains, self.p_fittings):
            if isinstance(fit, list):
                fit = fit[0]
            r = len(dom)
            sum_range += r
            sum_rank += fit.get_rank() * r
        try:
            self.rank = sum_rank / sum_range
        except ZeroDivisionError:
            self.rank = 1
        return self.rank
    
    def get_degree(self):
        return self.degree
    
    def get_margin(self):
        return self.get_error() * self.epsilon_rel
    
    def __lt__(self, other):
        # weigh based on rank of individual fittings
        # again if error between a factor, choose simpler
        margin = max(min(self.get_margin(), other.get_margin()), self.epsilon)
        
        if abs(self.get_error() - other.get_error()) < margin:
            if self.get_rank() == other.get_rank():
                return self.get_error() < other.get_error()
            return self.get_rank() < other.get_rank()
        return self.get_error() < other.get_error()

    def __str__(self, full=False):
        sformat = '{}-way $\eta={:.3f}%$ ({}), $\psi={:.3f}$ at {}/{}'
        sparams = []
        for i, fit in enumerate(self.p_fittings):
            if isinstance(fit, list):
                fit = fit[0]
            sformat += "\n    {} {}"
            sparams.append(i)
            if full:
                sparams.append(fit.print_full())
            else:
                sparams.append(fit)
        return sformat.format(self.degree,
                              float(self.get_error()) * 100,
                              self.comp_method,
                              self.get_rank(),
                              self.split_points,
                              [d[-1]for d in self.domains][:-1],
                              *sparams)
                               
    def print_full(self):
        return self.__str__(full=True)


class BestNStore:
    def __init__(self, n):
        self.n = n
        self.stored = []
        
    def add(self, node):
        self.stored.append(node)
        self.stored.sort()
        if len(self.stored) > self.n:
            self.stored = self.stored[:-1]
            
    def merge(self, other):
        for node in other.get():
            self.add(node)
    
    def get(self):
        return self.stored
    
    def empty(self):
        return not self.stored


def calculate_2_split(params):
    idx, end, x, y, maximum, maxima, gamma, comp_method, epsilon, epsilon_rel, only_best_x = params
    n = 2
    
    # splitpoints and domains
    idx = [idx] + [None] * (n - 2)
    split = [None] * n
    split[0] = (0, idx[0] + 1)
    split[1] = (idx[0] + 1, end)
    
    best_fittings = BestNStore(only_best_x)
    cur_fitting = CombinedFitting(n, idx, comp_method=comp_method, epsilon=epsilon, epsilon_rel=epsilon_rel)
    for i in range(n):
        xs = x[split[i][0]:split[i][1]]
        ys = y[split[i][0]:split[i][1]]
        
        cur_fitting.add_partial_fitting(xs, create_model(xs, ys, maximum=maximum, verbose=False, 
                                                         gamma=gamma, comp_method=comp_method, epsilon=epsilon, epsilon_rel=epsilon_rel))
        
    if not cur_fitting.is_complete():
        return best_fittings
        
    best_fittings.add(cur_fitting)
    return best_fittings

def calculate_3_split(params):
    idx, end, x, y, maximum, maxima, gamma, comp_method, epsilon, epsilon_rel, only_best_x = params
    n = 3
    
    # splitpoints and domains
    idx = [idx] + [None] * (n - 2)
    split = [None] * n
    split[0] = (0, idx[0] + 1)
    idx2_range = range(idx[0] + 1, end - 1)
    
    best_fittings = BestNStore(only_best_x)
    
    if maxima:
        idx2_range = maxima[maxima.index(idx[0]) + 1:]
    for idx2 in idx2_range:
        idx[1] = idx2
        split[1] = (idx[0] + 1, idx[1] + 1)
        split[2] = (idx[1] + 1, end)
        
        cur_fitting = CombinedFitting(n, idx.copy(), comp_method=comp_method, epsilon=epsilon, epsilon_rel=epsilon_rel)
        for i in range(n):
            xs = x[split[i][0]:split[i][1]]
            ys = y[split[i][0]:split[i][1]]
        
            cur_fitting.add_partial_fitting(xs, create_model(xs, ys, maximum=maximum, verbose=False, 
                                                             gamma=gamma, comp_method=comp_method, epsilon=epsilon, epsilon_rel=epsilon_rel))
        
        if not cur_fitting.is_complete():
            continue
            
        best_fittings.add(cur_fitting)
    return best_fittings

def calculate_4_split(params):
    idx, end, x, y, maximum, maxima, gamma, comp_method, epsilon, epsilon_rel, only_best_x = params
    n = 4
    
    idx = [idx] + [None] * (n - 2)
    split = [None] * n
    split[0] = (0, idx[0] + 1)
    idx2_range = range(idx[0] + 1, end - 1)
    
    best_fittings = BestNStore(only_best_x)
    
    if maxima:
        idx2_range = maxima[maxima.index(idx[0]) + 1:]
    for idx2 in idx2_range:
        idx[1] = idx2
        split[1] = (idx[0] + 1, idx[1] + 1)
        idx3_range = range(idx[1] + 1, end - 1)
        if maxima:
            idx3_range = maxima[maxima.index(idx[1]) + 1:]
        for idx3 in idx3_range:
            idx[2] = idx3
            split[2] = (idx[1] + 1, idx[2] + 1)
            split[3] = (idx[2] + 1, end)
            
            cur_fitting = CombinedFitting(n, idx.copy(), comp_method=comp_method, epsilon=epsilon)
            for i in range(n):
                xs = x[split[i][0]:split[i][1]]
                ys = y[split[i][0]:split[i][1]]
            
                cur_fitting.add_partial_fitting(xs, create_model(xs, ys, maximum=maximum, verbose=False, 
                                                                 gamma=gamma, comp_method=comp_method, epsilon=epsilon, epsilon_rel=epsilon_rel))
            
            if not cur_fitting.is_complete():
                continue
            
            best_fittings.add(cur_fitting)
    return best_fittings

def calculate_5_split(params):
    idx, end, x, y, maximum, maxima, gamma, comp_method, epsilon, epsilon_rel, only_best_x = params
    n = 5
    
    idx = [idx] + [None] * (n - 2)
    split = [None] * n
    split[0] = (0, idx[0] + 1)
    idx2_range = range(idx[0] + 1, end - 1)
    
    best_fittings = BestNStore(only_best_x)
    
    if maxima:
        idx2_range = maxima[maxima.index(idx[0]) + 1:]
    for idx2 in idx2_range:
        idx[1] = idx2
        split[1] = (idx[0] + 1, idx[1] + 1)
        idx3_range = range(idx[1] + 1, end - 1)
        if maxima:
            idx3_range = maxima[maxima.index(idx[1]) + 1:]
        for idx3 in idx3_range:
            idx[2] = idx3
            split[2] = (idx[1] + 1, idx[2] + 1)
            idx4_range = range(idx[2] + 1, end - 1)
            if maxima:
                idx4_range = maxima[maxima.index(idx[2]) + 1:]
            for idx4 in idx4_range:
                idx[3] = idx4
                split[3] = (idx[2] + 1, idx[3] + 1)
                split[4] = (idx[3] + 1, end)
                
                cur_fitting = CombinedFitting(n, idx.copy(), comp_method=comp_method, epsilon=epsilon, epsilon_rel=epsilon_rel)
                for i in range(n):
                    xs = x[split[i][0]:split[i][1]]
                    ys = y[split[i][0]:split[i][1]]
                
                    cur_fitting.add_partial_fitting(xs, create_model(xs, ys, maximum=maximum, verbose=False,
                                                                     gamma=gamma, comp_method=comp_method, epsilon=epsilon, epsilon_rel=epsilon_rel))
                
                if not cur_fitting.is_complete():
                    continue
                
                best_fittings.add(cur_fitting)
    return best_fittings

def calculate_6_split(params):
    idx, end, x, y, maximum, maxima, gamma, comp_method, epsilon, epsilon_rel, only_best_x = params
    n = 6
    
    idx = [idx] + [None] * (n - 2)
    split = [None] * n
    split[0] = (0, idx[0] + 1)
    idx2_range = range(idx[0] + 1, end - 1)
    
    best_fittings = BestNStore(only_best_x)
    
    if maxima:
        idx2_range = maxima[maxima.index(idx[0]) + 1:]
    for idx2 in idx2_range:
        idx[1] = idx2
        split[1] = (idx[0] + 1, idx[1] + 1)
        idx3_range = range(idx[1] + 1, end - 1)
        if maxima:
            idx3_range = maxima[maxima.index(idx[1]) + 1:]
        for idx3 in idx3_range:
            idx[2] = idx3
            split[2] = (idx[1] + 1, idx[2] + 1)
            idx4_range = range(idx[2] + 1, end - 1)
            if maxima:
                idx4_range = maxima[maxima.index(idx[2]) + 1:]
            for idx4 in idx4_range:
                idx[3] = idx4
                split[3] = (idx[2] + 1, idx[3] + 1)
                idx5_range = range(idx[3] + 1, end - 1)
                if maxima:
                    idx5_range = maxima[maxima.index(idx[3]) + 1:]
                for idx5 in idx5_range:
                    idx[4] = idx5
                    split[4] = (idx[3] + 1, idx[4] + 1)
                    split[5] = (idx[4] + 1, end)
                    
                    cur_fitting = CombinedFitting(n, idx.copy(), comp_method=comp_method, epsilon=epsilon, epsilon_rel=epsilon_rel)
                    for i in range(n):
                        xs = x[split[i][0]:split[i][1]]
                        ys = y[split[i][0]:split[i][1]]
                    
                        cur_fitting.add_partial_fitting(xs, create_model(xs, ys, maximum=maximum, verbose=False,
                                                                         gamma=gamma, comp_method=comp_method, epsilon=epsilon, epsilon_rel=epsilon_rel))
                    
                    if not cur_fitting.is_complete():
                        continue
                    
                    best_fittings.add(cur_fitting)
    return best_fittings


# In[ ]:


def calculate_derivative(x, y, log=False, r=1):
    # calculate the local right-sided derivative using the next r values
    derivative = [0] * r
    if log:
        x = np.log10(x)
    for i, _ in enumerate(x[r:-r]):
        derivative.append(( abs(y[i + 1 + r] - y[i + 1]) / (x[i + 1 + r] - x[i + 1]) ))
    for _ in range(len(x) - len(derivative)):
        derivative.append(0)
    return derivative

def normalize(x):
    maximum = max(x)
    if not maximum:
        return x
    return [val / maximum for val in x]

def smooth(x):
    # alternatives: https://en.wikipedia.org/wiki/Kernel_smoother
    range = 1
    result = []
    for i, _ in enumerate(x):
        lower = max(0, i - range)
        upper = min(len(x), i + range)
        avg = sum(x[lower:upper]) / (upper - lower)
        med = np.median(x[lower:upper])
        #print("{} {} {}".format(lower, upper, avg))
        result.append(med)
    return result

def quantize(x):
    factor = 2
    result = [0] * factor
    for i, _ in enumerate(x[factor:-factor]):
        i = i + factor
        #pprint(x[i - factor:i + factor + 1])
        if x[i] == max(x[i - factor:i + factor + 1]):
            result.append(x[i])
        else:
            result.append(0)
    for _ in range(len(x) - len(result)):
        result.append(0)
    return result

def calculate_2nd_derivative(x, y, log):
    derivative = calculate_derivative(x, y, r=2, log=log)
    derivative2 = calculate_derivative(x, derivative, r=2, log=log)
    return [normalize(derivative), normalize(derivative2), quantize(normalize(derivative2))]

def get_k_maxima(x, k=5, j=0, max_index=0):
    '''
    return k maxima with each j surrounding values
    '''
    
    x = x.copy() # otherwise we delete actual plotted values
    k = min(k, len(x))
    ordered = sorted(x)
    _maxima = []
    maxima = []
    found = 0
    for i in range(len(x)):
        maxi = ordered[-(1 + i)]
        #if not maxi >= 0.2:
        #    break
        idx = x.index(maxi)
        if max_index and idx > max_index:
            # lies outside of restricted x range, skip
            print('skipping maximum {} at {}'.format(maxi, idx))
            continue
        else:
            found += 1
        _maxima.append(idx)
        # in case we have the same multiple times, reset this so index will find the next
        x[idx] = 0
        if found == k:
            break
    # add surrounding j
    for i in _maxima:
        for a in range(1, j + 1):
            if i - a >= 0:
                maxima.append(i - a)
            if i + a <= len(x) - 1:
                maxima.append(i + a)
            maxima.append(i)
        if j == 0:
            maxima = _maxima
    maxima = list(set(maxima))
    return sorted(maxima)

def add_derivative_plot(plt, derivatives, name, name_extra='', key='', log_scale=False, axis_label='',
                        additional_plot_exports=None):
    if not derivatives:
        print('No derivatives data')
        return
    fig, ax = plt.subplots(figsize=(9,6))
    ax.set_prop_cycle(tumcolor_cycler)
    
    min_x_value = 10000000
    max_x_value = 0
    prev = None
    for (x, deriv, idx) in derivatives:
        min_x_value = min(min_x_value, min(x))
        max_x_value = max(max_x_value, max(x))
        for i, y in enumerate(deriv):
            label = '1st derivative'
            marker = 'x'
            linewidth = 1.0
            ms = 3
            if i == 1:
                label = '2nd derivative'
            if i == 2:
                label = 'quant. norm. 2nd derivative'
                marker = 'D'
                linewidth = 0.0
                ms = 7
            if i == 3:
                label = 'chosen maxima'
                marker = 'D'
                linewidth = 0.0
                ms = 7
                #for a in y:
                #    prev[int(a)] = 0.5
                x = [a for num, a in enumerate(x) if num in y]
                y = [a for num, a in enumerate(prev) if num in y]
            
            ax.plot(x, y, marker=marker, ms=ms, label="{} {}".format(idx, label), linewidth=linewidth)
            prev = y
    
    plt.ylim(bottom=0, top=1)
    plt.xlim(left=min_x_value)
    plt.xlim(right=max_x_value)
    if log_scale:
        plt.xscale('log')
    
    ax.grid()
    ax.set(ylabel="Abs. derivative [-]",
           xlabel=axis_label)
        
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    save_plt('loop_{}derivative_{}'.format(name_extra + '_' if name_extra else '', key), name=name)
    for ape in additional_plot_exports:
        rprint('Additional export as {}'.format(ape))
        savefig('figures/{}_loop_derivative_{}.{}'.format(name, key, ape), format=ape)
    plt.show()

