# source: https://github.com/robintibor/braindecode

import mne
import scipy.signal
import numpy as np
import pandas as pd
from collections import OrderedDict, Counter
from numpy.random import RandomState

def mne_apply(func, raw, verbose='WARNING'):
    """
    Apply function to data of `mne.io.RawArray`.
    
    Parameters
    ----------
    func: function
        Should accept 2d-array (channels x time) and return modified 2d-array
    raw: `mne.io.RawArray`
    verbose: bool
        Whether to log creation of new `mne.io.RawArray`.
    Returns
    -------
    transformed_set: Copy of `raw` with data transformed by given function.
    """
    new_data = func(raw.get_data())
    return mne.io.RawArray(new_data, raw.info, verbose=verbose)
    

def bandpass_cnt(data, low_cut_hz, high_cut_hz, fs, filt_order=3, axis=0,
                 filtfilt=False):
    """
     Bandpass signal applying **causal** butterworth filter of given order.
    Parameters
    ----------
    data: 2d-array
        Time x channels
    low_cut_hz: float
    high_cut_hz: float
    fs: float
    filt_order: int
    filtfilt: bool
        Whether to use filtfilt instead of lfilter
    Returns
    -------
    bandpassed_data: 2d-array
        Data after applying bandpass filter.
    """
    if (low_cut_hz == 0 or low_cut_hz is None) and (
                    high_cut_hz == None or high_cut_hz == fs / 2.0):
        # log.info("Not doing any bandpass, since low 0 or None and "
        #         "high None or nyquist frequency")
        return data.copy()
    if low_cut_hz == 0 or low_cut_hz == None:
        #log.info("Using lowpass filter since low cut hz is 0 or None")
        return lowpass_cnt(data, high_cut_hz, fs, filt_order=filt_order, axis=axis)
    if high_cut_hz == None or high_cut_hz == (fs / 2.0):
        #log.info(
        #    "Using highpass filter since high cut hz is None or nyquist freq")
        return highpass_cnt(data, low_cut_hz, fs, filt_order=filt_order, axis=axis)

    nyq_freq = 0.5 * fs
    low = low_cut_hz / nyq_freq
    high = high_cut_hz / nyq_freq
    b, a = scipy.signal.butter(filt_order, [low, high], btype='bandpass')
    assert filter_is_stable(a), "Filter should be stable..."
    if filtfilt:
        data_bandpassed = scipy.signal.filtfilt(b, a, data, axis=axis)
    else:
        data_bandpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_bandpassed

def highpass_cnt(data, low_cut_hz, fs, filt_order=3, axis=0):
    """
     Highpass signal applying **causal** butterworth filter of given order.
    Parameters
    ----------
    data: 2d-array
        Time x channels
    low_cut_hz: float
    fs: float
    filt_order: int
    Returns
    -------
    highpassed_data: 2d-array
        Data after applying highpass filter.
    """
    if (low_cut_hz is None) or (low_cut_hz == 0):
        #log.info("Not doing any highpass, since low 0 or None")
        return data.copy()
    b, a = scipy.signal.butter(filt_order, low_cut_hz / (fs / 2.0),
                               btype='highpass')
    assert filter_is_stable(a)
    data_highpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_highpassed


def lowpass_cnt(data, high_cut_hz, fs, filt_order=3, axis=0):
    """
     Lowpass signal applying **causal** butterworth filter of given order.
    Parameters
    ----------
    data: 2d-array
        Time x channels
    high_cut_hz: float
    fs: float
    filt_order: int
    Returns
    -------
    lowpassed_data: 2d-array
        Data after applying lowpass filter.
    """
    if (high_cut_hz is None) or (high_cut_hz ==  fs / 2.0):
        #log.info(
        #   "Not doing any lowpass, since high cut hz is None or nyquist freq.")
        return data.copy()
    b, a = scipy.signal.butter(filt_order, high_cut_hz / (fs / 2.0),
                               btype='lowpass')
    assert filter_is_stable(a)
    data_lowpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_lowpassed

def filter_is_stable(a):
    """
    Check if filter coefficients of IIR filter are stable.
    
    Parameters
    ----------
    a: list or 1darray of number
        Denominator filter coefficients a.
    Returns
    -------
    is_stable: bool
        Filter is stable or not.  
    Notes
    ----
    Filter is stable if absolute value of all  roots is smaller than 1,
    see [1]_.
    
    References
    ----------
    .. [1] HYRY, "SciPy 'lfilter' returns only NaNs" StackOverflow,
       http://stackoverflow.com/a/8812737/1469195
    """
    assert a[0] == 1.0, (
        "a[0] should normally be zero, did you accidentally supply b?\n"
        "a: {:s}".format(str(a)))
    # from http://stackoverflow.com/a/8812737/1469195
    return np.all(np.abs(np.roots(a))<1)

def exponential_running_standardize(data, factor_new=0.001,
                                    init_block_size=None, eps=1e-4):
    """
    Perform exponential running standardization. 
    
    Compute the exponental running mean :math:`m_t` at time `t` as 
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.
    
    Then, compute exponential running variance :math:`v_t` at time `t` as 
    :math:`v_t=\mathrm{factornew} \cdot (m_t - x_t)^2 + (1 - \mathrm{factornew}) \cdot v_{t-1}`.
    
    Finally, standardize the data point :math:`x_t` at time `t` as:
    :math:`x'_t=(x_t - m_t) / max(\sqrt{v_t}, eps)`.
    
    
    Parameters
    ----------
    data: 2darray (time, channels)
    factor_new: float
    init_block_size: int
        Standardize data before to this index with regular standardization. 
    eps: float
        Stabilizer for division by zero variance.
    Returns
    -------
    standardized: 2darray (time, channels)
        Standardized data.
    """
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    if init_block_size is not None:
        other_axis = tuple(range(1, len(data.shape)))
        init_mean = np.mean(data[0:init_block_size], axis=other_axis,
                            keepdims=True)
        init_std = np.std(data[0:init_block_size], axis=other_axis,
                          keepdims=True)
        init_block_standardized = (data[0:init_block_size] - init_mean) / \
                                  np.maximum(eps, init_std)
        standardized[0:init_block_size] = init_block_standardized
    return standardized

def create_signal_target_from_raw_mne(
        raw, name_to_start_codes, epoch_ival_ms,
        name_to_stop_codes=None,
        prepad_trials_to_n_samples=None,
        one_hot_labels=False,
        one_label_per_trial=True):
    """
    Create SignalTarget set from given `mne.io.RawArray`.
    
    Parameters
    ----------
    raw: `mne.io.RawArray`
    name_to_start_codes: OrderedDict (str -> int or list of int)
        Ordered dictionary mapping class names to marker code or marker codes.
        y-labels will be assigned in increasing key order, i.e.
        first classname gets y-value 0, second classname y-value 1, etc.
    epoch_ival_ms: iterable of (int,int)
        Epoching interval in milliseconds. In case only `name_to_codes` given,
        represents start offset and stop offset from start markers. In case
        `name_to_stop_codes` given, represents offset from start marker
        and offset from stop marker. E.g. [500, -500] would mean 500ms
        after the start marker until 500 ms before the stop marker.
    name_to_stop_codes: dict (str -> int or list of int), optional
        Dictionary mapping class names to stop marker code or stop marker codes.
        Order does not matter, dictionary should contain each class in
        `name_to_codes` dictionary.
    prepad_trials_to_n_samples: int
        Pad trials that would be too short with the signal before it (only
        valid if name_to_stop_codes is not None).
    one_hot_labels: bool, optional
        Whether to have the labels in a one-hot format, e.g. [0,0,1] or to
        have them just as an int, e.g. 2
    one_label_per_trial: bool, optional
        Whether to have a timeseries of labels or just a single label per trial. 
    Returns
    -------
    dataset: :class:`.SignalAndTarget`
        Dataset with `X` as the trial signals and `y` as the trial labels.
    """
    data = raw.get_data()
    events = np.array([raw.info['events'][:,0],
                      raw.info['events'][:,2]]).T
    fs = raw.info['sfreq']
    return create_signal_target(
        data, events, fs, name_to_start_codes,
        epoch_ival_ms,
        name_to_stop_codes=name_to_stop_codes,
        prepad_trials_to_n_samples=prepad_trials_to_n_samples,
        one_hot_labels=one_hot_labels,
        one_label_per_trial=one_label_per_trial)

def create_signal_target(data, events, fs, name_to_start_codes, epoch_ival_ms,
                         name_to_stop_codes=None, prepad_trials_to_n_samples=None,
                         one_hot_labels=False, one_label_per_trial=True):
    """
    Create SignalTarget set given continuous data.
    
    Parameters
    ----------
    data: 2d-array of number
        The continuous recorded data. Channels x times order.
    events: 2d-array
        Dimensions: Number of events, 2. For each event, should contain sample
        index and marker code.
    fs: number
        Sampling rate.
    name_to_start_codes: OrderedDict (str -> int or list of int)
        Ordered dictionary mapping class names to marker code or marker codes.
        y-labels will be assigned in increasing key order, i.e.
        first classname gets y-value 0, second classname y-value 1, etc.
    epoch_ival_ms: iterable of (int,int)
        Epoching interval in milliseconds. In case only `name_to_codes` given,
        represents start offset and stop offset from start markers. In case
        `name_to_stop_codes` given, represents offset from start marker
        and offset from stop marker. E.g. [500, -500] would mean 500ms
        after the start marker until 500 ms before the stop marker.
    name_to_stop_codes: dict (str -> int or list of int), optional
        Dictionary mapping class names to stop marker code or stop marker codes.
        Order does not matter, dictionary should contain each class in
        `name_to_codes` dictionary.
    prepad_trials_to_n_samples: int, optional
        Pad trials that would be too short with the signal before it (only
        valid if name_to_stop_codes is not None).
    one_hot_labels: bool, optional
        Whether to have the labels in a one-hot format, e.g. [0,0,1] or to
        have them just as an int, e.g. 2
    one_label_per_trial: bool, optional
        Whether to have a timeseries of labels or just a single label per trial. 
    Returns
    -------
    dataset: :class:`.SignalAndTarget`
        Dataset with `X` as the trial signals and `y` as the trial labels.
    """
    if name_to_stop_codes is None:
        return _create_signal_target_from_start_and_ival(
            data, events, fs, name_to_start_codes, epoch_ival_ms,
            one_hot_labels=one_hot_labels,
            one_label_per_trial=one_label_per_trial)
    else:
        return _create_signal_target_from_start_and_stop(
            data, events, fs, name_to_start_codes, epoch_ival_ms,
            name_to_stop_codes, prepad_trials_to_n_samples,
            one_hot_labels=one_hot_labels,
            one_label_per_trial=one_label_per_trial)
        
def _create_signal_target_from_start_and_ival(
        data, events, fs, name_to_codes, epoch_ival_ms,
        one_hot_labels, one_label_per_trial):
    cnt_y, i_start_stops = _create_cnt_y_and_trial_bounds_from_start_and_ival(
        data.shape[1], events, fs, name_to_codes, epoch_ival_ms
    )
    signal_target = _create_signal_target_from_cnt_y_start_stops(
        data, cnt_y, i_start_stops, prepad_trials_to_n_samples=None,
        one_hot_labels=one_hot_labels,
        one_label_per_trial=one_label_per_trial)
    # make into arrray as all should have same dimensions
    signal_target.X = np.array(signal_target.X, dtype=np.float32)
    signal_target.y = np.array(signal_target.y, dtype=np.int64)
    return signal_target

def _create_cnt_y_and_trial_bounds_from_start_and_ival(
        n_samples, events, fs, name_to_start_codes, epoch_ival_ms):
    ival_in_samples = ms_to_samples(np.array(epoch_ival_ms), fs)
    start_offset = np.int32(np.round(ival_in_samples[0]))
    # we will use ceil but exclusive...
    stop_offset = np.int32(np.ceil(ival_in_samples[1]))
    mrk_code_to_name_and_y = _to_mrk_code_to_name_and_y(name_to_start_codes)
    class_to_n_trials = Counter()
    n_classes = len(name_to_start_codes)
    cnt_y = np.zeros((n_samples, n_classes), dtype=np.int64)
    i_start_stops = []
    for i_sample, mrk_code in zip(events[:, 0], events[:, 1]):
        start_sample = int(i_sample) + start_offset
        stop_sample = int(i_sample) + stop_offset
        if mrk_code in mrk_code_to_name_and_y:
            if start_sample < 0:
                #log.warning(
                #    "Ignore trial with marker code {:d}, would start at "
                #    "sample {:d}".format(mrk_code, start_sample))
                continue
            if stop_sample > n_samples:
                #log.warning("Ignore trial with marker code {:d}, would end at "
                #            "sample {:d} of {:d}".format(
                #    mrk_code, stop_sample - 1, n_samples - 1))
                continue

            name, this_y = mrk_code_to_name_and_y[mrk_code]
            i_start_stops.append((start_sample, stop_sample))
            cnt_y[start_sample:stop_sample, this_y] = 1
            class_to_n_trials[name] += 1
    #log.info("Trial per class:\n{:s}".format(str(class_to_n_trials)))
    return cnt_y, i_start_stops

def ms_to_samples(ms, fs):
    """
    Compute milliseconds to number of samples.
    
    Parameters
    ----------
    ms: number
        Milliseconds
    fs: number
        Sampling rate
    Returns
    -------
    n_samples: int
        Number of samples
    """
    return ms * fs / 1000.0

def _to_mrk_code_to_name_and_y(name_to_codes):
    # Create mapping from marker code to class name and y=classindex
    mrk_code_to_name_and_y = {}
    for i_class, class_name in enumerate(name_to_codes):
        codes = name_to_codes[class_name]
        if hasattr(codes, '__len__'):
            for code in codes:
                assert code not in mrk_code_to_name_and_y
                mrk_code_to_name_and_y[code] = (class_name, i_class)
        else:
            assert codes not in mrk_code_to_name_and_y
            mrk_code_to_name_and_y[codes] = (class_name, i_class)
    return mrk_code_to_name_and_y

def _create_signal_target_from_cnt_y_start_stops(
        data,
        cnt_y,
        i_start_stops,
        prepad_trials_to_n_samples,
        one_hot_labels,
        one_label_per_trial):
    if prepad_trials_to_n_samples is not None:
        new_i_start_stops = []
        for i_start, i_stop in i_start_stops:
            if (i_stop - i_start) > prepad_trials_to_n_samples:
                new_i_start_stops.append((i_start, i_stop))
            elif i_stop >= prepad_trials_to_n_samples:
                new_i_start_stops.append(
                    (i_stop - prepad_trials_to_n_samples, i_stop))
            else:
                #log.warning("Could not pad trial enough, therefore not "
                #            "not using trial from {:d} to {:d}".format(
                #    i_start, i_stop
                #))
                continue

    else:
        new_i_start_stops = i_start_stops

    X = []
    y = []
    for i_start, i_stop in new_i_start_stops:
        if i_start < 0:
            #log.warning("Trial start too early, therefore not "
            #            "not using trial from {:d} to {:d}".format(
            #    i_start, i_stop
            #))
            continue
        if i_stop > data.shape[1]:
            #log.warning("Trial stop too late (past {:d}), therefore not "
            #            "not using trial from {:d} to {:d}".format(
            #    data.shape[1] - 1,
            #    i_start, i_stop
            #))
            continue
        X.append(data[:, i_start:i_stop].astype(np.float32))
        y.append(cnt_y[i_start:i_stop])

    # take last label always
    if one_label_per_trial:
        new_y = []
        for this_y in y:
            # if destroying one hot later, just set most occuring class to 1
            unique_labels, counts = np.unique(
                this_y, axis=0, return_counts=True)
            if not one_hot_labels:
                meaned_y = np.mean(this_y, axis=0)
                this_new_y = np.zeros_like(meaned_y)
                this_new_y[np.argmax(meaned_y)] = 1
            else:
                # take most frequency occurring label combination
                this_new_y = unique_labels[np.argmax(counts)]

            if len(unique_labels) > 1:
                log.warning("Different labels within one trial: {:s},"
                            "setting single trial label to  {:s}".format(
                    str(unique_labels), str(this_new_y)
                ))
            new_y.append(this_new_y)
        y = new_y
    if not one_hot_labels:
        # change from one-hot-encoding to regular encoding
        # with -1 as indication none of the classes are present
        new_y = []
        for this_y in y:
            if one_label_per_trial:
                if np.sum(this_y) == 0:
                    this_new_y = -1
                else:
                    this_new_y = np.argmax(this_y)
                if np.sum(this_y) > 1:
                    log.warning(
                        "Have multiple active classes and will convert to "
                        "lowest class")
            else:
                if np.max(np.sum(this_y, axis=1)) > 1:
                    log.warning(
                        "Have multiple active classes and will convert to "
                        "lowest class")
                this_new_y = np.argmax(this_y, axis=1)
                this_new_y[np.sum(this_y, axis=1) == 0] = -1
            new_y.append(this_new_y)
        y = new_y
    if one_label_per_trial:
        y = np.array(y, dtype=np.int64)

    return SignalAndTarget(X, y)

class SignalAndTarget(object):
    """
    Simple data container class.
    Parameters
    ----------
    X: 3darray or list of 2darrays
        The input signal per trial.
    y: 1darray or list
        Labels for each trial.
    """
    def __init__(self, X, y, z=None):
        assert len(X) == len(y)
        self.X = X
        self.y = y
        if type(z) != 'NoneType':
            assert len(X) == len(y)
            self.z = z
        else:
            self.z = None
        
class CropsFromTrialsIterator(object):
    """
    Iterator sampling crops out the trials so that each sample 
    (after receptive size of the ConvNet) in each trial is predicted.
    
    Predicting the given input batches can lead to some samples
    being predicted multiple times, if the receptive field size
    (input_time_length - n_preds_per_input + 1) is not a divisor
    of the trial length.  :func:`compute_preds_per_trial_from_crops`
    can help with removing the overlapped predictions again for evaluation.
    Parameters
    ----------
    batch_size: int
    input_time_length: int
        Input time length of the ConvNet, determines size of batches in
        3rd dimension.
    n_preds_per_input: int
        Number of predictions ConvNet makes per one input. Can be computed
        by making a forward pass with the given input time length, the
        output length in 3rd dimension is n_preds_per_input.
    seed: int
        Random seed for initialization of `numpy.RandomState` random generator
        that shuffles the batches.
    
    See Also
    --------
    braindecode.experiments.monitors.compute_preds_per_trial_from_crops : Assigns predictions to trials, removes overlaps.
    """
    def __init__(self, batch_size, input_time_length, n_preds_per_input,
                 seed=(2017, 6, 28)):
        self.batch_size = batch_size
        self.input_time_length = input_time_length
        self.n_preds_per_input = n_preds_per_input
        self.seed = seed
        self.rng = RandomState(self.seed)

    def reset_rng(self):
        self.rng = RandomState(self.seed)

    def get_batches(self, dataset, shuffle):
        # start always at first predictable sample, so
        # start at end of receptive field
        n_receptive_field = self.input_time_length - self.n_preds_per_input + 1
        i_trial_starts = [n_receptive_field - 1] * len(dataset.X)
        i_trial_stops = [trial.shape[1] for trial in dataset.X]

        # Check whether input lengths ok
        input_lens = i_trial_stops
        for i_trial, input_len in enumerate(input_lens):
            assert input_len >= self.input_time_length, (
                "Input length {:d} of trial {:d} is smaller than the "
                "input time length {:d}".format(input_len, i_trial,
                                                self.input_time_length))
        start_stop_blocks_per_trial = _compute_start_stop_block_inds(
            i_trial_starts, i_trial_stops, self.input_time_length,
            self.n_preds_per_input, check_preds_smaller_trial_len=True)
        for i_trial, trial_blocks in enumerate(start_stop_blocks_per_trial):
            assert trial_blocks[0][0] == 0
            assert trial_blocks[-1][1] == i_trial_stops[i_trial]

        return self._yield_block_batches(dataset.X, dataset.y, dataset.z,
                                        start_stop_blocks_per_trial,
                                        shuffle=shuffle)

    def _yield_block_batches(self, X, y, z, start_stop_blocks_per_trial, shuffle):
        # add trial nr to start stop blocks and flatten at same time
        i_trial_start_stop_block = [(i_trial, start, stop)
                                      for i_trial, block in
                                          enumerate(start_stop_blocks_per_trial)
                                      for (start, stop) in block]
        i_trial_start_stop_block = np.array(i_trial_start_stop_block)
        if i_trial_start_stop_block.ndim == 1:
            i_trial_start_stop_block = i_trial_start_stop_block[None,:]

        blocks_per_batch = get_balanced_batches(len(i_trial_start_stop_block),
                                       batch_size=self.batch_size,
                                       rng=self.rng,
                                       shuffle=shuffle)
        for i_blocks in blocks_per_batch:
            start_stop_blocks = i_trial_start_stop_block[i_blocks]
            batch = _create_batch_from_i_trial_start_stop_blocks(
                X, y, z, start_stop_blocks, self.n_preds_per_input)
            yield batch
            
def split_into_two_sets(dataset, first_set_fraction=None, n_first_set=None):
    """
    Split set into two sets either by fraction of first set or by number
    of trials in first set.
    Parameters
    ----------
    dataset: :class:`.SignalAndTarget`
    first_set_fraction: float, optional
        Fraction of trials in first set.
    n_first_set: int, optional
        Number of trials in first set
    Returns
    -------
    first_set, second_set: :class:`.SignalAndTarget`
        The two splitted sets.
    """
    assert (first_set_fraction is None) != (n_first_set is None), (
        "Pass either first_set_fraction or n_first_set")
    if n_first_set is None:
        n_first_set = int(round(len(dataset.X) * first_set_fraction))
    assert n_first_set < len(dataset.X)
    first_set = apply_to_X_y_z(lambda a: a[:n_first_set], dataset)
    second_set = apply_to_X_y_z(lambda a: a[n_first_set:], dataset)
    return first_set, second_set

def apply_to_X_y_z(fn, *sets):
    """
    Apply a function to all `X` and `y` attributes of all given sets.
    
    Applies function to list of X arrays and to list of y arrays separately.
    
    Parameters
    ----------
    fn: function
        Function to apply
    sets: :class:`.SignalAndTarget` objects
    Returns
    -------
    result_set: :class:`.SignalAndTarget`
        Dataset with X and y as the result of the
        application of the function.
    """
    z = None
    X = fn(*[s.X for s in sets])
    y = fn(*[s.y for s in sets])
    
    if type(sets[0].z) == np.ndarray:
        z = fn(*[s.z for s in sets])
    return SignalAndTarget(X,y,z)

def _compute_start_stop_block_inds(i_trial_starts, i_trial_stops,
                                   input_time_length, n_preds_per_input,
                                   check_preds_smaller_trial_len):
    """
    Compute start stop block inds for all trials
    Parameters
    ----------
    i_trial_starts: 1darray/list of int
        Indices of first samples to predict(!).
    i_trial_stops: 1darray/list of int
        Indices one past last sample to predict.
    input_time_length: int
    n_preds_per_input: int
    check_preds_smaller_trial_len: bool
        Check whether predictions fit inside trial
    Returns
    -------
    start_stop_blocks_per_trial: list of list of (int, int)
        Per trial, a list of 2-tuples indicating start and stop index
        of the inputs needed to predict entire trial.
    """
    # create start stop indices for all batches still 2d trial -> start stop
    start_stop_blocks_per_trial = []
    for i_trial in range(len(i_trial_starts)):
        i_trial_start = i_trial_starts[i_trial]
        i_trial_stop = i_trial_stops[i_trial]
        start_stop_blocks = _get_start_stop_blocks_for_trial(
            i_trial_start, i_trial_stop, input_time_length,
            n_preds_per_input)

        if check_preds_smaller_trial_len:
            # check that block is correct, all predicted samples together
            # should be the trial samples
            all_predicted_samples = [
                range(stop - n_preds_per_input,
                      stop) for _,stop in start_stop_blocks]
            # this check takes about 50 ms in performance test
            # whereas loop itself takes only 5 ms.. deactivate it if not necessary
            assert np.array_equal(
                range(i_trial_starts[i_trial], i_trial_stops[i_trial]),
                np.unique(np.concatenate(all_predicted_samples)))

        start_stop_blocks_per_trial.append(start_stop_blocks)
    return start_stop_blocks_per_trial

def _get_start_stop_blocks_for_trial(i_trial_start, i_trial_stop,
                                     input_time_length, n_preds_per_input):

    """
    Compute start stop block inds for one trial
    Parameters
    ----------
    i_trial_start:  int
        Index of first sample to predict(!).
    i_trial_stops: 1daray/list of int
        Index one past last sample to predict.
    input_time_length: int
    n_preds_per_input: int
    Returns
    -------
    start_stop_blocks: list of (int, int)
        A list of 2-tuples indicating start and stop index
        of the inputs needed to predict entire trial.
    """
    start_stop_blocks = []
    i_window_stop = i_trial_start  # now when we add sample preds in loop,
    # first sample of trial corresponds to first prediction
    while i_window_stop < i_trial_stop:
        i_window_stop += n_preds_per_input
        i_adjusted_stop = min(i_window_stop, i_trial_stop)
        i_window_start = i_adjusted_stop - input_time_length
        start_stop_blocks.append((i_window_start, i_adjusted_stop))

    return start_stop_blocks

def get_balanced_batches(n_trials, rng, shuffle, n_batches=None,
                         batch_size=None):
    """Create indices for batches balanced in size 
    (batches will have maximum size difference of 1).
    Supply either batch size or number of batches. Resulting batches
    will not have the given batch size but rather the next largest batch size
    that allows to split the set into balanced batches (maximum size difference 1).
    Parameters
    ----------
    n_trials : int
        Size of set.
    rng : RandomState
    shuffle : bool
        Whether to shuffle indices before splitting set.
    n_batches : int, optional
    batch_size : int, optional
    Returns
    -------
    """
    assert batch_size is not None or n_batches is not None
    if n_batches is None:
        n_batches = int(np.round(n_trials / float(batch_size)))

    if n_batches > 0:
        min_batch_size = n_trials // n_batches
        n_batches_with_extra_trial = n_trials % n_batches
    else:
        n_batches = 1
        min_batch_size = n_trials
        n_batches_with_extra_trial = 0
    assert n_batches_with_extra_trial < n_batches
    all_inds = np.array(range(n_trials))
    if shuffle:
        rng.shuffle(all_inds)
    i_start_trial = 0
    i_stop_trial = 0
    batches = []
    for i_batch in range(n_batches):
        i_stop_trial += min_batch_size
        if i_batch < n_batches_with_extra_trial:
            i_stop_trial += 1
        batch_inds = all_inds[range(i_start_trial, i_stop_trial)]
        batches.append(batch_inds)
        i_start_trial = i_stop_trial
    assert i_start_trial == n_trials
    return batches

def _create_batch_from_i_trial_start_stop_blocks(X, y, z, i_trial_start_stop_block,
                                                 n_preds_per_input=None):
    Xs = []
    ys = []
    zs = []
    for i_trial, start, stop in i_trial_start_stop_block:
        Xs.append(X[i_trial][:,start:stop])
        if not hasattr(y[i_trial], '__len__'):
            ys.append(y[i_trial])
            if type(z) == np.ndarray:
                zs.append(z[i_trial])
        else:
            assert n_preds_per_input is not None
            ys.append(y[i_trial][stop-n_preds_per_input:stop])
            if type(z) == np.ndarray:
                zs.append(z[i_trial][stop-n_preds_per_input:stop])
            
    batch_X = np.array(Xs)
    batch_y = np.array(ys)
    batch_z = np.array(zs)
    # add empty fourth dimension if necessary
    if batch_X.ndim == 3:
        batch_X = batch_X[:,:,:, None]
    return batch_X, batch_y, batch_z