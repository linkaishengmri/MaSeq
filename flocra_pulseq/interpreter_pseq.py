# -*- coding: utf-8 -*-
# pulseq_assembler.py
# Written by Kaisheng Lin based on codes from Lincoln Craven-Brightman

import numpy as np
import logging # For errors
from flocra_pulseq.interpreter import PSInterpreter
class PseqInterpreter(PSInterpreter):
    

    def __init__(self, rf_center=3e+6, rf_amp_max=5e+3, grad_max=1e+7,
                 gx_max=None, gy_max=None, gz_max=None,
                 clk_t=1/122.88, tx_t=123/122.88, grad_t=1229/122.88,
                 tx_warmup=1000, tx_zero_end=True, grad_zero_end=True,
                 log_file = 'pseq_interpreter', log_level = 20,
                 orientation = [2,0,1], # grad orientation 
                 blank_time = 1000, # us, blank time before RF start to ensure that tx_warmup work properly
                 rx_gate_mode = 0, # 0: no rx_gate output, 1: [TODO] set rx_gate as 2nd TX_gate 
                 tx_ch = 0, # TX channel index: either 0 or 1
                 rx_ch = 0, # RX channel index: either 0 or 1. [TODO] Both 0 and 1 is under construction
                 grad_eff = [0.4113, 0.9094,1.0000],  # gradient coefficient of efficiency
                 use_multi_freq = False,):
        """
        Create PSInterpreter object for FLOCRA with system parameters.

        Args:
            rf_center (float): RF center (local oscillator frequency) in Hz.
            rf_amp_max (float): Default 5e+3 -- System RF amplitude max in Hz.
            grad_max (float): Default 1e+6 -- System gradient max in Hz/m.
            gx_max (float): Default None -- System X-gradient max in Hz/m. If None, defaults to grad_max.
            gy_max (float): Default None -- System Y-gradient max in Hz/m. If None, defaults to grad_max.
            gz_max (float): Default None -- System Z-gradient max in Hz/m. If None, defaults to grad_max.
            clk_t (float): Default 1/122.88 -- System clock period in us.
            tx_t (float): Default 123/122.88 -- Transmit raster period in us.
            grad_t (float): Default 1229/122.88 -- Gradient raster period in us.
            tx_warmup (float): Default 1000 -- Warmup time to turn on tx_gate before Tx events in us.
            tx_zero_end (bool): Default True -- Force zero at the end of RF shapes
            grad_zero_end (bool): Default True -- Force zero at the end of Gradient/Trap shapes
            log_file (str): Default 'ps_interpreter' -- File (.log appended) to write run log into.
            log_level (int): Default 20 (INFO) -- Logger level, 0 for all, 20 to ignore debug.
            orientation (list): Default [2,0,1] -- Specifies the gradient direction of the frequency/Phase/Slice encoding
            blank_time (float): Default 1000 -- Blank time before RF start to ensure that tx_warmup work properly 
            rx_gate_mode (int): Default 0 -- no rx gate output
            tx_ch (int): Default 0 -- Channel index: either 0 or 1
            rx_ch (int): Default 0 -- RX channel index: either 0 or 1. 
            grad_eff (list): Default [0.4113, 0.9094,1.0000] -- Gradient coefficient of efficiency

        """
        super().__init__( rf_center=rf_center, rf_amp_max=rf_amp_max, grad_max=grad_max,
                 gx_max=gx_max, gy_max=gy_max, gz_max=gz_max,
                 clk_t=clk_t, tx_t=tx_t, grad_t=grad_t,
                 tx_warmup=tx_warmup, tx_zero_end=tx_zero_end, grad_zero_end=grad_zero_end,
                 log_file = log_file, log_level = log_level)
        self._orientation = orientation
        self._global_or_str = ['grad_vx', 'grad_vy', 'grad_vz']
        self._blank_time = blank_time
        self._rx_gate_mode = rx_gate_mode
        self._tx_ch = tx_ch
        self._rx_ch = rx_ch
        self._grad_eff = grad_eff
        self._use_multi_freq = use_multi_freq
        self._freq_offset = {}
         # Redefine var_name
        self._var_names = ('tx0', 'tx1', 'grad_vx', 'grad_vy', 'grad_vz', 'grad_vz2',
         'rx0_en', 'rx1_en', 'rx_gate' ,'tx_gate')
        if use_multi_freq:
            self._var_names = self._var_names + ('lo0_freq_offset', 'lo1_freq_offset', 'lo0_rst', 'lo1_rst')
            
    # Encode all blocks
    def _stream_all_blocks(self):
        """
        Encode all blocks into sequential time updates.

        Returns:
            dict: tuples of np.ndarray times, updates with variable name keys
            int: number of sequence readout points
        """

       

        # Prep containers, zero at start
        out_data = {}
        times = {var: [np.zeros(1)] for var in self._var_names}
        updates = {var: [np.zeros(1)] for var in self._var_names}
        start = self._blank_time
        readout_total = 0

        # Encode all blocks
        for block_id in self._blocks.keys():
            self._logger.info(f'streaming block {block_id} ...')
            if self._version_major == 1 and self._version_minor >= 4:
                var_dict, duration, readout_num = self._stream_block_v3(block_id)
            else:
                var_dict, duration, readout_num = self._stream_block(block_id)

            for var in self._var_names:
                times[var].append(var_dict[var][0] + start)
                updates[var].append(var_dict[var][1])

            start += duration
            readout_total += readout_num

        # Clean up final arrays
        for var in self._var_names:
            # Make sure times are ordered, and overwrite duplicates to last inserted update
            time_sorted, unique_idx = np.unique(np.flip(np.concatenate(times[var])), return_index=True)
            update_sorted = np.flip(np.concatenate(updates[var]))[unique_idx]

            # Compressed repeated values
            update_compressed_idx = np.concatenate([[0], np.nonzero(update_sorted[1:] - update_sorted[:-1])[0] + 1])
            update_arr = update_sorted[update_compressed_idx]
            time_arr = time_sorted[update_compressed_idx]

            # Zero everything at end
            time_arr = np.concatenate((time_arr, np.zeros(1) + start))
            update_arr = np.concatenate((update_arr, np.zeros(1)))

            out_data[var] = (time_arr, update_arr)

        self._logger.info(f'done')
        return (out_data, readout_total)

    # Convert individual block into PR commands (duration, gates), TX offset, and GRAD offset
    def _stream_block(self, block_id):
        """
        Encode block into sequential time updates

        Args:
            block_id (int): Block id key for block in object dict memory to be encoded

        Returns:
            dict: tuples of np.ndarray times, updates with variable name keys
            float: duration of the block
            int: readout count for the block
        """
        out_dict = {var: [] for var in self._var_names}
        readout_num = 0
        duration = 0

        block = self._blocks[block_id]
        # Preset all variables
        for var in self._var_names:
             out_dict[var] = (np.zeros(0, dtype=int),) * 2

        # Minimum duration of block
        if block['delay'] != 0:
            duration = max(duration, self._delay_events[block['delay']])

        # Tx and Tx gate updates
        tx_id = block['rf']
        if tx_id != 0: 
            out_dict['tx0'] = (self._tx_times[tx_id], self._tx_data[tx_id])
            duration = max(duration, self._tx_durations[tx_id])
            tx_gate_start = self._tx_times[tx_id][0] - self._tx_warmup
            self._error_if(tx_gate_start < 0,
                f'Tx warmup ({self._tx_warmup}) of RF event {tx_id} is longer than delay ({self._tx_times[tx_id][0]})')
            out_dict['tx_gate'] = (np.array([tx_gate_start, self._tx_durations[tx_id]]),
                np.array([1, 0]))

        # Gradient updates
        for grad_ch in ('gx', 'gy', 'gz'):
            grad_id = block[grad_ch]
            if grad_id != 0:
                grad_var_name = grad_ch[0] + 'rad_v' + grad_ch[1] # To get the correct varname for output g[CH] -> grad_v[CH]
                self._error_if(np.any(np.abs(self._grad_data[grad_id] / self._grad_max[grad_ch]) > 1), 
                    f'Gradient event {grad_id} for {grad_ch} in block {block_id} is larger than {grad_ch} max')
                out_dict[grad_var_name] = (self._grad_times[grad_id], self._grad_data[grad_id] / self._grad_max[grad_ch])
                duration = max(duration, self._grad_durations[grad_id])

        # Rx updates
        rx_id = block['adc']
        if rx_id != 0:
            rx_event = self._adc_events[rx_id]
            rx_start = rx_event['delay']
            rx_end = rx_start + rx_event['num'] * self._rx_t
            readout_num += rx_event['num']
            out_dict['rx0_en'] = (np.array([rx_start, rx_end]), np.array([1, 0]))
            duration = max(duration, rx_end)

        # Return durations for each PR and leading edge values
        return (out_dict, duration, int(readout_num))
    #endregion

    # Convert individual block into PR commands (duration, gates), TX offset, and GRAD offset
    def _stream_block_v2(self, block_id):
        """
        Encode block into sequential time updates

        Args:
            block_id (int): Block id key for block in object dict memory to be encoded

        Returns:
            dict: tuples of np.ndarray times, updates with variable name keys
            float: duration of the block
            int: readout count for the block
        """
        out_dict = {var: [] for var in self._var_names}
        readout_num = 0
        duration = 0

        block = self._blocks[block_id]
        # Preset all variables
        for var in self._var_names:
             out_dict[var] = (np.zeros(0, dtype=int),) * 2

        # Minimum duration of block
        duration = block['dur'] * self._definitions['BlockDurationRaster'] * 1e6
        # if block['delay'] != 0:
            # duration = max(duration, self._delay_events[block['delay']])

        # Tx and Tx gate updates
        tx_id = block['rf']
        if tx_id != 0: 
            out_dict['tx0'] = (self._tx_times[tx_id], self._tx_data[tx_id])
            # duration = max(duration, self._tx_durations[tx_id])
            tx_gate_start = self._tx_times[tx_id][0] - self._tx_warmup
            self._error_if(tx_gate_start < 0,
                f'Tx warmup ({self._tx_warmup}) of RF event {tx_id} is longer than delay ({self._tx_times[tx_id][0]})')
            out_dict['tx_gate'] = (np.array([tx_gate_start, self._tx_durations[tx_id]]),
                np.array([1, 0]))
            

        # Gradient updates
        for grad_ch in ('gx', 'gy', 'gz'):
            grad_id = block[grad_ch]
            if grad_id != 0:
                grad_var_name = grad_ch[0] + 'rad_v' + grad_ch[1] # To get the correct varname for output g[CH] -> grad_v[CH]
                self._error_if(np.any(np.abs(self._grad_data[grad_id] / self._grad_max[grad_ch]) > 1), 
                    f'Gradient event {grad_id} for {grad_ch} in block {block_id} is larger than {grad_ch} max')
                out_dict[grad_var_name] = (self._grad_times[grad_id], self._grad_data[grad_id] / self._grad_max[grad_ch])
                # duration = max(duration, self._grad_durations[grad_id])

        # Rx updates
        rx_id = block['adc']
        if rx_id != 0:
            rx_event = self._adc_events[rx_id]
            rx_start = rx_event['delay']
            rx_end = rx_start + rx_event['num'] * self._rx_t
            readout_num += rx_event['num']
            out_dict['rx0_en'] = (np.array([rx_start, rx_end]), np.array([1, 0]))
            # duration = max(duration, rx_end)

        # Return durations for each PR and leading edge values
        return (out_dict, duration, int(readout_num))
    #endregion

    # Function stream_block_v3 is based on v2 but add some channel selector
    def _stream_block_v3(self, block_id):
        """
        Encode block into sequential time updates

        Args:
            block_id (int): Block id key for block in object dict memory to be encoded

        Returns:
            dict: tuples of np.ndarray times, updates with variable name keys
            float: duration of the block
            int: readout count for the block
        """

        # channel selector
        rx_ch_name = 'rx0_en' if self._rx_ch == 0 else 'rx1_en'
        tx_ch_name = 'tx0' if self._tx_ch == 0 else 'tx1'
        

        out_dict = {var: [] for var in self._var_names}
        readout_num = 0
        duration = 0

        block = self._blocks[block_id]
        # Preset all variables
        for var in self._var_names:
             out_dict[var] = (np.zeros(0, dtype=int),) * 2

        # Minimum duration of block
        duration = block['dur'] * self._definitions['BlockDurationRaster'] * 1e6
        # if block['delay'] != 0:
            # duration = max(duration, self._delay_events[block['delay']])

        # Tx and Tx gate updates
        tx_id = block['rf']
        if tx_id != 0: 
            out_dict[tx_ch_name] = (self._tx_times[tx_id], self._tx_data[tx_id])
            # duration = max(duration, self._tx_durations[tx_id])
            tx_gate_start = self._tx_times[tx_id][0] - self._tx_warmup

            # Tx warmup can be longer than block delay but this condition will still throw a warning instead of an error.
            self._warning_if(tx_gate_start < 0,
                f'Tx warmup ({self._tx_warmup}) of RF event {tx_id} is longer than delay ({self._tx_times[tx_id][0]})')
            out_dict['tx_gate'] = (np.array([tx_gate_start, self._tx_durations[tx_id]]),
                np.array([1, 0]))
            if self._use_multi_freq:
                lofreq_start_time = self._tx_times[tx_id][0] - self._tx_warmup / 2 
                out_dict['lo'+tx_ch_name[2]+'_freq_offset'] = (np.array([lofreq_start_time]),np.array([self._freq_offset[tx_id]]))
                out_dict['lo'+tx_ch_name[2]+'_rst'] = (np.array([lofreq_start_time, lofreq_start_time+1/122.88]),np.array([1, 0]))
                
                


        # Gradient updates
        for grad_ch in ('gx', 'gy', 'gz'):
            grad_id = block[grad_ch]
            if grad_id != 0:
                # grad_var_name = grad_ch[0] + 'rad_v' + grad_ch[1] # To get the correct varname for output g[CH] -> grad_v[CH]
                grad_channel_idx = {'x': 0, 'y': 1, 'z': 2}.get(grad_ch[1], "Invalid input")
                target_idx = self._orientation[grad_channel_idx]
                grad_var_name = self._global_or_str[target_idx]
                self._error_if(np.any(np.abs(self._grad_data[grad_id] / self._grad_max[grad_ch]) > 1), 
                    f'Gradient event {grad_id} for {grad_ch} in block {block_id} is larger than {grad_ch} max')
                out_dict[grad_var_name] = (self._grad_times[grad_id], self._grad_data[grad_id] / self._grad_max[grad_var_name[0]+grad_var_name[-1]] * self._grad_eff[target_idx])
                # duration = max(duration, self._grad_durations[grad_id])

        # Rx updates
        rx_id = block['adc']
        if rx_id != 0:
            rx_event = self._adc_events[rx_id]
            rx_start = rx_event['delay']
            rx_end = rx_start + rx_event['num'] * self._rx_t
            readout_num += rx_event['num']
            out_dict[rx_ch_name] = (np.array([rx_start, rx_end]), np.array([1, 0]))
            # duration = max(duration, rx_end)

        # Return durations for each PR and leading edge values
        return (out_dict, duration, int(readout_num))
    #endregion
    # Wrapper for full compilation
    def interpret(self, pulseq_file):
        """
        Interpret FLOCRA array from PulSeq .seq file

        Args:
            pulseq_file (str): PulSeq file to compile from

        Returns:
            dict: tuple of numpy.ndarray time and update arrays, with variable name keys
            dict: parameter dictionary containing raster times, readout numbers, and any file-defined variables
        """
        self._logger.info(f'Interpreting ' + pulseq_file)
        if self.is_assembled:
            self._logger.info('Re-initializing over old sequence...')
            # [TODO]: change input params
            self.__init__(rf_center=self._rf_center, rf_amp_max=self._rf_amp_max, 
                gx_max=self._grad_max['gx'], gy_max=self._grad_max['gy'], gz_max=self._grad_max['gz'],
                clk_t=self._clk_t, tx_t=self._tx_t, grad_t=self._grad_t)
        self._read_pulseq(pulseq_file)
        self._compile_tx_data()
        self._compile_grad_data()
        self.out_data, self.readout_number = self._stream_all_blocks()
        self.is_assembled = True
        param_dict = {'readout_number' : self.readout_number, 'tx_t' : self._tx_t, 'rx_t' : self._rx_t, 'grad_t': self._grad_t}
        for key, value in self._definitions.items():
            if key in param_dict:
                self._logger.warning(f'Key conflict: overwriting key [{key}], value [{param_dict[key]}] with new value [{value}]')
            param_dict[key] = value
        print(f'read {len(self._definitions)} definitions, {len(self._blocks)} blocks, {len(self._shapes)} shapes, {len(self._adc_events)} adc events, {len(self._rf_events)} rf events, {len(self._grad_events)} gradient shapes')
        return (self.out_data, param_dict)
    
    def _read_pulseq(self, pulseq_file):
        # Open file
        with open(pulseq_file) as f:
            self._logger.info('Opening PulSeq file...')
            line = '\n'
            next_line = ''

            while True:
                if not next_line:
                    line = f.readline()
                else:
                    line = next_line
                    next_line = ''
                if line == '': break
                key = self._simplify(line)
                if key in self._pulseq_keys:
                    next_line = self._pulseq_keys[key](f)

        # Check that all ids are valid
        self._logger.info('Validating ids...')
        if self._version_major == 1 and self._version_minor <= 3:
            var_names = ('delay', 'rf', 'gx', 'gy', 'gz', 'adc', 'ext')
            var_dicts = [self._delay_events, self._rf_events, self._grad_events, self._grad_events, self._grad_events, self._adc_events, {}]
            for block in self._blocks.values():
                for i in range(len(var_names)):
                    id_n = block[var_names[i]]
                    self._error_if(id_n != 0 and id_n not in var_dicts[i], f'Invalid {var_names[i]} id: {id_n}')
            for rf in self._rf_events.values():
                self._error_if(rf['mag_id'] not in self._shapes, f'Invalid magnitude shape id: {rf["mag_id"]}')
                self._error_if(rf['phase_id'] not in self._shapes, f'Invalid phase shape id: {rf["phase_id"]}')
            for grad in self._grad_events.values():
                if len(grad) == 3:
                    self._error_if(grad['shape_id'] not in self._shapes, f'Invalid grad shape id: {grad["shape_id"]}')
            self._logger.info('Valid ids')

            # Check that all delays are multiples of clk_t
            for events in [self._blocks.values(), self._rf_events.values(), self._grad_events.values(),
                            self._adc_events.values()]:
                for event in events:
                    self._warning_if(int(event['delay'] / self._clk_t) * self._clk_t != event['delay'],
                        f'Event delay {event["delay"]} is not a multiple of clk_t')
            for delay in self._delay_events.values():
                self._warning_if(int(delay / self._clk_t) * self._clk_t != delay,
                    f'Delay event {delay} is not a multiple of clk_t')
        else:
            # version >= 1.4
            var_names = ('rf', 'gx', 'gy', 'gz', 'adc', 'ext')
            var_dicts = [self._rf_events, self._grad_events, self._grad_events, self._grad_events, self._adc_events, {}]
            for block in self._blocks.values():
                for i in range(len(var_names)):
                    id_n = block[var_names[i]]
                    self._error_if(id_n != 0 and id_n not in var_dicts[i], f'Invalid {var_names[i]} id: {id_n}')
            for rf in self._rf_events.values():
                self._error_if(rf['mag_id'] not in self._shapes, f'Invalid magnitude shape id: {rf["mag_id"]}')
                self._error_if(rf['phase_id'] not in self._shapes, f'Invalid phase shape id: {rf["phase_id"]}')
            for grad in self._grad_events.values():
                if len(grad) == 3:
                    self._error_if(grad['shape_id'] not in self._shapes, f'Invalid grad shape id: {grad["shape_id"]}')
            self._logger.info('Valid ids')

        if self._use_multi_freq:
            for rf_id, rf in self._rf_events.items():
                self._freq_offset[rf_id] = rf['freq'] * 1e-6
                
        else:
            # Check that RF/ADC (TX/RX) only have one frequency offset -- can't be set within one file.
            freq = None
            base_id = None
            base_str = None
            for rf_id, rf in self._rf_events.items():
                if freq is None:
                    freq = rf['freq']
                    base_id = rf_id
                    base_str = 'RF'
                self._error_if(rf['freq'] != freq, f"Frequency offset of RF event {rf_id} ({rf['freq']}) doesn't match that of {base_str} event {base_id} ({freq})")
            for adc_id, adc in self._adc_events.items():
                if freq is None:
                    freq = adc['freq']
                    base_id = adc_id
                    base_str = 'ADC'
                self._error_if(adc['freq'] != freq, f"Frequency offset of ADC event {adc_id} ({adc['freq']}) doesn't match that of {base_str} event {base_id} ({freq})")
            if freq is not None and freq != 0:
                self._rf_center += freq
                self._logger.info(f'Adding freq offset {freq} Hz. New center / linear oscillator frequency: {self._rf_center}')

        # Check that ADC has constant dwell time
        dwell = None
        for adc_id, adc in self._adc_events.items():
            if dwell is None:
                dwell = adc['dwell']/1000
                base_id = adc_id
            self._error_if(adc['dwell']/1000 != dwell, f"Dwell time of ADC event {adc_id} ({adc['dwell']}) doesn't match that of ADC event {base_id} ({dwell})")
        if dwell is not None:
            self._rx_div = np.round(dwell / self._clk_t).astype(int)
            self._rx_t = self._clk_t * self._rx_div
            self._warning_if(self._rx_div * self._clk_t != dwell,
                f'Dwell time ({dwell}) rounded to {self._rx_t}, multiple of clk_t ({self._clk_t})')

        self._logger.info('PulSeq file loaded')

        


# Sample usage
if __name__ == '__main__':
    ps = PSInterpreter(grad_t=1)
    inp_file = '../mgh-flocra/test_sequences/tabletop_radial_v2_2d_pulseq.seq'
    out_data, params = ps.interpret(inp_file)

    import matplotlib.pyplot as plt

    names = [' tx', ' gx', ' gy', ' gz', 'adc']
    data = [out_data['tx0'], out_data['grad_vx'], out_data['grad_vy'], out_data['grad_vz'], out_data['tx_gate']]

    for i in range(5):
        print(f'{names[i]} minimum entry difference magnitude: {np.min(np.abs(data[i][1][1:] - data[i][1][:-1]))}')
        print(f'{names[i]} entries below 1e-6 difference: {np.sum(np.abs(data[i][1][1:] - data[i][1][:-1]) < 1e-6)}')
        print(f'{names[i]} entries below 1e-5 difference: {np.sum(np.abs(data[i][1][1:] - data[i][1][:-1]) < 1e-5)}')
        print(f'{names[i]} entries below 1e-4 difference: {np.sum(np.abs(data[i][1][1:] - data[i][1][:-1]) < 1e-4)}')
        print(f'{names[i]} entries below 1e-3 difference: {np.sum(np.abs(data[i][1][1:] - data[i][1][:-1]) < 1e-3)}')
    
    print("Completed successfully")
