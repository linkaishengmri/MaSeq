import os
import sys
#*****************************************************************************
# Get the directory of the current script
main_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(main_directory)
parent_directory = os.path.dirname(parent_directory)

# Define the subdirectories you want to add to sys.path
subdirs = ['MaSeq', 'marcos_client']

# Add the subdirectories to sys.path
for subdir in subdirs:
    full_path = os.path.join(parent_directory, subdir)
    sys.path.append(full_path)
#******************************************************************************
import numpy as np
import matplotlib.pyplot as plt

from pypulseq.convert import convert
import configs.hw_config_pseq as hw
import dict_utils
import fid_result_display

from pulseq_assembler import PSAssembler

from local_config import grad_board
import experiment as ex
ROUNDING = 0 

class pulseq_interpreter(PSAssembler):
    def __init__(self, rf_center=3e+6, rf_amp_max=5e+3, grad_max=1e+7,
                 clk_t=7e-3, tx_t=1.001, grad_t=10.003,
                 pulseq_t_match=False, ps_tx_t=1, ps_grad_t=10,
                 rf_delay_preload=False, addresses_per_grad_sample=1,
                 tx_warmup=0, grad_pad=0, adc_pad=0,
                 rf_pad_type='ext', grad_pad_type='ext',orientation = [0,1,2]):
        super().__init__(rf_center, rf_amp_max, grad_max,
                 clk_t, tx_t, grad_t,
                 pulseq_t_match, ps_tx_t, ps_grad_t,
                 rf_delay_preload, addresses_per_grad_sample,
                 tx_warmup, grad_pad, adc_pad,
                 rf_pad_type, grad_pad_type)
        self._orientation = orientation
        self._global_or_str = ['grad_vx', 'grad_vy', 'grad_vz']
    def log(str):
        print(str)
    
    # Overwrite some assemble function
    # _read_shapes function in PSassembler is not available for both uncompressed and compressed shape. 
    def _read_shapes(self, f):
        """
        Read SHAPES (rastered shapes) section in PulSeq file f to object dict memory.
        Shapes are formatted with two header lines, followed by lines of single data points in compressed pulseq shape format

        Args:
            f (_io.TextIOWrapper): File pointer to read from

        Returns:
            str: Raw next line in file after section ends
        """
        rline = ''
        line = ''
        self._logger.info('Shapes: Reading...')
        while True:
            line = f.readline()
            rline = self._simplify(line)
            if line == '' or rline in self._pulseq_keys: break
            if len(rline.split()) == 2 and rline.split()[0].lower() == 'shape_id':
                shape_id = int(rline.split()[1])
                n = int(self._simplify(f.readline()).split()[1])
                self._warning_if(shape_id in self._shapes, f'Repeat shape ID {shape_id}, overwriting')
                self._shapes[shape_id] = np.zeros(n)

                x_temp=[]
                while True:
                    new_line=self._simplify(f.readline())
                    if new_line == '':
                        break
                    else:
                        x_temp.append(float(new_line))
                if(len(x_temp)==n):
                    self._shapes[shape_id] = np.array(x_temp)
                elif(len(x_temp)<n):
                    i = 0
                    prev = -2
                    x = 0
                    line_idx=0
                    while i < n:
                        dx = x_temp[line_idx]
                        line_idx = line_idx + 1
                        x += dx
                        self._warning_if(x > 1 or x < 0, f'Shape {shape_id} entry {i} is {x}, outside of [0, 1], rounding')
                        if x > 1:
                            x = 1
                        elif x < 0:
                            x = 0
                        self._shapes[shape_id][i] = x
                        if dx == prev:
                            r = int(x_temp[line_idx])
                            line_idx=line_idx+1
                            for _ in range(0, r):
                                i += 1
                                x += dx
                                self._warning_if(x > 1 or x < 0, f'Shape {shape_id} entry {i} is {x}, outside of [0, 1], rounding')
                                if x > 1:
                                    x = 1
                                elif x < 0:
                                    x = 0
                                self._shapes[shape_id][i] = x
                        i += 1
                        prev = dx

        self._logger.info('Shapes: Complete')

        return rline
     # [GRADIENTS] <id> <amp> <amp_shape_id> <time_shape_id> <delay>  
    def _read_grad_events(self, f):
        """
        Read GRADIENTS (gradient event) section in PulSeq file f to object dict memory.
        Gradient events are formatted like: <id> <amp> <shape_id> <delay>

        Args:
            f (_io.TextIOWrapper): File pointer to read from

        Returns:
            str: Raw next line in file after section ends
        """
        var_names = ('amp', 'amp_shape_id' , 'time_shape_id', 'delay')
        rline = ''
        line = ''
        self._logger.info('Gradients: Reading...')
        while True:
            line = f.readline()
            rline = self._simplify(line)
            if line == '' or rline in self._pulseq_keys: break

            tmp = rline.split()
            if len(tmp) == 5: # GRAD <id> <amp> <amp_shape_id> <time_shape_id> <delay>  
                data_line = [int(tmp[0]), float(tmp[1]), int(tmp[2]), int(tmp[3]), int(tmp[4])]
                self._warning_if(data_line[0] in self._grad_events, f'Repeat gradient ID {data_line[0]} in GRADIENTS, overwriting')
                self._grad_events[data_line[0]] = {var_names[i] : data_line[i+1] for i in range(len(var_names))}
            
        self._logger.info('Gradients: Complete')
        return rline

    # [TRAP] <id> <amp> <rise> <flat> <fall> <delay>
    def _read_trap_events(self, f):
        """
        Read TRAP (trapezoid gradient event) section in PulSeq file f to object dict memory.
        Trapezoid gradient events are formatted like: <id> <amp> <rise> <flat> <fall> <delay>

        Args:
            f (_io.TextIOWrapper): File pointer to read from

        Returns:
            str: Raw next line in file after section ends
        """
        var_names = ('amp', 'rise', 'flat', 'fall', 'delay')
        rline = ''
        line = ''
        self._logger.info('Trapezoids: Reading...')
        while True:
            line = f.readline()
            rline = self._simplify(line)
            if line == '' or rline in self._pulseq_keys: break

            tmp = rline.split()
            if len(tmp) == 6: # TRAP <id> <amp> <rise> <flat> <fall> <delay>
                data_line = [int(tmp[0]), float(tmp[1]), int(tmp[2]), int(tmp[3]), int(tmp[4]), float(tmp[5])]
                self._warning_if(data_line[0] in self._grad_events, f'Repeat gradient ID {data_line[0]} in TRAP, overwriting')
                self._grad_events[data_line[0]] = {var_names[i] : data_line[i+1] for i in range(len(var_names))}
            # elif len(tmp) == 5: # TRAP <id> <amp> <rise> <flat> <fall> NO DELAY
            #     data_line = [int(tmp[0]), float(tmp[1]), int(tmp[2]), int(tmp[3]), int(tmp[4])]
            #     data_line.append(0)
            #     self._warning_if(data_line[0] in self._grad_events, f'Repeat gradient ID {data_line[0]} in TRAP, overwriting')
            #     self._grad_events[data_line[0]] = {var_names[i] : data_line[i+1] for i in range(len(var_names))}

        self._logger.info('Trapezoids: Complete')

        return rline

    # [RF] <id> <amp> <mag_id> <phase_id> <time_id> <delay> <freq> <phase>
    # <time_id> is added
    def _read_rf_events(self, f):
        """
        Read RF (RF event) section in PulSeq file f to object dict memory.
        RF events are formatted like: <id> <amp> <mag_id> <phase_id> <time_id> <delay> <freq> <phase>

        Args:
            f (_io.TextIOWrapper): File pointer to read from

        Returns:
            str: Raw next line in file after section ends
        """
        var_names = ('amp', 'mag_id', 'phase_id', 'time_id', 'delay', 'freq', 'phase')
        rline = ''
        line = ''
        self._logger.info('RF: Reading...')
        while True:
            line = f.readline()
            rline = self._simplify(line)
            if line == '' or rline in self._pulseq_keys: break

            tmp = rline.split()
            if len(tmp) == 8: # <id> <amp> <mag_id> <phase_id> <time_id> <delay> <freq> <phase>
                data_line = [int(tmp[0]), float(tmp[1]), int(tmp[2]), int(tmp[3]), int(tmp[4]), int(tmp[5]), float(tmp[6]), float(tmp[7])]
                self._warning_if(data_line[0] in self._rf_events, f'Repeat RF ID {data_line[0]}, overwriting')
                self._rf_events[data_line[0]] = {var_names[i] : data_line[i+1] for i in range(len(var_names))}

        self._logger.info('RF: Complete')

        return rline
    def _compile_tx_data(self):
        """
        Compile transmit data from object dict memory into bytes
        """

        self._logger.info('Compiling Tx data...')
        
        self._ps_tx_t = self._definitions['RadiofrequencyRasterTime'] *1e6
        self._tx_t = self._ps_tx_t 

        tx_data = []
        curr_offset = 0
        self._tx_out = {}
    

        # Process each rf event
        for tx_id, tx in self._rf_events.items():
            # Collect mag/phase shapes
            mag_shape = self._shapes[tx['mag_id']]
            phase_shape = self._shapes[tx['phase_id']]
            if len(mag_shape) != len(phase_shape):
                self._logger.warning(f'Tx envelope of RF event {tx_id} has mismatched magnitude and phase information,'
                                    ' the last entry of the shorter will be extended')

            # Array length, unitless -- extends shorter of phase/mag shape to length of longer                     
            pulse_len = int((max(len(mag_shape), len(phase_shape)) - 1) * self._ps_tx_t / self._tx_t) + 1 # unitless
            
            # Interpolate values on falling edge (and extend past end of shorter, if needed)
            x = np.flip(np.linspace(pulse_len * self._tx_t, 0, num=pulse_len, endpoint=False)) # us
            mag_x_ps = np.flip(np.linspace(len(mag_shape)* self._ps_tx_t, 0, num=len(mag_shape), endpoint=False))
            phase_x_ps = np.flip(np.linspace(len(phase_shape)* self._ps_tx_t, 0, num=len(phase_shape), endpoint=False))
            mag_interp = np.interp(x, mag_x_ps, mag_shape) * tx['amp'] / self._rf_amp_max
            phase_interp = np.interp(x, phase_x_ps, phase_shape) * 2 * np.pi

            # Add tx warmup padding
            pulse_len += self._tx_warmup_samples
            tx_env = np.zeros(pulse_len, dtype=np.complex64)

            # Convert to complex tx envelope
            tx_env[self._tx_warmup_samples:] = np.exp((phase_interp + tx['phase']) * 1j) * mag_interp
            
            if np.any(np.abs(tx_env) > 1.0):
                self._logger.warning(f'Magnitude of RF event {tx_id} was too large, 16-bit signed overflow will occur')
            
            # Concatenate tx data and track offsets
            tx_data.extend(tx_env.tolist())
            self._tx_offsets[tx_id] = curr_offset
            self._tx_durations[tx_id] = pulse_len * self._tx_t
            self._tx_delays[tx_id] = tx['delay']
            curr_offset += pulse_len


            # compile to Marcos format
            self._tx_out[tx_id]=(x, tx_env)

            
            # plt.plot(x,tx_env)
            # plt.show()
        
        # Compile as bytes (16 bits for real and imaginary)
        self._logger.info('Rx Compiled!')

        tx_arr = np.array(tx_data)
        self.tx_arr = tx_arr

        


        # delete OCRA format
        # tx_arr = np.array(tx_data)
        # # Save TX array for external use
        # self.tx_arr = tx_arr
        # temp_bytearray = bytearray(4 * tx_arr.size)

        # tx_i = np.round(32767 * tx_arr.real).astype(np.uint16)
        # tx_q = np.round(32767 * tx_arr.imag).astype(np.uint16)

        # temp_bytearray[::4] = (tx_i & 0xff).astype(np.uint8).tobytes()
        # temp_bytearray[1::4] = (tx_i >> 8).astype(np.uint8).tobytes()
        # temp_bytearray[2::4] = (tx_q & 0xff).astype(np.uint8).tobytes()
        # temp_bytearray[3::4] = (tx_q >> 8).astype(np.uint8).tobytes()

        # self.tx_bytes = bytes(temp_bytearray)
        self._logger.info('Tx data compiled')

    # 
    def _compile_grad_data(self):
        """
        Compile gradient events from object dict memory into Marcos dict
        """
        
        self._logger.info('Compiling gradient data...')
        
        

        grad_steps = hw.grad_steps
        
        self._grad_out = {}

        for key, event in self._grad_events.items():
            if len(event) == 4:
                amp_ref = event['amp'] / self._grad_max  # Normalize amplitude
                amp_shape_id = event['amp_shape_id']
                time_shape_id = event['time_shape_id']
                delay = event['delay']

                grad_dwell = self._definitions['GradientRasterTime'] * 1e6 # us

                amp_shape = self._shapes[amp_shape_id]
                time_shape = self._shapes[time_shape_id]
                if len(amp_shape) != len(time_shape):
                    self._logger.error(f'Gradien event {key} has mismatched amplitude and time length')

                time_vector = delay + grad_dwell * self._shapes[time_shape_id]
                amp_vector = amp_ref * self._shapes[amp_shape_id]
                self._grad_out[key] = (time_vector, amp_vector)
                #tmppp=0
            else:
                amp = event['amp'] / self._grad_max  # Normalize amplitude
                rise = event['rise']
                flat = event['flat']
                fall = event['fall']
                delay = event['delay']

                # Time points
                rise_times = np.linspace(0, rise, grad_steps + 1)[1:] + delay  # Exclude the start point
                fall_times = np.linspace(rise + flat + delay, rise + flat + delay + fall, grad_steps + 1)[1:]

                time_points = np.concatenate(([0], rise_times,  fall_times))

                # Amplitudes
                rise_amplitudes = np.linspace(0, amp, grad_steps + 1)[1:] 
                fall_amplitudes = np.linspace(amp, 0, grad_steps + 1)[1:] 
                
                amplitudes = np.concatenate(([0], rise_amplitudes,  fall_amplitudes))

                self._grad_out[key] = (time_points, amplitudes)
                #tmppp=0

        self._logger.info('Gradient data compiled')

    def _compile_adc_data(self):
        """
        Compile adc events from object dict memory into Marcos dict
        """
        
        self._logger.info('Compiling adc data...')
        self._adc_out = {}

        for key, event in self._adc_events.items():
            num = event['num'] 
            dwell = event['dwell']
            self._adc_dw = dwell * 1e-3 #us
            delay = event['delay']
            freq = event['freq']
            phase = event['phase']
            self._error_if(freq != 0 or phase != 0, f'_adc_events Invalid id: {key}. Freq and Phase must be 0.')
            
            # Time points
            time_points = np.array([delay, delay + dwell * 1e-3 *num])

            # Amplitudes
            amplitudes = np.array([1, 0])

            self._adc_out[key] = (time_points, amplitudes)
 

        self._logger.info('Adc data compiled')
        
    def _compile_integrate(self):
        """
        Compiles all event blocks into a single dict
        """
        event_dict = {  'tx0':[np.array([]), np.array([])],
                        'rx0_en':[np.array([]), np.array([])],
                        'grad_vx':[np.array([]), np.array([])],
                        'grad_vy':[np.array([]), np.array([])],
                        'grad_vz':[np.array([]), np.array([])],
                        'tx_gate':[np.array([]), np.array([])]
                        }
 
        accumulate_time = hw.RFgatePreTime * 1e6 # Initial time is RFgatePreTime
        
        block_raster_time_us = self._definitions['BlockDurationRaster'] * 1e6
        for block in self._blocks.values():
            delay = block['delay']
            rf_id = block['rf']
            gx_id = block['gx']
            gy_id = block['gy']
            gz_id = block['gz']
            adc_id = block['adc']
            ext_id = block['ext']
            
            if(rf_id != 0):
                cur_t = event_dict['tx0'][0]
                cur_a = event_dict['tx0'][1]
                cur_time_arr = accumulate_time + self._tx_out[rf_id][0]
                cur_st = np.append(cur_t, cur_time_arr)
                cur_at = np.append(cur_a, self._tx_out[rf_id][1])
                event_dict['tx0'][0] = cur_st
                event_dict['tx0'][1] = cur_at

                cur_gate_t = event_dict['tx_gate'][0]
                cur_gate_a = event_dict['tx_gate'][1]
                cur_gate_st = np.append(cur_gate_t, (np.array([cur_time_arr[0] - hw.RFgatePreTime * 1e6, cur_time_arr[-1]])))
                cur_gate_at = np.append(cur_gate_a, (np.array([1,0])))
                
                event_dict['tx_gate'][0] = cur_gate_st
                event_dict['tx_gate'][1] = cur_gate_at
                

            if(gx_id != 0):
                ind_str = self._global_or_str[self._orientation[0]]
                cur_t = event_dict[ind_str][0]
                cur_a = event_dict[ind_str][1]
                time_array = accumulate_time + self._grad_out[gx_id][0]
                if cur_t.size != 0 and cur_t[-1] == time_array[0]:
                    assert np.abs(cur_a[-1] - self._grad_out[gx_id][1][0]) < 1e-5
                    cur_st = np.append(cur_t, time_array[1:])
                    cur_at = np.append(cur_a, self._grad_out[gx_id][1][1:])
                else: 
                    cur_st = np.append(cur_t, time_array)
                    cur_at = np.append(cur_a, self._grad_out[gx_id][1])
                event_dict[ind_str][0] = cur_st
                event_dict[ind_str][1] = cur_at
            if(gy_id != 0):
                ind_str = self._global_or_str[self._orientation[1]]
                cur_t = event_dict[ind_str][0]
                cur_a = event_dict[ind_str][1]
                time_array = accumulate_time + self._grad_out[gy_id][0]
                if cur_t.size != 0 and cur_t[-1] == time_array[0]:
                    assert np.abs(cur_a[-1] - self._grad_out[gy_id][1][0]) < 1e-5
                    cur_st = np.append(cur_t, time_array[1:])
                    cur_at = np.append(cur_a, self._grad_out[gy_id][1][1:])
                else: 
                    cur_st = np.append(cur_t, time_array)
                    cur_at = np.append(cur_a, self._grad_out[gy_id][1])
                    event_dict[ind_str][0] = cur_st
                    event_dict[ind_str][1] = cur_at
            if(gz_id != 0):
                ind_str = self._global_or_str[self._orientation[2]]
                cur_t = event_dict[ind_str][0]
                cur_a = event_dict[ind_str][1]
                time_array = accumulate_time + self._grad_out[gz_id][0]
                if cur_t.size != 0 and cur_t[-1] == time_array[0]:
                    assert np.abs(cur_a[-1] - self._grad_out[gz_id][1][0]) < 1e-5
                    cur_st = np.append(cur_t, time_array[1:])
                    cur_at = np.append(cur_a, self._grad_out[gz_id][1][1:])
                else:
                    cur_st = np.append(cur_t, time_array)
                    cur_at = np.append(cur_a, self._grad_out[gz_id][1])
                    event_dict[ind_str][0] = cur_st
                    event_dict[ind_str][1] = cur_at
            if(adc_id != 0):
                cur_t = event_dict['rx0_en'][0]
                cur_a = event_dict['rx0_en'][1]
                cur_st = np.append(cur_t, accumulate_time + self._adc_out[adc_id][0])
                cur_at = np.append(cur_a, self._adc_out[adc_id][1])
                event_dict['rx0_en'][0] = cur_st
                event_dict['rx0_en'][1] = cur_at
            
            accumulate_time = accumulate_time + delay * block_raster_time_us 


        event_dict['tx0'] = tuple(event_dict['tx0'])
        event_dict['rx0_en'] = tuple(event_dict['rx0_en'])
        event_dict['grad_vx'] = (event_dict['grad_vx'][0], event_dict['grad_vx'][1]*hw.gradFactor[0])
        event_dict['grad_vy'] = (event_dict['grad_vy'][0], event_dict['grad_vy'][1]*hw.gradFactor[1])
        event_dict['grad_vz'] = (event_dict['grad_vz'][0], event_dict['grad_vz'][1]*hw.gradFactor[2])

        # Set up gate variables
        self._logger.info('Return dict!')
        self._event_dict = event_dict
        return event_dict

 
    def assemble(self, pulseq_file, byte_format=True):
        """
        Assemble Marcos dict from PulSeq .seq file

        Args:
            pulseq_file (str): PulSeq file to assemble from
            byte_format (bool): Default True -- Return transmit and gradient data in bytes, rather than numpy.ndarray
        
        Returns:
            dict
        """
        self._logger.info(f'Assembling ' + pulseq_file)
        if self.is_assembled:
            self._logger.info('Overwriting old sequence...')
        
        self._read_pulseq(pulseq_file)
        self._compile_tx_data()
        self._compile_grad_data()
        self._compile_adc_data()
        self._compile_integrate()
        self.is_assembled = True

    def _read_pulseq(self, pulseq_file):
        """
        Read PulSeq file into object dict memory

        Args:
            pulseq_file (str): PulSeq file to assemble from
        """
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
        var_names = ('delay', 'rf', 'gx', 'gy', 'gz', 'adc', 'ext')
        var_dicts = [self._delay_events, self._rf_events, self._grad_events, self._grad_events, self._grad_events, self._adc_events, {}]
        # for block in self._blocks.values():
        #    for i in range(len(var_names)):
        #        id_n = block[var_names[i]]
        #        self._error_if(id_n != 0 and id_n not in var_dicts[i], f'Invalid {var_names[i]} id: {id_n}')
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
                self._warning_if(int(event['delay'] / self._clk_t + ROUNDING) * self._clk_t != event['delay'],
                    f'Delay is not a multiple of clk_t, rounding')
        for delay in self._delay_events.values():
            self._warning_if(int(delay / self._clk_t + ROUNDING) * self._clk_t != delay,
                f'Delay is not a multiple of clk_t, rounding')
        
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











def reset_grad():
    expt = ex.Experiment(lo_freq=5, halt_and_reset=True, grad_max_update_rate = 0.2, init_gpa=True)
    expt.add_flodict({ 
                        'grad_vx':[np.array([100, 100000]), np.array([.01,0])],
                        'grad_vy':[np.array([100, 100000]), np.array([.01,0])],
                        'grad_vz':[np.array([100, 100000]), np.array([.01,0])],
                        'grad_vz2':[np.array([100, 100000]), np.array([.01,0])],
                        
                        })
    rxd, msgs = expt.run()
    expt.close_server(only_if_sim=True)
    expt._s.close() # close socket on client

def interpreter_test(seq_para, plot_seq, lo_freq = 10.58, output_seq_picture = False):
    expt = ex.Experiment(lo_freq=lo_freq, rx_t=seq_para._adc_dw, halt_and_reset=True, grad_max_update_rate = 0.2, init_gpa=False)
    
    dict_pre = seq_para._event_dict
    # del dict_pre['grad_vx']
    # dict_pre['grad_vx'] = dict_pre['grad_vz'] 
    # del dict_pre['grad_vz']

    # delete null key
    keys_to_delete = [key for key, (arr1, arr2) in dict_pre.items() if arr1.size == 0 or arr2.size == 0]
    for key in keys_to_delete:
        del dict_pre[key]
    
    expt.add_flodict(dict_pre)
    if output_seq_picture or plot_seq:
        expt.plot_sequence()
        if output_seq_picture:
            plt.savefig('./interpreter/seq_pic/seq.png')
        if plot_seq:
            plt.show()
    print('Sequence is ready for running.')
    rxd, msgs = expt.run()
    #print(rxd)
    #rxplot(rxd)
    expt.close_server(only_if_sim=True)
    expt._s.close() # close socket on client
    return rxd, msgs


if __name__ == '__main__':

    max_grad_Hz = convert(from_value=hw.max_grad, from_unit='mT/m', gamma=hw.gammaB, to_unit='Hz/m')
    max_rf_Hz = hw.max_rf * 1e-6 * hw.gammaB
    tse_2_seq = pulseq_interpreter(rf_amp_max=max_rf_Hz, grad_max=max_grad_Hz, orientation = [2,1,0])
    tse_2_seq.assemble("./interpreter_pseq/seq_demo/se.seq")
    tse_2_seq._event_dict
    tse_2_seq._adc_out

    rxd, msgs = interpreter_test(seq_para = tse_2_seq, plot_seq = True)
    dict_utils.save_dict(rxd, 'se')
    fid_result_display.fid_result_display(data = rxd['rx0'],dwell=tse_2_seq._adc_dw * 1e-6)

