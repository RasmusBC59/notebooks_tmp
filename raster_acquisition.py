# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 14:17:52 2020

@author: QCoDeS_Public
"""

#from qcodes_ms_drivers.MDAC.MDAC import MDAC
from qcodes import Instrument, Measurement
import numpy as np
from functools import partial

from panel.widgets import Button, Toggle
from panel import Row, Column
import time
import numpy as np
import pandas as pd
import holoviews as hv

from holoviews import opts
from holoviews.streams import Pipe, Buffer

from tornado.ioloop import PeriodicCallback
from tornado import gen

#%%

class UHFLIRasterAcquisition(Instrument):
    
    def __init__(self, name: str,
                 setpoints_list,
                 uhfli, pixel_period,
                 demodulators, signals, trigger_input,
                 repetitions = 1, delay = 0., timeout = 0.,
                 **kwargs):
        super().__init__(name, **kwargs)
        self.setpoints = setpoints_list
        self.uhfli = uhfli
        self.pixel_period = pixel_period
        self.uhfli.daq.signals_clear()
        self.demodulators = [f'demod{i}' for i in demodulators]
        self.signals = signals
        self._signals_list = []
        self.trigger_input = trigger_input
        self.repetitions = repetitions
        self.delay = delay
        self.setup_parameters()
        self.timeout = timeout
        
    def setup_parameters(self):
        # add parameters
        for demod in self.demodulators:
            for sig in self.signals:
                self.add_parameter(demod + '_' + sig,
                                   parameter_class = ParameterWithMutableSetpoints,
                                   setpoints_list = self.setpoints)
                getattr(self, demod + '_' + sig).path = self.uhfli.daq.signals_add(demod, sig)
                getattr(self, demod + '_' + sig).get = partial(self._get_demod_result,
                                                               getattr(self, demod + '_' + sig).path)
                self._signals_list.append(demod + '_' + sig)
        
    def setup_daq(self):
        if self.trigger_input != 'continuous': 
            _ = self.uhfli.daq.trigger(self.demodulators[0], self.trigger_input)
            self.uhfli.daq.type('hardware')
        else:
            self.uhfli.daq.type('continuous')
        self.uhfli.daq.delay(self.delay)
        self.uhfli.daq.grid_repetitions(self.repetitions)
        # setup the grid
        self.uhfli.daq.grid_rows(1)
        self.uhfli.daq.grid_cols(reduce(lambda a, b: a * b, [i.shape[0] for i in self.setpoints]))
        self.uhfli.daq.grid_mode('linear')
        self.uhfli.daq.duration(reduce(lambda a, b: a * b, [i.shape[0] for i in self.setpoints]) * float(self.pixel_period()))
        
    def acquire(self):
        if self.timeout:
            t_out = self.timeout
        else:
            t_out = (1. + 2.* self.uhfli.daq.duration())
        self.uhfli.daq.measure(verbose = False, timeout = t_out)
        
    def _get_demod_result(self, path):
        try:
            return np.reshape(self.uhfli.daq.results[path].value,
                              reduce(lambda a, b: a + b, [i.shape for i in self.setpoints]))
        except KeyError:
            pass
    
    def measure(self, setup = False, acquire = True):
        if setup: self.setup_daq()
        meas = Measurement()
        for signal in self._signals_list:
            meas.register_parameter(getattr(self, signal))
        setpoint_meshgrid = getattr(self, self._signals_list[0]).setpoint_meshgrid()
        results = [(i, j) for i,j in zip(self.setpoints, setpoint_meshgrid)]
        if acquire: self.acquire()
        for signal in self._signals_list:
            results.append((getattr(self, signal), getattr(self, signal).get()))
        with meas.run() as datasaver:
            datasaver.add_result(*results)

    def measure_1d(self, meas_param, param_start, param_stop, param_npts, param_delay,
                   setup = False):
        if setup: self.setup_daq()
        meas_param.post_delay = param_delay
    
        meas_param_setpoints = Mutable1DSetpoints(meas_param.name, param_start,
                                                  param_stop, param_npts)
        expanded_param = [ParameterWithMutableSetpoints(i.name, [meas_param_setpoints] + i.setpoints) \
                          for i in [getattr(self, j) for j in self._signals_list]]
        results_holder = np.zeros((len(self._signals_list),) + expanded_param[0].shape)
        for i, val in enumerate(meas_param_setpoints.get()):
            meas_param.set(val)
            self.acquire()
            for j, signal in enumerate(self._signals_list):
                results_holder[j, i, :] = getattr(self, signal).get()
            
        meas = Measurement()
        for param in expanded_param:
            meas.register_parameter(param)
        setpoint_meshgrid = expanded_param[0].setpoint_meshgrid()
        results = [(i, j) for i,j in zip(expanded_param[0].setpoints,
                                         setpoint_meshgrid)]
        for i, signal in enumerate(self._signals_list):
            results.append((getattr(self, signal), results_holder[i,:,:]))
        with meas.run() as datasaver:
            datasaver.add_result(*results)
        
    def video_mode(self, signal = '', refresh_period = 100, port = 12345):
        if len(self.setpoints) != 2:
            raise RuntimeError('Only implemented for 2D grids')
        
        @gen.coroutine
        def data_grabber():
            if self.uhfli.daq._daq_module._module._module.finished():
                tmp_result = self.uhfli.daq._daq_module._module._module.read(flat=True)
                if getattr(self,signal).path in tmp_result.keys():
                    self.uhfli.daq._daq_module._get_result_from_dict(tmp_result)
                pipe.send((self.outer_setpoints.get(),
                           self.inner_setpoints.get(),
                           getattr(self, signal).get().T))
                self.uhfli.daq._daq_module._module._module.execute()

            
        def close_server_click(event):
            video_mode_server.stop()
            
        def save_measurement(event):
            self.measure(setup = False, acquire = False)
            
        def toggle_periodic_callback(event):
            if event.new is True:
                self.uhfli.daq._daq_module._module._module.subscribe(getattr(self,signal).path)
                video_mode_callback.start()
            else:
                self.uhfli.daq._daq_module._module._module.unsubscribe('*')
                video_mode_callback.stop()
        
        close_button = Button(name='close server',button_type = 'primary',
                              width=100)
        close_button.on_click(close_server_click)
        save_button = Button(name='save now',button_type = 'primary',
                              width=100)
        save_button.on_click(save_measurement)
        callback_toggle = Toggle(name='Refresh',
                                 value=False, button_type='primary',
                                 width=100)
        callback_toggle.param.watch(toggle_periodic_callback, 'value')
        
        hv.extension('bokeh')
        pipe = Pipe(data=[])
        image_dmap = hv.DynamicMap(hv.Image, streams=[pipe])
        image_dmap.opts(xlim=(-2, 2), ylim=(-2, 2), cmap = 'Magma')
        
        video_mode_callback = PeriodicCallback(data_grabber, refresh_period)
        video_mode_server = Row(image_dmap,
                                Column(close_button,
                                       callback_toggle,
                                       save_button)).show(port = port)

#%%

class UHFLIRasterAcquisition_deprecated(Instrument):
    
    def __init__(self, name: str,
                 outer_setpoints, inner_setpoints,
                 uhfli, pixel_period,
                 demodulators, signals, trigger_input,
                 repetitions = 1, delay = 0.,
                 **kwargs):
        super().__init__(name, **kwargs)
        self.outer_setpoints = outer_setpoints
        self.inner_setpoints = inner_setpoints
        self.uhfli = uhfli
        self.pixel_period = pixel_period
        self.uhfli.daq.signals_clear()
        self.demodulators = [f'demod{i}' for i in demodulators]
        self.signals = signals
        self._signals_list = []
        self.trigger_input = trigger_input
        self.repetitions = repetitions
        self.delay = delay
        self.setup_parameters()
        
    def setup_parameters(self):
        # add parameters
        for demod in self.demodulators:
            for sig in self.signals:
                self.add_parameter(demod + '_' + sig,
                                   parameter_class = Mutable2DParameterWithSetpoints,
                                   outer_setpoints = self.outer_setpoints,
                                   inner_setpoints = self.inner_setpoints)
                getattr(self, demod + '_' + sig).path = self.uhfli.daq.signals_add(demod, sig)
                getattr(self, demod + '_' + sig).get = partial(self._get_demod_result,
                                                               getattr(self, demod + '_' + sig).path)
                self._signals_list.append(demod + '_' + sig)
        
    def setup_daq(self):          
        _ = self.uhfli.daq.trigger(self.demodulators[0], self.trigger_input)
        self.uhfli.daq.type('hardware')
        self.uhfli.daq.delay(self.delay)
        self.uhfli.daq.grid_repetitions(self.repetitions)
        # setup the grid
        self.uhfli.daq.grid_rows(1)
        self.uhfli.daq.grid_cols(self.outer_setpoints.shape[0] * self.inner_setpoints.shape[0])
        self.uhfli.daq.grid_mode('linear')
        self.uhfli.daq.duration(self.outer_setpoints.shape[0] * self.inner_setpoints.shape[0] * float(self.pixel_period()))
        
    def acquire(self):
        self.uhfli.daq.measure(verbose = False,
                               timeout = (1. + 2.* self.uhfli.daq.duration()))
        
    def _get_demod_result(self, path):
        try:
            return np.reshape(self.uhfli.daq.results[path].value,
                              self.outer_setpoints.shape + self.inner_setpoints.shape)
        except KeyError:
            pass
    
    def measure(self, setup = False, acquire = True):
        if setup: self.setup_daq()
        if acquire: self.acquire()
        meas = Measurement()
        for signal in self._signals_list:
            meas.register_parameter(getattr(self, signal))
        X, Y = np.meshgrid(self.outer_setpoints.get(),
                           self.inner_setpoints.get(),
                           indexing = 'ij')
        results = [(self.outer_setpoints, X),
                   (self.inner_setpoints, Y)]
        for signal in self._signals_list:
            results.append((getattr(self, signal), getattr(self, signal).get()))
        with meas.run() as datasaver:
            datasaver.add_result(*results)
        
    def video_mode(self, signal = '', refresh_period = 100, port = 12345):

        
        @gen.coroutine
        def data_grabber():
            if self.uhfli.daq._daq_module._module._module.finished():
                tmp_result = self.uhfli.daq._daq_module._module._module.read(flat=True)
                if getattr(self,signal).path in tmp_result.keys():
                    self.uhfli.daq._daq_module._get_result_from_dict(tmp_result)
                pipe.send((self.outer_setpoints.get(),
                           self.inner_setpoints.get(),
                           getattr(self, signal).get().T))
                self.uhfli.daq._daq_module._module._module.execute()

            
        def close_server_click(event):
            video_mode_server.stop()
            
        def save_measurement(event):
            self.measure(setup = False, acquire = False)
            
        def toggle_periodic_callback(event):
            if event.new is True:
                self.uhfli.daq._daq_module._module._module.subscribe(getattr(self,signal).path)
                video_mode_callback.start()
            else:
                self.uhfli.daq._daq_module._module._module.unsubscribe('*')
                video_mode_callback.stop()
        
        close_button = Button(name='close server',button_type = 'primary',
                              width=100)
        close_button.on_click(close_server_click)
        save_button = Button(name='save now',button_type = 'primary',
                              width=100)
        save_button.on_click(save_measurement)
        callback_toggle = Toggle(name='Refresh',
                                 value=False, button_type='primary',
                                 width=100)
        callback_toggle.param.watch(toggle_periodic_callback, 'value')
        
        hv.extension('bokeh')
        pipe = Pipe(data=[])
        image_dmap = hv.DynamicMap(hv.Image, streams=[pipe])
        image_dmap.opts(xlim=(-2, 2), ylim=(-2, 2), cmap = 'Magma')
        
        video_mode_callback = PeriodicCallback(data_grabber, refresh_period)
        video_mode_server = Row(image_dmap,
                                Column(close_button,
                                       callback_toggle,



#%% uhfli raster

uhfli_raster = UHFLIRasterAcquisition('UHFraster',
                                      mdac_raster.outer_ramp, mdac_raster.inner_ramp, 
                                      uhfli, mdac_raster.pixel_period.get,
                                      [0], ['r'], 'trigin3')

#%%



