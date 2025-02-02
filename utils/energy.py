# reference: https://github.com/maufadel/EnergyMeter

import numpy as np
from pynvml import nvmlDeviceGetCount
from pynvml_utils import nvidia_smi

import time
import threading

class ThreadGpuSamplingPyNvml(threading.Thread):
    
    SAMPLING_PERIOD = 0.1
    
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name
        self.measuring = False
        self.exit = False
        self.power_draw_history = []
        self.activity_history = []
        self.nvsmi = nvidia_smi.getInstance()
        self.gpu_count = nvmlDeviceGetCount()

    def run(self):
        while True:
            if self.exit:
                break
            if self.measuring:
                nvml_outputs = self.nvsmi.DeviceQuery("power.draw,utilization.gpu").get("gpu")
                self.power_draw_history.append([output.get("power_readings").get("power_draw") for output in nvml_outputs])
                self.activity_history.append([output.get("utilization").get("gpu_util") for output in nvml_outputs])
            time.sleep(ThreadGpuSamplingPyNvml.SAMPLING_PERIOD)

class EnergyMeter:
    def __init__(self,include_idle=False):
        self.include_idle = include_idle
        self.thread_gpu = ThreadGpuSamplingPyNvml("GPU Sampling Thread")
        self.thread_gpu.start()
        self.start_time = None

    def start_session(self):
        self.start_time = time.time()
        self.thread_gpu.power_draw_history.clear()
        self.thread_gpu.activity_history.clear()
        self.thread_gpu.measuring = True

    def finish_session(self):
        self.thread_gpu.measuring = False
        return self.get_total_jules_gpu(time.time() - self.start_time), self.thread_gpu.activity_history, self.thread_gpu.power_draw_history
    
    def stop(self):
        self.thread_gpu.exit = True
    
    def get_total_jules_gpu(self, duration):
        if len(self.thread_gpu.activity_history) == 0:
            return np.zeros(self.thread_gpu.gpu_count)
        if self.include_idle:
            # The mean power draw is calculated over the entire duration, including idle periods.
            mean_p = np.mean(np.array(self.thread_gpu.power_draw_history), axis=0)
            te = mean_p * duration
        else:
            ah = np.array(self.thread_gpu.activity_history)
            if np.sum(ah) == 0:
                return 0
                
            pdh = np.array(self.thread_gpu.power_draw_history)
            if pdh.shape[0] > ah.shape[0]:
                pdh = pdh[:ah.shape[0], :]

            # Only power measurements during GPU utilization are aggregated.
            te = np.sum(pdh * (ah > 0), axis=0) * self.thread_gpu.SAMPLING_PERIOD
            
        return te
    
    def print_log(self):
        print(f'Started current logging at {self.start_time}...')
        print(f'Collected samples: {len(self.thread_gpu.power_draw_history)} for {self.thread_gpu.gpu_count} GPU(s)')
