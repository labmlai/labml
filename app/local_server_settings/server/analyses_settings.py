from .analyses.experiments.metrics import MetricsAnalysis
from .analyses.experiments import comparison
from .analyses.experiments import distributed_metrics

from .analyses.computers.cpu import CPUAnalysis
from .analyses.computers.gpu import GPUAnalysis
from .analyses.computers.memory import MemoryAnalysis
from .analyses.computers.network import NetworkAnalysis
from .analyses.computers.disk import DiskAnalysis
from .analyses.computers.process import ProcessAnalysis
from .analyses.computers.battery import BatteryAnalysis

experiment_analyses = [MetricsAnalysis]

computer_analyses = [CPUAnalysis,
                     GPUAnalysis,
                     MemoryAnalysis,
                     NetworkAnalysis,
                     DiskAnalysis,
                     BatteryAnalysis,
                     ProcessAnalysis]
