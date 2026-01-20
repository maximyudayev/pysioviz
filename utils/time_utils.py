import time as _time
from time import perf_counter
from threading import Lock


class SingletonMeta(type):
  _instances = {}
  _lock: Lock = Lock()

  def __call__(cls, *args, **kwargs):
    with cls._lock:
      if cls not in cls._instances:
        instance = super().__call__(*args, **kwargs)
        cls._instances[cls] = instance
    return cls._instances[cls]


class SystemTime(metaclass=SingletonMeta):
  """Highly accurate system time Singleton that uses performance counters of the host device.

  Performance counters of the host devices are not Unix float numbers since epoch,
  but a float starting from 0 on the boot-up of the device. This Singleton uses less accurate system time
  with the `time.time()` call and combines with the performance counters, to get highly accurate time of the host device.

  This Singleton's methods should not be called directly. Use the global methods instead.

  Each subprocess of the application must be initialized via `init_time(...)`
  with a reference time obtained in the parent process via `get_ref_time()`.
  """

  def __init__(self):
    rough_time = _time.time()
    counter = perf_counter()
    self._ref_time = rough_time - counter

  def time(self) -> float:
    return self._ref_time + perf_counter()

  def _get_ref_time(self) -> float:
    return self._ref_time

  def _set_ref_time(self, ref_time: float) -> None:
    self._ref_time = ref_time


def init_time(ref_time: float) -> None:
  """Initialize the current process's `SystemTime` Singleton with a common reference time.

  Args:
      ref_time (float): Time obtained via `get_ref_time()` in the parent process to use as a reference for performance counters in the current process.
  """
  SystemTime()._set_ref_time(ref_time)


def get_ref_time() -> float:
  """Gets the current process's `SystemTime` Singleton reference time."""
  return SystemTime()._get_ref_time()


def get_time() -> float:
  """Gets the highly accurate current system time."""
  return SystemTime().time()
