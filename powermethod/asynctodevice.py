import threading
import queue
import numpy as np
from numbapro import cuda


class AsyncToDevice(object):
    def __init__(self, devnum=0):
        self.devnum = devnum
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()

    def worker(self):
        cuda.select_device(self.devnum)
        while True:
            with cuda.gpus[self.devnum]:
                host, device, stream = self.queue.get()
                device.copy_to_device(host, stream=stream)
                self.queue.task_done()

    def to_device(self, host, stream):
        with cuda.gpus[self.devnum]:
            device = cuda.device_array_like(host, stream=stream)
            self.queue.put((host, device, stream))
            return device

    def join(self):
        self.queue.join()


class AsyncToHost(object):
    def __init__(self, devnum=0):
        self.devnum = devnum
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()

    def worker(self):
        cuda.select_device(self.devnum)
        while True:
            with cuda.gpus[self.devnum]:
                host, device, stream = self.queue.get()
                device.copy_to_host(host, stream=stream)
                self.queue.task_done()

    def to_host(self, device, stream):
        with cuda.gpus[self.devnum]:
            host = np.empty(shape=device.shape, dtype=device.dtype)
            self.queue.put((host, device, stream))
            return host

    def join(self):
        self.queue.join()


def main():
    cuda.select_device(0)
    a2d = AsyncToDevice()
    a2h = AsyncToHost()

    dlist = []
    stream = cuda.stream()
    for i in range(10):
        A = np.arange(1000)
        dA = a2d.to_device(A, stream=stream)
        dlist.append(dA)
    a2d.join()

    out = []
    for dA in dlist:
        A = a2h.to_host(dA, stream=stream)
        out.append(A)
    a2h.join()


if __name__ == '__main__':
    main()
