from threading import Thread
from .viz import Viz

class EmgVisualizer(Thread):
    def __init__(self, emg_data):
        super().__init__()

        # Visualize data stream in main thread:
        secs = 10             # Time window of plots (in seconds)
        ylim = (-1000, 1000)  # y-limits of plots
        ica = False           # Perform and visualize ICA alongside raw data
        update_interval = 10  # Update plots every X ms
        max_points = 250      # Maximum number of data points to visualize per channel (render speed vs. resolution)

        self.emg_data = emg_data
        self.emg_data.start()
        self.viz = Viz(emg_data, window_secs=secs, plot_exg=True, plot_imu=False, plot_ica=ica,
                update_interval_ms=update_interval, ylim_exg=ylim, max_points=250)
        
    def run(self):
        self.viz.start()

    def pause(self):
        pass

    def stop(self):
        self.viz.stop()
        self.emg_data.stop()
        self.emg_data.join()
        self.viz.join()