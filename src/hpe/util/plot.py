import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Time Series Plot
def plot_time_series(data_frame):
    num_channels = data_frame.shape[1]
    fig, axes = plt.subplots(num_channels, 1, figsize=(10, 4 * num_channels), sharex=True)
    
    for i in range(num_channels):
        axes[i].plot(data_frame.iloc[:, i])
        axes[i].set_title(f'Channel {i+1}')
        axes[i].set_ylabel('Signal Amplitude')
        if i == num_channels - 1:
            axes[i].set_xlabel('Time')
    
    plt.tight_layout()
    plt.show()

# Frequency Domain Analysis
def plot_fft(data_frame, sampling_rate):
    n = data_frame.shape[0]
    frequencies = np.fft.fftfreq(n, d=1/sampling_rate)
    
    plt.figure(figsize=(10, 6))
    for i in range(data_frame.shape[1]):
        fft_values = np.fft.fft(data_frame.iloc[:, i])
        plt.plot(frequencies[:n//2], np.abs(fft_values[:n//2]), label=f'Channel {i+1}')
    plt.title('Frequency Domain Analysis (FFT)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

# Spectrogram
def plot_spectrogram(data_frame, sampling_rate):
    plt.figure(figsize=(10, 6))
    for i in range(data_frame.shape[1]):
        plt.specgram(data_frame.iloc[:, i], Fs=sampling_rate, label=f'Channel {i+1}')
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar(label='Intensity')
    plt.legend()
    plt.show()

# Histogram
def plot_histogram(data_frame):
    plt.figure(figsize=(10, 6))
    for i in range(data_frame.shape[1]):
        plt.hist(data_frame.iloc[:, i], bins=50, alpha=0.5, label=f'Channel {i+1}')
    plt.title('Histogram of EMG Signal')
    plt.xlabel('Signal Values')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Scatterplot
def plot_scatter(data_frame_1, data_frame_2):
    plt.figure(figsize=(8, 6))
    for i in range(min(data_frame_1.shape[1], data_frame_2.shape[1])):
        plt.scatter(data_frame_1.iloc[:, i], data_frame_2.iloc[:, i], label=f'Channel {i+1}')
    plt.title('Scatterplot')
    plt.xlabel('DataFrame 1 Values')
    plt.ylabel('DataFrame 2 Values')
    plt.legend()
    plt.show()

# Boxplot
def plot_boxplot(data_frame):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data_frame)
    plt.title('Boxplot of EMG Features')
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.show()

# Correlation Heatmap
def plot_correlation_heatmap(data_frame):
    corr_matrix = data_frame.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()

# PCA Visualization
from sklearn.decomposition import PCA
def plot_pca(data_frame, labels):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_frame)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], hue=labels)
    plt.title('PCA Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Labels')
    plt.show()
