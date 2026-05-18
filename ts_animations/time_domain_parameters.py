"""Auto-split from legacy monolithic script."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def time_domain_parameters() -> None:
    class FourierVisualizer:
        def __init__(self):
            # Time domain parameters
            self.t = np.linspace(0, 10, 1000)
            self.dt = self.t[1] - self.t[0]
            # Generate composite signal
            self.f1, self.f2 = 1.0, 2.5  # frequencies
            self.signal = (
                np.sin(2 * np.pi * self.f1 * self.t)
                + 0.5 * np.sin(2 * np.pi * self.f2 * self.t)
                + 0.2 * np.random.randn(len(self.t))
            )
            # Compute FFT
            self.fft_vals = np.fft.fft(self.signal)
            self.freqs = np.fft.fftfreq(len(self.t), self.dt)
            # Store reconstructed signals
            self.reconstruct_signals()

        def reconstruct_signals(self):
            """Reconstruct signals with different numbers of frequencies"""
            self.reconstructions = []
            sorted_indices = np.argsort(np.abs(self.fft_vals))[::-1]
            for n_freqs in range(1, 51):
                fft_filtered = np.zeros_like(self.fft_vals, dtype=complex)
                fft_filtered[sorted_indices[:n_freqs]] = self.fft_vals[
                    sorted_indices[:n_freqs]
                ]
                fft_filtered[sorted_indices[-n_freqs + 1 :]] = self.fft_vals[
                    sorted_indices[-n_freqs + 1 :]
                ]
                reconstructed = np.fft.ifft(fft_filtered).real
                self.reconstructions.append(reconstructed)

        def create_animation(self):
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(3, 2)
            # Create subplots
            ax1 = fig.add_subplot(gs[0, :])  # Original signal
            ax2 = fig.add_subplot(gs[1, 0])  # Frequency domain
            ax3 = fig.add_subplot(gs[1, 1])  # Phase spectrum
            ax4 = fig.add_subplot(gs[2, :])  # Reconstruction

            def animate(frame):
                # Clear axes
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.clear()

                # Plot 1: Original Signal
                window = 200
                start_idx = frame % (len(self.t) - window)
                end_idx = start_idx + window
                ax1.plot(
                    self.t[start_idx:end_idx],
                    self.signal[start_idx:end_idx],
                    "b-",
                    label="Original Signal",
                )
                ax1.set_title("Time Domain Signal")
                ax1.set_xlabel("Time")
                ax1.set_ylabel("Amplitude")
                ax1.legend()
                ax1.grid(True)
                # Plot 2: Frequency Spectrum
                positive_freq_mask = self.freqs >= 0
                ax2.plot(
                    self.freqs[positive_freq_mask],
                    np.abs(self.fft_vals)[positive_freq_mask],
                    "r-",
                    label="Magnitude Spectrum",
                )
                ax2.set_title("Frequency Spectrum")
                ax2.set_xlabel("Frequency (Hz)")
                ax2.set_ylabel("Magnitude")
                ax2.grid(True)
                # Plot 3: Phase Spectrum
                ax3.plot(
                    self.freqs[positive_freq_mask],
                    np.angle(self.fft_vals)[positive_freq_mask],
                    "g-",
                    label="Phase Spectrum",
                )
                ax3.set_title("Phase Spectrum")
                ax3.set_xlabel("Frequency (Hz)")
                ax3.set_ylabel("Phase (radians)")
                ax3.grid(True)
                # Plot 4: Signal Reconstruction
                reconstruction_idx = min(frame // 2, len(self.reconstructions) - 1)
                ax4.plot(
                    self.t[start_idx:end_idx],
                    self.signal[start_idx:end_idx],
                    "b-",
                    alpha=0.5,
                    label="Original",
                )
                ax4.plot(
                    self.t[start_idx:end_idx],
                    self.reconstructions[reconstruction_idx][start_idx:end_idx],
                    "r-",
                    label=f"Reconstruction\n({reconstruction_idx + 1} frequencies)",
                )
                ax4.set_title("Signal Reconstruction")
                ax4.set_xlabel("Time")
                ax4.set_ylabel("Amplitude")
                ax4.legend()
                ax4.grid(True)
                plt.tight_layout()

            anim = FuncAnimation(
                fig,
                animate,
                frames=200,  # Adjust for smoothness
                interval=100,  # 100ms between frames
                repeat=True,
            )
            return anim

    # Create and save animation
    visualizer = FourierVisualizer()
    anim = visualizer.create_animation()
    anim.save("fourier_transform.gif", writer="pillow", fps=10)
    plt.close()
