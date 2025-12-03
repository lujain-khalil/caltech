import numpy as np
from . import config as Config

class NetworkSimulator:
    def __init__(self, avg_bw_mbps=Config.DEFAULT_BW, std_bw=3.0, avg_rtt_ms=Config.DEFAULT_RTT, std_rtt=10):
        """
        Simulates network conditions.
        avg_bw_mbps: Average Bandwidth in Megabits per second
        avg_rtt_ms: Average Round Trip Time in milliseconds
        """
        self.avg_bw = avg_bw_mbps
        self.std_bw = std_bw
        self.avg_rtt = avg_rtt_ms
        self.std_rtt = std_rtt

    def sample_network_state(self):
        """
        Returns a sample (bandwidth_bps, rtt_sec)
        """
        # Sample Bandwidth (ensure it doesn't go below 0.1 Mbps)
        bw = np.random.normal(self.avg_bw, self.std_bw)
        bw = max(0.1, bw) 
        
        # Sample RTT (ensure it doesn't go below 1ms)
        rtt = np.random.normal(self.avg_rtt, self.std_rtt)
        rtt = max(1.0, rtt)
        
        # Convert to base units: bits per second and seconds
        bw_bps = bw * 1_000_000
        rtt_sec = rtt / 1000.0
        
        return bw_bps, rtt_sec

    def estimate_transmission_time(self, data_size_bytes, bw_bps, rtt_sec):
        """
        Calculates T_comm based on the paper's formula [cite: 112]
        T_comm = B_act / (bw/8) + alpha + beta * RTT
        """
        # Fixed overheads (alpha) and queuing delays (beta) 
        # assumed small constants for this simulation
        alpha = 0.005  # 5ms handshake overhead 
        beta = 1.0     
        
        transmission_delay = data_size_bytes / (bw_bps / 8.0)
        t_comm = transmission_delay + alpha + (beta * rtt_sec)
        
        return t_comm