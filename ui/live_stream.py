import threading
import time
from typing import Optional, Dict, Any
from collections import deque
import numpy as np
try:
    from scipy.signal import find_peaks, butter, filtfilt
except Exception:
    find_peaks = None
    butter = None
    filtfilt = None

try:
    from pylsl import StreamInlet, resolve_byprop, resolve_streams
except ImportError:
    # Allow import even if pylsl not installed; raise on start
    StreamInlet = None
    resolve_byprop = None
    resolve_streams = None

try:
    import neurokit2 as nk
except ImportError:
    nk = None

class LiveLSLReader:
    def __init__(self, stream_name: str = 'OpenSignals', forced_mapping: Optional[Dict[str, int]] = None, timeout: float = 5.0) -> None:
        self.stream_name = stream_name
        self.forced_mapping = forced_mapping or {}
        self.timeout = timeout 
        self._inlet: Optional[StreamInlet] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.latest: Dict[str, Any] = {
            'timestamp': None,
            'sample': None,
            'channel_count': 0,
            'srate': None,
            'stream_info': None,
        }
        self.window_sec = 10.0
        self._ts_buf = deque(maxlen=2000)
        self._ch_bufs = [deque(maxlen=2000) for _ in range(4)]
        self._record = False
        self._record_buf = []

    def start(self) -> None:
        if StreamInlet is None:
            raise RuntimeError("pylsl is not installed. Please install 'pylsl'.")
        streams = resolve_byprop('name', self.stream_name, timeout=self.timeout)
        if not streams:
            raise RuntimeError(f"No LSL stream found with name '{self.stream_name}'")
        self._inlet = StreamInlet(streams[0])
        info = self._inlet.info()
        with self._lock:
            self.latest['channel_count'] = info.channel_count()
            self.latest['srate'] = info.nominal_srate()
            self.latest['stream_info'] = info
            try:
                labels = []
                desc = info.desc()
                chs = desc.child("channels") if desc else None
                for i in range(info.channel_count()):
                    c = chs.child("channel", i) if chs else None
                    label = c.child_value("label") if c else ""
                    labels.append(label if label else f"Ch{i+1}")
                if labels:
                    self.latest['channel_labels'] = labels
            except Exception:
                pass
            maxlen = int(max(100, self.latest['srate'] * self.window_sec))
            self._ts_buf = deque(maxlen=maxlen)
            self._ch_bufs = [deque(maxlen=maxlen) for _ in range(max(4, self.latest['channel_count']))]
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._inlet = None

    def _run(self) -> None:
        assert self._inlet is not None
        while not self._stop.is_set():
            try:
                sample, ts = self._inlet.pull_sample(timeout=0.1)
                if sample is None:
                    continue
                with self._lock:
                    self.latest['timestamp'] = ts
                    self.latest['sample'] = sample
                    self._ts_buf.append(ts)
                    for i in range(min(len(self._ch_bufs), len(sample))):
                        self._ch_bufs[i].append(sample[i])
                    if self._record:
                        self._record_buf.append((ts, list(sample)))
            except Exception:
                time.sleep(0.05)

    def start_recording(self) -> None:
        with self._lock:
            self._record = True
            self._record_buf = []

    def stop_recording(self):
        with self._lock:
            self._record = False
            return list(self._record_buf)

    def get_latest_metrics(self) -> Dict[str, Any]:
        with self._lock:
            ts = np.array(self._ts_buf, dtype=float)
            ch_bufs = [np.array(self._ch_bufs[i], dtype=float) for i in range(len(self._ch_bufs))]
            srate = float(self.latest.get('srate') or 50)

        metrics = {'hr_bpm': None, 'hrv_rmssd_ms': None, 'hrv_sdnn_ms': None, 'resp_rate_bpm': None, 'emg_rms': None, 'emg_median_freq': None}
        if len(ch_bufs) == 0 or any(buf.size < 50 for buf in ch_bufs[:3]):
            return metrics

        # ECG - Aligné avec process_data.py
        ecg_idx = self.forced_mapping.get('ecg', 1)
        ecg_raw = ch_bufs[ecg_idx]
        self.latest['ecg_channel_idx'] = ecg_idx

        hr_bpm = None
        rmssd = None
        sdnn = None
        if nk is not None:
            try:
                # Nettoyage ECG comme dans process_data.py
                ecg_cleaned = nk.ecg_clean(ecg_raw, sampling_rate=int(srate), method="neurokit")
                _, info = nk.ecg_process(ecg_cleaned, sampling_rate=int(srate))
                r_peaks = info.get("ECG_R_Peaks", np.array([]))
                
                if r_peaks is not None and len(r_peaks) >= 2:
                    hr = nk.signal_rate(r_peaks, sampling_rate=int(srate), desired_length=len(ecg_cleaned))
                    hr_mean = np.nanmean(hr)
                    if not np.isnan(hr_mean) and hr_mean > 0:
                        hr_bpm = float(hr_mean)
                    
                    # Calculer HRV (RMSSD et SDNN)
                    hrv = nk.hrv_time(info, sampling_rate=int(srate), show=False)
                    if hrv is not None:
                        rmssd_val = float(hrv.get("HRV_RMSSD", np.nan))
                        sdnn_val = float(hrv.get("HRV_SDNN", np.nan))
                        if not np.isnan(rmssd_val):
                            rmssd = rmssd_val
                        if not np.isnan(sdnn_val):
                            sdnn = sdnn_val
            except Exception:
                pass

        metrics['hr_bpm'] = hr_bpm
        metrics['hrv_rmssd_ms'] = rmssd
        metrics['hrv_sdnn_ms'] = sdnn

        # Respiration - Aligné avec process_data.py
        resp_rate = None
        resp_idx = self.forced_mapping.get('resp', 3)
        if len(ch_bufs) > resp_idx and ch_bufs[resp_idx].size > 50:
            try:
                resp = ch_bufs[resp_idx]
                # Filtrage basse fréquence comme dans process_data.py
                if nk:
                    resp_cleaned = nk.signal_filter(resp, sampling_rate=int(srate), lowcut=None, highcut=2, method="butterworth", order=4)
                    # Utiliser rsp_process comme dans process_data.py
                    rsp_signals, _ = nk.rsp_process(resp_cleaned, sampling_rate=int(srate))
                    rate_series = rsp_signals.get("RSP_Rate")
                    if rate_series is not None:
                        resp_rate = float(np.nanmean(rate_series))
                else:
                    # Fallback avec find_peaks si neurokit2 n'est pas disponible
                    if find_peaks:
                        min_dist = int(max(1, srate * 1.0))
                        rp, _ = find_peaks(resp, distance=min_dist, prominence=np.std(resp) * 0.3)
                        if rp.size >= 2:
                            resp_rate = (rp.size / (len(resp) / srate)) * 60.0
            except Exception:
                pass
        metrics['resp_rate_bpm'] = resp_rate

        # EMG - Aligné avec process_data.py
        emg_idx = self.forced_mapping.get('emg', 2)
        emg_rms = None
        emg_mf = None
        if len(ch_bufs) > emg_idx and ch_bufs[emg_idx].size > 10:
            try:
                emg = ch_bufs[emg_idx]
                if nk:
                    emg_cleaned = nk.emg_clean(emg, sampling_rate=int(srate))
                else:
                    emg_cleaned = emg
                
                # Calculer RMS sur valeur absolue comme dans process_data.py
                emg_rect = np.abs(emg_cleaned)
                emg_rms = float(np.sqrt(np.mean(np.square(emg_rect)))) if emg_rect.size else None
                
                # Calculer fréquence médiane comme dans process_data.py
                if nk:
                    nyquist = srate / 2.0
                    low, high = 20.0, min(450.0, nyquist - 1.0)
                    if high > low:
                        try:
                            power = nk.signal_power(emg_cleaned, sampling_rate=int(srate), frequency_band=[low, high])
                            emg_mf = float(power.get("Median Frequency", np.nan))
                            if np.isnan(emg_mf):
                                emg_mf = None
                        except Exception:
                            pass
            except Exception:
                pass
        
        metrics['emg_rms'] = emg_rms
        metrics['emg_median_freq'] = emg_mf

        return metrics

    def get_latest(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.latest)
