from typing import List, Tuple, Union
from statsmodels.tsa.filters.hp_filter import hpfilter
import numpy as np
import pywt
from astropy.stats import biweight_midvariance
from astropy.timeseries import LombScargle
from joblib import Parallel, delayed
from pmdarima.arima import auto_arima
from scipy import integrate
from scipy.special import binom
from scipy.fftpack import fft
from scipy.signal import fftconvolve, find_peaks
from pandas.tseries.frequencies import to_offset
import pandas as pd


def _extract_trend(x, lamb=129600):
    """use hp-filter to detrend
    Args:
        x (_type_): original values
        lamb (_type_): regulization parameter, A value of 1600 issuggested for quarterly data.
        Ravn and Uhlig suggest using a value of 6.25 (1600/4**4) for annual data
        and 129600 (1600*3**4) for monthly data.

    Returns:
        _type_: values after detrend
    """
    _, trend = hpfilter(x, lamb)
    x_detrend = x - trend
    return x_detrend


def _detect_freq(freq):
    freq = to_offset(freq).name
    if "AS-" in freq or "A-" in freq or freq == "AS" or freq == "A":
        return 6.25
    elif "QS-" in freq or "Q-" in freq or freq == "QS" or freq == "Q":
        return 1600
    elif "MS-" in freq or "M-" in freq or freq == "MS" or freq == "M":
        return 129600
    else:
        return 10e6


class DensityClustering:
    def __init__(self, time_idx: np.ndarray, powers: np.ndarray, periods: Tuple[int, float]) -> None:
        self.periods = periods
        self.powers = powers
        self.bin_margin = [0] + [len(time_idx) / i for i in range(len(time_idx) // 2, 0, -1)]

    def compute(self) -> List[Tuple[float, float]]:
        # 按hints从小到大拍
        p_sort = sorted(self.periods, key=lambda x: x[1], reverse=False)
        Cluster_result = []
        cluster = []
        hint_p = None
        bg_point = 0
        for i in range(len(p_sort)):
            epsilon, bg_point = self._get_nowhint_binvalue(hint_p, bg_point)
            if p_sort[i][1] <= epsilon:
                cluster.append(p_sort[i])
            else:
                Cluster_result.append(cluster)
                cluster = [p_sort[i]]
            hint_p = p_sort[i][1]
        return self._get_cluster_cetroids(Cluster_result[1:])  # 去除第一个[nan,nan]

    def _get_nowhint_binvalue(self, hint_p: float, bg_point: int) -> Tuple[float, int]:
        # 第一个点初始化
        if hint_p is None:
            return 0, 0
        else:
            for i in range(bg_point, len(self.bin_margin) - 1):
                # 左闭右开
                if self.bin_margin[i] <= hint_p and self.bin_margin[i + 1] > hint_p:
                    # 返回下一个bin的大小+1, 下次开始搜索的位置
                    return self.bin_margin[i + 1] + 1, i

    def _get_cluster_cetroids(self, clusters: Tuple[int, float]) -> List[Tuple[float, float]]:
        centroids = []
        for cluster in clusters:
            # 获得每个cluster的period均值及power均值
            centroids.append(
                (
                    np.mean([i[1] for i in cluster]),
                    np.max([self.powers[i[0]] for i in cluster]),
                )
            )
        return centroids  # [(hint_p_centroid, power_centroid)]


class Autoperiod:
    def __init__(
        self,
        times: List[int],
        values: Union[List[float], List[int]],
        plotter=None,
        mc_iterations: int = 100,
        confidence_level: float = 0.99,
        random_state: int = 666,
        win_size: int = 5,
        thres1: float = 0.1,
        thres2: float = 0.1,
        auto_time_differencing: bool = False,
        n_jobs: int = 1,
    ):
        """Automatically detection of periodicity of time series

        Args:
            times (List[int]): index indicating the order of values
            values (Union[List[float], List[int]]): time-series values
            plotter ([type], optional): whether to plot. Defaults to None.
            mc_iterations (int, optional): number of random series obtained by permutation. Defaults to 100.
            confidence_level (float, optional): the percentile of power used as power_threshold in. Defaults to 0.99.
            random_state (int, optional): random seed used to generate list of random seeds in permutation. Defaults to 666.
            win_size (int, optional): one-side window length to check acf peak. Defaults to 5.
            thres1 (float, optional): threshold of difference between potential_peak and [potential_peak+/-win_size]. Defaults to 0.01.
            thres2 (float, optional):
                threshold of difference between slope of [potential_peak,potential_peak-win_size] and \
                slope of [potential_peak+win_size,potential_peak], by default 0.01
            auto_time_differencing (bool, optional): whether to perform k-th difference automately evaluated by pmdarima. Defaults to False.
            n_jobs (int, optional): number of CPU resources to be used. Defaults to 1.
        """

        self.random_state = random_state
        self._mc_iterations = mc_iterations
        self.confidence_level = confidence_level
        self.peakdic = {}
        self.n_jobs = n_jobs

        # time diferencing to make the input series stationary
        if auto_time_differencing:  # using `pmdarima` to determine the time differencing order
            order_ = self.__estimate_time_diferencing_order(values)
            self.values = np.diff(values, order_)
            self.time_idx = times[order_:] - times[order_] if times[order_] != 0 else times[order_:]
        else:
            self.time_idx = times - times[0] if times[0] != 0 else times
            self.values = values
        self.time_span = self.time_idx[-1] - self.time_idx[0]
        # assert self.time_span % (len(self.time_idx) - 1) == 0, "Check your time_idx"
        self.time_interval = self.time_span / (len(self.time_idx) - 1)
        self.plotter = plotter
        self.acf = self.autocorrelation()  # normalize acf
        self.acf /= np.max(self.acf)
        self.acf *= 100
        self.win_size = win_size  # 寻找acf peak时的左右窗口长度
        self.thres1 = thres1  # acf差值的最小阈值
        self.thres2 = thres2  # 斜率差值的最小阈值

        freqs, self.powers = self._compute_power_spectrum(self.values)
        self.periods = 1 / freqs
        self._power_threshold = self._compute_power_threshold(self.values)
        self._period_hints = self._get_period_hints()
        self._potential_period_list = self._get_potential_acfpeaks()
        self._get_actual_periods()
        # print("final=", self.period_final)
        self.plot()

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, random_seed):
        np.random.seed(random_seed)
        self._random_state = random_seed

    @property
    def period_list(self):
        # 按ACF值倒序排列返回
        return sorted(self.period_final, key=lambda x: self.acf[x], reverse=True)

    @property
    def period(self):
        return [self._period]

    def period_area(self):
        period_region = self._sinwave > (np.max(self._sinwave) / 2)

        on_period_area = integrate.trapz(
            self.values[: self._period * 10][period_region],
            self.time_idx[: self._period * 10][period_region],
        )
        off_period_area = integrate.trapz(
            self.values[: self._period * 10][~period_region],
            self.time_idx[: self._period * 10][~period_region],
        )
        return on_period_area, off_period_area

    @property
    def phase_shift_guess(self):
        return self.time_idx[np.argmax(self.values)]

    @staticmethod
    def __estimate_time_diferencing_order(values: np.ndarray) -> int:
        return auto_arima(
            values,
            start_p=1,
            start_q=1,
            max_p=1,
            max_q=1,
            max_d=4,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        ).order[1]

    def _get_period_hints(self) -> List[Tuple[int, float]]:
        periods__ = []
        for i, period in enumerate(self.periods):
            # periods__=[(20,6.00031),(16,10.12421)...]
            periods__.append((i, period))

        period_hints = DensityClustering(self.time_idx, self.powers, periods__).compute()
        # print(self._power_threshold)
        if self.plotter:
            self.plotter.plot_periodogram(
                self.periods,
                self.powers,
                period_hints,
                self._power_threshold[-1],
                self.time_span / 2,
            )
        return period_hints

    def _compute_power_spectrum(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        freqs, powers = LombScargle(self.time_idx, series).autopower(
            minimum_frequency=1 / self.time_span,
            maximum_frequency=1 / (self.time_interval * 2),
            normalization="psd",
        )
        _power_norm_factor = 1 / (2 * np.var(series - np.mean(series)))
        # double the power as the astropy lomb-scargle implementation halves it during the psd normalization
        return freqs, powers * 2 * _power_norm_factor

    def _compute_power_threshold(self, series: np.ndarray) -> float:
        series_ = np.copy(series)

        def shuf_func(series):
            np.random.shuffle(series)
            _, powers = self._compute_power_spectrum(series)
            return np.max(powers)

        extreme_values = Parallel(n_jobs=self.n_jobs)(
            delayed(shuf_func)(series_) for _ in range(self._mc_iterations)
        )
        return [np.quantile(extreme_values, self.confidence_level, axis=0)]

    def _find_tarnow(self, search_st: int, pre_idx: int) -> int:
        search_ed = (
            pre_idx + self.win_size + 1
            if pre_idx + self.win_size + 1 < len(self.time_idx)
            else len(self.time_idx)
        )
        # if search_ed <= search_st:
        #     return pre_idx
        find_idx = pre_idx + np.argmax(self.acf[pre_idx:search_ed])
        if pre_idx == find_idx or search_ed == len(self.time_idx):
            return find_idx
        else:
            return self._find_tarnow(search_ed, find_idx)

    def _find_tarpre(self, search_st: int, pre_idx: int) -> int:
        if search_st < 0:
            return pre_idx
        find_idx = np.argmax(self.acf[search_st : pre_idx + 1])
        if find_idx == self.win_size:
            return pre_idx
        else:
            return -1 * pre_idx

    def _get_potential_acfpeaks(self) -> List[int]:
        i = 0
        potential_period_list = []
        while i < len(self.time_idx):
            # 长度为self.win_size*2的区间内找到acf最大点
            win_ed = (
                i + self.win_size * 2 + 1
                if i + self.win_size * 2 + 1 < len(self.time_idx)
                else len(self.time_idx)
            )
            tarpre_idx = i + np.argmax(self.acf[i:win_ed])
            if tarpre_idx >= i + self.win_size:
                # 判断tarpre的acf是否是前后self.win_size中最大的，若不是则继续往后寻找，找出第一个符合的点
                tarnow_idx = self._find_tarnow(i + self.win_size * 2 + 1, tarpre_idx)
            else:
                tarnow_idx = self._find_tarpre(tarpre_idx - self.win_size, tarpre_idx)
            # 下次循环从符合点+self.win_size+1开始，因为[符合点,符合点+self.win_size]间不可能有局部最大值点
            i = abs(tarnow_idx) + self.win_size + 1
            if tarnow_idx <= 0 or (tarnow_idx in [0, len(self.time_idx) - 1]):
                continue
            # print(tarnow_idx)
            # 获取其acf
            tarnow_acf = self.acf[tarnow_idx]
            # 窗口起终点
            bg_idx = tarnow_idx - self.win_size if tarnow_idx - self.win_size > 0 else 0
            ed_idx = tarnow_idx + self.win_size if tarnow_idx + self.win_size < len(self.time_idx) else -1
            # acf差值是否大于self.thres1
            if tarnow_acf - self.acf[ed_idx] > self.thres1 and tarnow_acf - self.acf[bg_idx] > self.thres1:
                # print('tarnow_idx=',tarnow_idx)
                slope1 = (tarnow_acf - self.acf[bg_idx]) / (tarnow_idx - bg_idx)
                slope2 = (self.acf[ed_idx] - tarnow_acf) / (ed_idx - tarnow_idx)
                # 是否满足斜率条件及self.thres2
                if slope1 > 0 and slope2 < 0 and slope1 - slope2 > self.thres2:
                    potential_period_list.append(self.time_idx[tarnow_idx])

        potential_period_list = [int(item) for item in potential_period_list]
        # 防止potential_period_list为空报错
        try:
            acf_thres = sorted([self.acf[item] for item in potential_period_list])[
                int(0.95 * len(potential_period_list))
            ]
            potential_period_list_final = [idx for idx in potential_period_list if self.acf[idx] >= acf_thres]
        except:
            # print("no peak available")
            return []
        return potential_period_list_final

    def validate_hint3(self, p: Tuple[float, float]) -> int:
        # 获取centroid点及power
        power_hint = p[1]
        hint_p = p[0]
        try:
            # 寻找距离最近的peak_idx
            peak_idx = np.argmin([abs(hint_p - item) for item in self._potential_period_list])
            # 记录匹配到该peak点的hint_points点中最大的power
            if (self._potential_period_list[peak_idx] not in self.peakdic) or (
                power_hint > self.peakdic[self._potential_period_list[peak_idx]]
            ):  # self.peakdic={peak_period:max(power)}
                self.peakdic[self._potential_period_list[peak_idx]] = power_hint
            return self._potential_period_list[peak_idx]  # return peak_period
        except:
            return None

    def autocorrelation(self) -> List[float]:
        acf = fftconvolve(self.values, self.values[::-1], mode="full")
        return acf[acf.size // 2 :]

    def plot(self):
        self._period = None
        self._sinwave = None
        if None not in self.period_final and len(self.period_final) > 0:
            self._period = self.period_final[0]  # 最小周期用于画图
            phase_shift = self.time_idx[np.argmax(self.values[: self._period * 10])]
            amplitude = np.max(self.values) / 2
            # 画前10个周期
            self._sinwave = (
                np.cos(2 * np.pi / self._period * (self.time_idx[: self._period * 10] - phase_shift))
                * amplitude
                + amplitude
            )
        if self.plotter:
            if self._period:
                # 有周期时画10个周期长度
                self.plotter.plot_timeseries(
                    self.time_idx[: self._period * 10], self.values[: self._period * 10]
                )
                self.plotter.plot_acf(self.time_idx, self.acf)
                self.plotter.plot_sinwave(self.time_idx[: self._period * 10], self._sinwave)
                self.plotter.plot_area_ratio(*self.period_area())
            else:
                # 没有周期时按原全序列画图
                self.plotter.plot_timeseries(self.time_idx, self.values)
                self.plotter.plot_acf(self.time_idx, self.acf)

    def _get_actual_periods(self):
        period2 = []
        # 找到离候选点p_hints最近的peak
        """
        for i, p in self._period_hints:
            period2.append(self.validate_hint2(i, p))
        """
        for p in self._period_hints:
            period2.append(self.validate_hint3(p))
        self.period_final = sorted(list(set(period2)))

        if None in self.period_final:
            print("no nearrest_peak available")
        else:
            # 去除周期性的整数倍,仅当整数倍周期的acf/power较小时
            i = 1
            while i < len(self.period_final):
                if 0 in [
                    self.period_final[i] % item
                    for item in self.period_final[:i]
                    if self.acf[self.period_final[i]] < self.acf[item]  # acf
                    # power
                    or self.peakdic[self.period_final[i]] < self.powers[item]
                ]:
                    self.period_final.pop(i)
                else:
                    i += 1


class RobustPeriod:
    def __init__(
        self,
        times: List[int],
        values: Union[List[float], List[int]],
        plotter=None,
        num_wavelet: int = 8,
        c: float = 2,  # Huber function hyperparameter
        wavelet_method: str = "db10",
    ):
        assert wavelet_method.startswith("db"), "wavelet method must be Daubechies family"

        self.time_idx = times - times[0] if times[0] != 0 else times
        self.wavelet_method = wavelet_method
        self.plotter = plotter
        self.num_wavelet = num_wavelet
        # decide lamb according to freq
        self.huber_c = c
        # remove extreme outliers, TODO: can be polished
        processed_values = self._remove_outliers(values, self.huber_c)
        # Daubechies MODWT
        self._wave = self.modwt(processed_values, self.wavelet_method, level=self.num_wavelet)
        # Robust Unbiased Wavelet Variance to sort levels
        self._bivar = np.array([biweight_midvariance(w) for w in self._wave])
        order = [list(self._bivar).index(item) for item in sorted(self._bivar, reverse=True)]
        # padding zeroes to its 2-times length and calculate periodogram
        waves = np.hstack([self._wave[order], np.zeros_like(self._wave)])
        periodograms = []
        self._p_vals = []
        for i, x in enumerate(waves):
            print(f"Calculating periodogram for level {order[i]}")
            perio = self._fft_reg(x)
            p_val = self._fisher_g_test(perio)
            periodograms.append(perio)
            self._p_vals.append(p_val)
        self._periodograms = np.array(periodograms)

        periods = []
        for i, p in enumerate(self._periodograms):
            if self._p_vals[i] <= 0.05:
                final_period = self.get_ACF_period(p)
                periods.append(final_period)
        periods = np.array(periods)
        # to keep the order, use pd.unique instead of np.unique
        self._final_periods = pd.DataFrame(periods[periods > 0])[0].unique()

    @property
    def period(self):
        return [int(p) for p in self._final_periods]

    @property
    def acf(self):
        return np.array([self._huber_acf(p) for p in self._periodograms])

    @property
    def p_val(self):
        return self._p_vals

    @property
    def periodograms(self):
        return self._periodograms

    @property
    def bivar(self):
        return self._bivar

    @property
    def wavelets(self):
        return self._wave

    def _remove_outliers(self, x, c):
        mu = np.median(x)
        s = np.mean(np.abs(x - np.median(x)))
        return np.sign((x - mu) / s) * np.minimum(np.abs((x - mu) / s), c)

    def _circular_convolve_d(self, h_t, v_j_1, j):
        """jth level decomposition

        Args:
            h_t : h / sqrt(2)
            v_j_1 : the (j-1)th scale coefficients
            j : level

        Returns:
            _type_: _description_
        """
        N = len(v_j_1)
        L = len(h_t)
        # Matching the paper
        L_j = min(N, (2 ** 4 - 1) * (L - 1))
        w_j = np.zeros(N)
        l = np.arange(L)
        for t in range(N):
            index = np.mod(t - 2 ** (j - 1) * l, N)
            v_p = np.array([v_j_1[ind] for ind in index])
            # Keeping up to L_j items
            w_j[t] = (np.array(h_t)[:L_j] * v_p[:L_j]).sum()
        return w_j

    def _fft_reg(self, series):
        ffts = fft(series)
        perior = np.abs(ffts) ** 2
        return np.array(perior)

    def _fisher_g_test(self, period):
        g = max(period) / np.sum(period)
        pval = self._p_val_g_stat(g, len(period))
        return pval

    def _p_val_g_stat(self, g0, N):
        g0 = 1e-8 if g0 == 0 else g0
        terms = np.arange(1, int(np.floor(1 / g0)) + 1, dtype="int32")
        # Robust Period Equation
        def event_term(k, N=N, g0=g0):
            return (-1) ** (k - 1) * binom(N, k) * (1 - k * g0) ** (N - 1)

        vect_event_term = np.vectorize(event_term)
        pval = min(sum(vect_event_term(terms, N, g0)), 1)
        return pval

    def _huber_acf(self, periodogram):
        N_prime = len(periodogram)
        N = N_prime // 2
        K = np.arange(N)

        cond_1 = periodogram[range(N)]  # k = 0,1,...N-1
        cond_2 = (periodogram[2 * K] - periodogram[2 * K + 1]).sum() ** 2 / N_prime  # k=N
        cond_3 = [periodogram[N_prime - k] for k in range(N + 1, N_prime)]  # k=N+1,...,2N-1

        P_bar = np.hstack([cond_1, cond_2, cond_3])
        P = np.real(np.fft.ifft(P_bar))
        # (N-t)*P0
        denom = (N - np.arange(0, N)) * P[0]
        # Pt / ((N-t) * P0)
        res = P[:N] / denom

        return res

    def get_ACF_period(self, periodogram):
        N = len(periodogram)
        k = np.argmax(periodogram)
        res = self._huber_acf(periodogram)
        # TODO: why trim and scale
        res_trim = res[: int(len(res) * 0.8)]  # The paper didn't use entire ACF
        # min-max scale
        res_scaled = 2 * ((res_trim - res_trim.min()) / (res_trim.max() - res_trim.min())) - 1
        # the predefined height threshold is 0.5
        peaks, _ = find_peaks(res_scaled, height=0.5)
        distances = np.diff(peaks)
        # calculate the median distance of peaks who satisfy threshold
        acf_period = np.median(distances) if len(distances) > 0 else 0
        # range
        Rk = (0.5 * ((N / (k + 1)) + (N / k)) - 1, 0.5 * ((N / k) + (N / (k - 1))) + 1)
        final_period = acf_period if (Rk[1] >= acf_period >= Rk[0]) else 0

        return final_period

    def modwt(self, x, filters="db10", level=3):
        wavelet = pywt.Wavelet(filters)
        h = wavelet.dec_hi  # wavelet filter
        g = wavelet.dec_lo  # scaling filter
        # TODO: why /np.sqrt(2)
        h_t = np.array(h) / np.sqrt(2)
        g_t = np.array(g) / np.sqrt(2)
        wavecoeff = []
        v_j_1 = x
        for j in range(level):
            w = self._circular_convolve_d(h_t, v_j_1, j + 1)
            v_j_1 = self._circular_convolve_d(g_t, v_j_1, j + 1)
            # if j > 0:
            wavecoeff.append(w)
        # wavecoeff.append(v_j_1)
        return np.vstack(wavecoeff)
