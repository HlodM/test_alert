import pandas as pd
import numpy as np
from typing import Tuple, Optional
import binance
from binance.client import HistoricalKlinesType
from binance.exceptions import BinanceAPIException
from tqdm import tqdm


TZ_DIFF = 3                                             # time zone diff (msk=3)
RESIST_WIDTH = 0.0008                                   # resist channel width (+-)
MINS_BEFORE = 160                                       # before alert time history window length
MINS_AFTER = 120                                        # after alert time history window length
MAX_AREA = 4                                            # area to find max (+-)
BARS_PER_LEVEL = 50                                     # count bars per confidence level
DOWNLOAD_PATH = "alert_from_algo1_071223.csv"           # path to download file
SAVE_PATH = "filtered.csv"                              # path to save result file


def convert_to_timestamp(time: str, tz: int = TZ_DIFF) -> int:
    """
    Convert stroke time format to timestamp
    :param time: time
    :param tz: time zone diff, by default = 3
    :return: timestamp
    """
    time_dt = pd.to_datetime(time).floor("min") - pd.Timedelta(tz, "h")
    return int(time_dt.timestamp()) * 1000


def prepare_alert_df(
        filename: str,
        bars_per_level: int = BARS_PER_LEVEL,
        save_path: str = SAVE_PATH,
        n_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Read file as pd.DataFrame, add resist_level, n_touches, conf_level, resist_length and max_ratio_p
    columns and save result df to save_path
    :param filename: filename to read
    :param bars_per_level: count bars per confidence level, by default = 50
    :param save_path: path to  save result pd.DataFrame
    :param n_rows: count rows to read, by default = None
    :return: result pd.DataFrame
    """
    df = pd.read_csv(filename, index_col=0, nrows=n_rows)
    df["alert_price"] = df["alert_price"].apply(lambda x: float(x.split(":")[1]))
    client = binance.Client()
    tqdm.pandas()
    df["resist_level"], df["n_touches"], df["conf_level"], df["resist_length"], df["max_ratio_p"] = zip(
        *df.progress_apply(lambda x: find_resistance_length(x, client, bars_per_level), axis=1)
    )
    df.to_csv(save_path)
    return df


def find_resistance_length(
        data: pd.Series,
        client: binance.client.Client,
        bars_per_level: int = BARS_PER_LEVEL,
        mins_before: int = MINS_BEFORE,
        mins_after: int = MINS_AFTER,
        tz_diff: int = TZ_DIFF,
) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[int], Optional[float]]:
    """
    Download historical data and return resist level, count of touches of resist level,
    confidence level as ratio of count of bars under resist level and bars_per_level,
    length of resist level and ratio of max price after alert in mins_after frame and alert price
    :param data: row from alert_df
    :param client: binance client to download historical data
    :param bars_per_level: count bars per confidence level
    :param mins_before: before alert time history window
    :param mins_after: after alert time history window
    :param tz_diff: time zone diff
    :return: resist_level, n_touches, conf_level, bars_count under resist_level, max_ratio in percentage
    """
    alert_ts = convert_to_timestamp(data["Time_(MSK)"], tz_diff)
    start_ts = alert_ts - mins_before * 60 * 1000
    end_ts = alert_ts + mins_after * 60 * 1000

    try:
        hist_data = client.get_historical_klines(
            symbol=data["symbol"],
            interval=client.KLINE_INTERVAL_1MINUTE,
            start_str=start_ts,
            end_str=end_ts,
            klines_type=HistoricalKlinesType.FUTURES,
        )
    except BinanceAPIException:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    high = np.array([float(row[2]) for row in hist_data])
    before_high = high[:mins_before]
    after_high = high[mins_before:]
    resist_level, i = find_first_max(before_high)
    max_ratio_p = round((max(after_high) / data["alert_price"] - 1) * 100, 3)
    if resist_level:
        bars_count, n_touches = get_bars_count(before_high, resist_level, mins_before)
        return resist_level, n_touches, int(bars_count // bars_per_level), bars_count, max_ratio_p
    return resist_level, 0, 0, 0, max_ratio_p


def is_max(data: np.ndarray, index: int, area: int = MAX_AREA) -> bool:
    """
    Check if point is resist level
    :param data: before alert time historical data
    :param index: index of point
    :param area: area under consideration
    :return: is point resist level or not
    """
    return data[index] == max(data[index-area: index+area+1])


def find_first_max(data: np.ndarray, mins_before: int = MINS_BEFORE, area: int = MAX_AREA) -> tuple:
    """
    Find first max that satisfied resist level conditions
    :param data: time historical data
    :param mins_before: before alert time historical window length
    :param area: area to find max
    :return: resist level and its index
    """
    for i in range(mins_before-area-1, area-1, -1):
        if is_max(data, i):
            return data[i], i
    return 0, 0


def get_bars_count(
        data: np.ndarray,
        resist_level: float,
        mins_before: int = MINS_BEFORE,
        resist_width: float = RESIST_WIDTH
) -> Tuple[int, int]:
    """
    Find length of resistance level
    :param data: time historical data
    :param resist_level: level of resistance
    :param mins_before: length of before alert time historical window
    :param resist_width: resist channel width
    :return: resist level length and count of touches
    """
    start_resist = end_resist = 0
    n_touches = 0
    touch = False
    for i in range(mins_before-2, 0, -1):
        if start_resist:
            if (data[i] / resist_level >= 1 + resist_width) or ((end_resist or start_resist) - i > 148):
                break
            elif (1 - resist_width < data[i] / resist_level < 1 + resist_width) and data[i] > data[i+1]:
                touch = True
                n_touches += 1
            elif data[i] < data[i+1] and touch:
                end_resist = i + 1
                touch = False
        elif (1 - resist_width < data[i] / resist_level < 1 + resist_width) and data[i] > data[i+1]:
            start_resist = i
            n_touches += 1

    if end_resist:
        return start_resist - end_resist + 1, n_touches
    return int(bool(start_resist)), n_touches


if __name__ == "__main__":
    df = prepare_alert_df(DOWNLOAD_PATH)
