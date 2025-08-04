import os
import tarfile
import io
import numpy as np
import json
from tqdm import tqdm
import zstandard as zstd
import databento as db
import pandas as pd
import polars as pl
import numpy.typing as npt
import uuid
from streaming import MDSWriter
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error, mean_absolute_error


DATA_DIR = "/Users/danielfisher/data/databento/"
IMAGE_DIR = "/Users/danielfisher/data/market_depths"
CURRENCY = "AUDUSD"

MICRO_PIP_SIZE = 0.00001
TRUE_PIP_SIZE = 0.0001
TICKS_PER_BIN = 100
INPUT_ROWS = 500 * TICKS_PER_BIN  # 50_000

LOOKFORWARD_OFFSET = 500
LOOKFORWARD_INPUT = 5000
LOOKFORWARD_ROWS = LOOKFORWARD_INPUT + LOOKFORWARD_OFFSET  # 5_000
LOOKBACK_ROWS = 5000  # 5_000
JUMP_SIZE = 500

MAX_VOLUME = 1000

ASK_PRICE_COLUMNS = [f"ask_px_{str(i).zfill(2)}" for i in range(10)]
ASK_VOL_COLUMNS = [f"ask_sz_{str(i).zfill(2)}" for i in range(10)]
ASK_COUNT_COLUMNS = [f"ask_ct_{str(i).zfill(2)}" for i in range(10)]
ASK_ANCHOR_COLUMN = ["ask_px_00"]


BID_PRICE_COLUMNS = [f"bid_px_{str(i).zfill(2)}" for i in range(10)]
BID_VOL_COLUMNS = [f"bid_sz_{str(i).zfill(2)}" for i in range(10)]
BID_COUNT_COLUMNS = [f"bid_ct_{str(i).zfill(2)}" for i in range(10)]
BID_ANCHOR_COLUMN = ["bid_px_00"]

# Initialize shard counters and WRITERS
SHARD_COUNTS = {"train": 0, "val": 0, "test": 0}
WRITERS = {"train": None, "val": None, "test": None}
SAMPLE_COUNTS = {"train": 0, "val": 0, "test": 0}
SAMPLES_PER_SHARD = 10000
WEBDATASET_OUTPUT_DIR = "./webdataset"
NBINS = 13

MOSIAC_DATASET_OUTPUT_DIR = f"./mosaic_dataset_forward{LOOKFORWARD_INPUT}_offset{LOOKFORWARD_OFFSET}_backward{LOOKBACK_ROWS}_{NBINS}bins"


def get_lob_filepaths(base_path: str, currency: str) -> list[str]:
    files = os.listdir(os.path.join(base_path, f"{currency}-micro"))
    return sorted([f"{base_path}/{currency}-micro/{f}" for f in files if ".zst" in f])


def decompress_file(path_to_file: str, default_output_file: str = "trades.dbn") -> None:
    with open(path_to_file, "rb") as f:
        decompressor = zstd.ZstdDecompressor()
        with open(default_output_file, "wb") as out:
            decompressor.copy_stream(f, out)


def dbd_to_df_max_symbol(dbn_filepath: str, majority_symbol: bool = True) -> pd.DataFrame:
    data = db.DBNStore.from_file(dbn_filepath)
    df = data.to_df()

    if majority_symbol:
        max_count = None
        max_symbol = None
        for symbol in df.symbol.unique():
            if max_count is None:
                max_count = df[df.symbol == symbol].shape[0]
                max_symbol = symbol
                continue

            symbol_count = df[df.symbol == symbol].shape[0]
            if symbol_count > max_count:
                max_count = symbol_count
                max_symbol = symbol

        return df[df.symbol == max_symbol]

    else:
        return df


def build_price_to_index(sub_df: pd.DataFrame) -> dict:
    most_recent_mid_price = (
        (sub_df[-1][ASK_ANCHOR_COLUMN] + sub_df[-1][BID_ANCHOR_COLUMN]) / 2
    ).to_numpy()[0][0]

    ask_bin_start = most_recent_mid_price + (1 / 2)
    bid_bin_start = most_recent_mid_price - (1 / 2)

    ask_price_bins = np.arange(ask_bin_start, ask_bin_start + (200 + 1), 1)
    bid_price_bins = np.arange(bid_bin_start, bid_bin_start - (200 + 1), -1)

    price_bins = np.array(list(bid_price_bins[::-1]) + list(ask_price_bins))

    # Create a dictionary {price: index}
    return {int(price): idx for idx, price in enumerate(price_bins)}


def build_ask_market_depth(sub_df: pd.DataFrame, price_to_index: dict) -> npt.NDArray:
    grouped_ask_price_columns = (
        sub_df[ASK_PRICE_COLUMNS + ["tick_bin"]].group_by(["tick_bin"]).mean().sort(by="tick_bin")
        // 1
    )  # deals with mid point median e.g. 2, 3 -> 2.5
    grouped_ask_volume_columns = (
        sub_df[ASK_VOL_COLUMNS + ["tick_bin"]]
        .group_by(["tick_bin"])
        .mean()
        .sort(by="tick_bin")[ASK_VOL_COLUMNS]
    )

    idx_columns = []
    for col in ASK_PRICE_COLUMNS:
        idx_column = f"{col}_idx"
        grouped_ask_price_columns = grouped_ask_price_columns.with_columns(
            pl.col(col).replace_strict(price_to_index, default=None).alias(idx_column)
        )
        idx_columns.append(idx_column)

    grouped_ask_price_columns = grouped_ask_price_columns[idx_columns]

    y_coords = grouped_ask_price_columns.to_numpy().T
    x_coords = np.tile(np.arange(y_coords.shape[1]), (y_coords.shape[0], 1))
    mapped_volumes = np.zeros((len(price_to_index), y_coords.shape[1])) * np.nan

    null_mask = np.isnan(y_coords)
    y_coords = y_coords[~null_mask].flatten().astype(int)
    x_coords = x_coords[~null_mask].flatten().astype(int)
    volume = grouped_ask_volume_columns.to_numpy().T[~null_mask].flatten()
    mapped_volumes[y_coords, x_coords] = volume

    nan_mask = np.isnan(mapped_volumes)
    mapped_volumes[nan_mask] = 0
    ask_market_volume = np.cumsum(mapped_volumes, axis=0)
    ask_market_volume = ask_market_volume[::-1, :]

    return ask_market_volume


def build_bid_market_depth(sub_df: pd.DataFrame, price_to_index: dict) -> npt.NDArray:
    grouped_bid_price_columns = (
        sub_df[BID_PRICE_COLUMNS + ["tick_bin"]].group_by(["tick_bin"]).mean().sort(by="tick_bin")
        // 1
    )  # deals with mid point median e.g. 2, 3 -> 2.5
    grouped_bid_volume_columns = (
        sub_df[BID_VOL_COLUMNS + ["tick_bin"]]
        .group_by(["tick_bin"])
        .mean()
        .sort(by="tick_bin")[BID_VOL_COLUMNS]
    )

    idx_columns = []
    for col in BID_PRICE_COLUMNS:
        idx_column = f"{col}_idx"
        grouped_bid_price_columns = grouped_bid_price_columns.with_columns(
            pl.col(col).replace_strict(price_to_index, default=None).alias(idx_column)
        )
        idx_columns.append(idx_column)

    grouped_bid_price_columns = grouped_bid_price_columns[idx_columns]

    y_coords = grouped_bid_price_columns.to_numpy().T
    x_coords = np.tile(np.arange(y_coords.shape[1]), (y_coords.shape[0], 1))
    mapped_volumes = np.zeros((len(price_to_index), y_coords.shape[1])) * np.nan

    null_mask = np.isnan(y_coords)
    y_coords = y_coords[~null_mask].flatten().astype(int)
    x_coords = x_coords[~null_mask].flatten().astype(int)
    volume = grouped_bid_volume_columns.to_numpy().T[~null_mask].flatten()
    mapped_volumes[y_coords, x_coords] = volume

    nan_mask = np.isnan(mapped_volumes)
    mapped_volumes[nan_mask] = 0
    mapped_volumes = mapped_volumes[::-1, :]

    bid_market_volume = np.cumsum(mapped_volumes, axis=0)

    return bid_market_volume


def get_writer(split):
    """Get or create writer for the specified split"""
    if WRITERS[split] is None or SAMPLE_COUNTS[split] >= SAMPLES_PER_SHARD:
        if WRITERS[split] is not None:
            WRITERS[split].close()
        SHARD_COUNTS[split] += 1
        shard_path = f"{WEBDATASET_OUTPUT_DIR}/{split}-{SHARD_COUNTS[split]:06d}.tar"
        WRITERS[split] = tarfile.open(shard_path, "w")
        SAMPLE_COUNTS[split] = 0
    return WRITERS[split]


def process_file(f):
    """Process one input file and yield samples"""

    decompress_file(f)
    df = dbd_to_df_max_symbol("trades.dbn")

    # Convert to int
    df[ASK_PRICE_COLUMNS] = (df[ASK_PRICE_COLUMNS] / MICRO_PIP_SIZE).round().astype(int)
    df[BID_PRICE_COLUMNS] = (df[BID_PRICE_COLUMNS] / MICRO_PIP_SIZE).round().astype(int)
    df["mid_price"] = df[ASK_ANCHOR_COLUMN + BID_ANCHOR_COLUMN].mean(axis=1)

    samples = []
    # print(df.shape[0]-LOOKFORWARD_ROWS-JUMP_SIZE)
    for stop_row in range(INPUT_ROWS, df.shape[0] - LOOKFORWARD_ROWS, JUMP_SIZE):
        # print(stop_row)
        start_row = stop_row - INPUT_ROWS
        target_start_row = stop_row + 1 + LOOKFORWARD_OFFSET
        target_stop_row = stop_row + LOOKFORWARD_ROWS
        # print(target_stop_row)
        # print(df.shape)
        # print(df.shape[0]-LOOKFORWARD_ROWS)
        # print()
        # Process input
        input_df = pl.DataFrame(df[start_row:stop_row])
        input_df = input_df.with_columns(
            (pl.int_range(0, INPUT_ROWS) // TICKS_PER_BIN).alias("tick_bin")
        )
        price_to_index = build_price_to_index(input_df)
        ask_depth = build_ask_market_depth(input_df, price_to_index)
        bid_depth = build_bid_market_depth(input_df, price_to_index)
        combined = ask_depth - bid_depth
        # combined = combined / np.max(np.abs(combined))
        combined = combined / MAX_VOLUME

        # mid_point = TARGET_ROWS // 2
        lookback_mean = (df["mid_price"][stop_row - LOOKBACK_ROWS : stop_row]).mean()
        lookforward_mean = (df["mid_price"][target_start_row:target_stop_row]).mean()

        sample_mid_price = df["mid_price"].iloc[stop_row]
        sample_point_price = df["mid_price"].iloc[target_stop_row - 2]

        lookforward_min = (df["mid_price"][target_start_row:target_stop_row]).min()
        lookforward_max = (df["mid_price"][target_start_row:target_stop_row]).max()
        mean_change = (lookforward_mean - lookback_mean) / lookback_mean
        sample_change = (sample_point_price - lookback_mean) / lookback_mean
        point_change = (sample_point_price - sample_mid_price) / sample_mid_price

        if NBINS == 13:
            if TICKS_PER_BIN == 100:
                if LOOKFORWARD_INPUT == 5000:
                    BIN_1 = 0.47 * TRUE_PIP_SIZE
                    BIN_2 = 1.55 * TRUE_PIP_SIZE
                    BIN_3 = 2.69 * TRUE_PIP_SIZE
                    BIN_4 = 3.92 * TRUE_PIP_SIZE
                    BIN_5 = 5.45 * TRUE_PIP_SIZE
                    BIN_6 = 7.73 * TRUE_PIP_SIZE

                if LOOKFORWARD_INPUT == 3000:
                    BIN_1 = 0.5 * TRUE_PIP_SIZE
                    BIN_2 = 1.7 * TRUE_PIP_SIZE
                    BIN_3 = 3 * TRUE_PIP_SIZE
                    BIN_4 = 4.3 * TRUE_PIP_SIZE
                    BIN_5 = 6 * TRUE_PIP_SIZE
                    BIN_6 = 8.45 * TRUE_PIP_SIZE

            if mean_change >= BIN_6:
                class_label = 12
            elif mean_change > BIN_5:
                class_label = 11
            elif mean_change > BIN_4:
                class_label = 10
            elif mean_change > BIN_3:
                class_label = 9
            elif mean_change > BIN_2:
                class_label = 8
            elif mean_change > BIN_1:
                class_label = 7
            elif mean_change > -BIN_1:
                class_label = 6
            elif mean_change > -BIN_2:
                class_label = 5
            elif mean_change > -BIN_3:
                class_label = 4
            elif mean_change > -BIN_4:
                class_label = 3
            elif mean_change > -BIN_5:
                class_label = 2
            elif mean_change > -BIN_6:
                class_label = 1
            else:
                class_label = 0

        if NBINS == 9:
            if TICKS_PER_BIN == 10:
                BIN_1 = 0.31 * TRUE_PIP_SIZE
                BIN_2 = 0.91 * TRUE_PIP_SIZE
                BIN_3 = 1.6 * TRUE_PIP_SIZE
                BIN_4 = 2.55 * TRUE_PIP_SIZE

            if TICKS_PER_BIN == 100:
                BIN_1 = 0.51 * TRUE_PIP_SIZE
                BIN_2 = 2.25 * TRUE_PIP_SIZE
                BIN_3 = 4 * TRUE_PIP_SIZE
                BIN_4 = 6.35 * TRUE_PIP_SIZE

            if mean_change >= BIN_4:
                class_label = 8
            elif mean_change > BIN_3:
                class_label = 7
            elif mean_change > BIN_2:
                class_label = 6
            elif mean_change > BIN_1:
                class_label = 5
            elif mean_change > -BIN_1:
                class_label = 4
            elif mean_change > -BIN_2:
                class_label = 3
            elif mean_change > -BIN_3:
                class_label = 2
            elif mean_change > -BIN_4:
                class_label = 1
            else:
                class_label = 0

        if NBINS == 7:
            if TICKS_PER_BIN == 10:
                BIN_1 = 0.3 * TRUE_PIP_SIZE
                BIN_2 = 0.9 * TRUE_PIP_SIZE
                BIN_3 = 1.7 * TRUE_PIP_SIZE
            if TICKS_PER_BIN == 100:
                BIN_1 = 0.7 * TRUE_PIP_SIZE
                BIN_2 = 2.7 * TRUE_PIP_SIZE
                BIN_3 = 5.5 * TRUE_PIP_SIZE

            if mean_change > BIN_3:
                class_label = 6
            elif mean_change > BIN_2:
                class_label = 5
            elif mean_change > BIN_1:
                class_label = 4
            elif mean_change > -BIN_1:
                class_label = 3
            elif mean_change > -BIN_2:
                class_label = 2
            elif mean_change > -BIN_3:
                class_label = 1
            else:
                class_label = 0

        if NBINS == 5:
            if TICKS_PER_BIN == 10:
                BIN_1 = TRUE_PIP_SIZE / 2
                BIN_2 = TRUE_PIP_SIZE * 1.5
            if TICKS_PER_BIN == 100:
                BIN_1 = 1 * TRUE_PIP_SIZE
                BIN_2 = 3 * TRUE_PIP_SIZE

            if mean_change > BIN_2:
                class_label = 4
            elif mean_change > BIN_1:
                class_label = 3
            elif mean_change > -BIN_1:
                class_label = 2
            elif mean_change > -BIN_2:
                class_label = 1
            else:
                class_label = 0

        # Create sample
        sample = {
            "__key__": f"sample-{uuid.uuid4().hex[:8]}-{stop_row:08d}",
            "input.npy": combined.astype(np.float32),
            "target.cls": class_label,
            "target.mean_reg": float(mean_change),
            "target.sample_reg": float(sample_change),
            "target.point_reg": float(point_change),
            "target.high_mid_reg": (lookforward_max - sample_mid_price) / sample_mid_price,
            "target.mid_low_reg": -((sample_mid_price - lookforward_min) / lookforward_min),
            "metadata.json": {
                "source_file": f,
                "start_row": start_row,
                "stop_row": stop_row,
                "target_start_row": target_start_row,
                "target_stop_row": target_stop_row,
                "lookforward_mean": lookforward_mean,
                "lookback_mean": lookback_mean,
            },
        }
        samples.append(sample)
    return samples


def create_sharded_webdataset(
    files_to_process,
    splits=(0.7, 0.15, 0.15),
):
    """

    Args:
        files_to_process: List of input files
        splits: Tuple of (train, val, test) ratios
        SAMPLES_PER_SHARD: Number of samples per tar file
        num_workers: Parallel processing threads
    """
    # Validate inputs
    assert sum(splits) == 1.0, "Splits must sum to 1.0"
    os.makedirs(WEBDATASET_OUTPUT_DIR, exist_ok=True)

    for i, f in tqdm(enumerate(files_to_process), total=len(files_to_process)):
        r = i / len(files_to_process)

        # if r < 0.65:
        #     continue

        if r < splits[0]:
            split = "train"
        elif r < splits[0] + splits[1]:
            split = "val"
        else:
            split = "test"

        try:
            samples = process_file(f)
        except Exception:
            continue

        for sample in samples:
            # Determine split (random assignment)

            # Get appropriate writer
            writer = get_writer(split)
            SAMPLE_COUNTS[split] += 1

            # Write sample to tar
            for key, value in sample.items():
                if key == "__key__":
                    continue

                if key.endswith(".npy"):
                    buf = io.BytesIO()
                    np.save(buf, value)
                    buf.seek(0)
                    tarinfo = tarfile.TarInfo(f"{sample['__key__']}.{key}")
                    tarinfo.size = buf.getbuffer().nbytes
                    writer.addfile(tarinfo, buf)
                elif key.endswith(".json"):
                    buf = io.BytesIO(json.dumps(value).encode())
                    tarinfo = tarfile.TarInfo(f"{sample['__key__']}.{key}")
                    tarinfo.size = buf.getbuffer().nbytes
                    writer.addfile(tarinfo, buf)
                else:  # scalar values
                    buf = io.BytesIO(str(value).encode())
                    tarinfo = tarfile.TarInfo(f"{sample['__key__']}.{key}")
                    tarinfo.size = buf.getbuffer().nbytes
                    writer.addfile(tarinfo, buf)

    # Close all WRITERS
    for writer in WRITERS.values():
        if writer is not None:
            writer.close()

    print("Dataset creation complete. Shards created:")
    print(f"Train: {SHARD_COUNTS['train']} shards, {SAMPLE_COUNTS['train']} samples")
    print(f"Val: {SHARD_COUNTS['val']} shards, {SAMPLE_COUNTS['val']} samples")
    print(f"Test: {SHARD_COUNTS['test']} shards, {SAMPLE_COUNTS['test']} samples")


def create_mosiac_dataset(files_to_process):
    """
    Creates sharded WebDataset with train/val/test splits

    Args:
        files_to_process: List of input files
        splits: Tuple of (train, val, test) ratios
        SAMPLES_PER_SHARD: Number of samples per tar file
        num_workers: Parallel processing threads
    """
    # Validate inputs
    n_files_to_process = len(files_to_process)
    train_stop = int(n_files_to_process * 0.7)
    val_start = train_stop + 1
    val_stop = int(n_files_to_process * 0.85)
    test_start = val_stop + 1

    train_files = files_to_process[: train_stop + 1]
    val_files = files_to_process[val_start : val_stop + 1]
    test_files = files_to_process[test_start:]

    # A dictionary mapping input fields to their data types
    columns = {
        "array": f"ndarray:float32:402,{INPUT_ROWS // TICKS_PER_BIN}",  # type, dtype, shape
        "class": "int",
        "mean_change": "float64",
        "sample_change": "float64",
        "point_change": "float64",
        "high_mid_reg": "float64",
        "mid_low_reg": "float64",
    }

    # Shard compression, if any
    # compression = 'zstd'
    compression = None

    # Save the samples as shards using MDSWriter
    for target, files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
        mean_list = []
        sample_list = []
        point_list = []
        with MDSWriter(
            out=f"{MOSIAC_DATASET_OUTPUT_DIR}/{target}/", columns=columns, compression=compression
        ) as out:
            for f in tqdm(files):
                try:
                    samples = process_file(f)
                    # alt_samples = alt_process_file(f)
                except Exception:
                    print("Failed on sampling")

                    continue

                for sample in samples:
                    to_write = {
                        "array": sample["input.npy"],
                        "class": sample["target.cls"],
                        "mean_change": sample["target.mean_reg"],
                        "sample_change": sample["target.sample_reg"],
                        "point_change": sample["target.point_reg"],
                        "high_mid_reg": sample["target.high_mid_reg"],
                        "mid_low_reg": sample["target.mid_low_reg"],
                    }
                    mean_list.append(sample["target.mean_reg"])
                    sample_list.append(sample["target.sample_reg"])
                    point_list.append(sample["target.point_reg"])
                    out.write(to_write)
                mean_array = np.array(mean_list)
                sample_array = np.array(sample_list)
                point_array = np.array(point_list)

                try:
                    r, p = pearsonr(mean_array, point_array)
                    print("Mean Point RMSE:", root_mean_squared_error(mean_array, point_array))
                    print("Mean Point MAE", mean_absolute_error(mean_array, point_array))
                    print(f"Mean - Point Correlation: {r}, p-value: {p}")

                    r, p = pearsonr(sample_array, point_array)
                    print("Sample Point RMSE:", root_mean_squared_error(sample_array, point_array))
                    print("Sample Point MAE", mean_absolute_error(sample_array, point_array))
                    print(f"Sample Point Correlation: {r}, p-value: {p}")
                    print()
                    print()

                    # _, bins = pd.qcut(mean_array, q=5, retbins=True)
                    # print(f"pd 5 qcut mean: {(bins / TRUE_PIP_SIZE).round(2)}")
                    # print("lt min bin mean: ", np.mean(mean_array[mean_array <= bins[1]]) / TRUE_PIP_SIZE)
                    # print("gt max bin mean: ", np.mean(mean_array[mean_array >= bins[-2]]) / TRUE_PIP_SIZE)
                    # _, bins = pd.qcut(sample_array, q=5, retbins=True)
                    # print(f"pd 5 qcut sample: {(bins / TRUE_PIP_SIZE).round(2)}")
                    # print("lt min bin mean: ", np.mean(sample_array[sample_array <= bins[1]]) / TRUE_PIP_SIZE)
                    # print("gt max bin mean: ", np.mean(sample_array[sample_array >= bins[-2]]) / TRUE_PIP_SIZE)
                    # _, bins = pd.qcut(point_array, q=5, retbins=True)
                    # print(f"pd 5 qcut point: {(bins / TRUE_PIP_SIZE).round(2)}")
                    # print("lt min bin mean: ", np.mean(point_array[point_array <= bins[1]]) / TRUE_PIP_SIZE)
                    # print("gt max bin mean: ", np.mean(point_array[point_array >= bins[-2]]) / TRUE_PIP_SIZE)
                    # print()
                    # _, bins = pd.qcut(mean_array, q=7, retbins=True)
                    # print(f"pd 7 qcut mean: {(bins / TRUE_PIP_SIZE).round(2)}")
                    # print("lt min bin mean: ", np.mean(mean_array[mean_array <= bins[1]]) / TRUE_PIP_SIZE)
                    # print("lt min bin mean: ", np.mean(mean_array[mean_array >= bins[-2]]) / TRUE_PIP_SIZE)
                    # _, bins = pd.qcut(sample_array, q=7, retbins=True)
                    # print(f"pd 7 qcut sample: {(bins / TRUE_PIP_SIZE).round(2)}")
                    # print("lt min bin mean: ", np.mean(sample_array[sample_array <= bins[1]]) / TRUE_PIP_SIZE)
                    # print("gt max bin mean: ", np.mean(sample_array[sample_array >= bins[-2]]) / TRUE_PIP_SIZE)
                    # _, bins = pd.qcut(point_array, q=7, retbins=True)
                    # print(f"pd 7 qcut point: {(bins / TRUE_PIP_SIZE).round(2)}")
                    # print("lt min bin mean: ", np.mean(point_array[point_array <= bins[1]]) / TRUE_PIP_SIZE)
                    # print("gt max bin mean: ", np.mean(point_array[point_array >= bins[-2]]) / TRUE_PIP_SIZE)
                    # print()
                    #     _, bins = pd.qcut(mean_array, q=9, retbins=True)
                    #     print(f"pd 9 qcut mean: {(bins / TRUE_PIP_SIZE).round(2)}")
                    #     print("lt min bin mean: ", np.mean(mean_array[mean_array <= bins[1]]) / TRUE_PIP_SIZE)
                    #     print("gt max bin mean: ", np.mean(mean_array[mean_array >= bins[-2]]) / TRUE_PIP_SIZE)
                    #     _, bins = pd.qcut(sample_array, q=9, retbins=True)
                    #     print(f"pd 9 qcut sample: {(bins / TRUE_PIP_SIZE).round(2)}")
                    #     print("lt min bin mean: ", np.mean(sample_array[sample_array <= bins[1]]) / TRUE_PIP_SIZE)
                    #     print("gt max bin mean: ", np.mean(sample_array[sample_array >= bins[-2]]) / TRUE_PIP_SIZE)
                    #     _, bins = pd.qcut(point_array, q=9, retbins=True)
                    #     print(f"pd 9 qcut point: {(bins / TRUE_PIP_SIZE).round(2)}")
                    #     print("lt min bin mean: ", np.mean(point_array[point_array <= bins[1]]) / TRUE_PIP_SIZE)
                    #     print("gt max bin mean: ", np.mean(point_array[point_array >= bins[-2]]) / TRUE_PIP_SIZE)

                    #     _, bins = pd.qcut(mean_array, q=11, retbins=True)
                    #     print(f"pd 11 qcut mean: {(bins / TRUE_PIP_SIZE).round(2)}")
                    #     print("lt min bin mean: ", np.mean(mean_array[mean_array <= bins[1]]) / TRUE_PIP_SIZE)
                    #     print("gt max bin mean: ", np.mean(mean_array[mean_array >= bins[-2]]) / TRUE_PIP_SIZE)
                    #     _, bins = pd.qcut(sample_array, q=11, retbins=True)
                    #     print(f"pd 11 qcut sample: {(bins / TRUE_PIP_SIZE).round(2)}")
                    #     print("lt min bin mean: ", np.mean(sample_array[sample_array <= bins[1]]) / TRUE_PIP_SIZE)
                    #     print("gt max bin mean: ", np.mean(sample_array[sample_array >= bins[-2]]) / TRUE_PIP_SIZE)
                    #     _, bins = pd.qcut(point_array, q=11, retbins=True)
                    #     print(f"pd 11 qcut point: {(bins / TRUE_PIP_SIZE).round(2)}")
                    #     print("lt min bin mean: ", np.mean(point_array[point_array <= bins[1]]) / TRUE_PIP_SIZE)
                    #     print("gt max bin mean: ", np.mean(point_array[point_array >= bins[-2]]) / TRUE_PIP_SIZE)

                    _, bins = pd.qcut(mean_array, q=13, retbins=True)
                    print(f"pd 13 qcut mean: {(bins / TRUE_PIP_SIZE).round(2)}")
                    print(
                        "lt min bin mean: ",
                        np.mean(mean_array[mean_array <= bins[1]]) / TRUE_PIP_SIZE,
                    )
                    print(
                        "gt max bin mean: ",
                        np.mean(mean_array[mean_array >= bins[-2]]) / TRUE_PIP_SIZE,
                    )
                    _, bins = pd.qcut(sample_array, q=13, retbins=True)
                    print(f"pd 13 qcut sample: {(bins / TRUE_PIP_SIZE).round(2)}")
                    print(
                        "lt min bin mean: ",
                        np.mean(sample_array[sample_array <= bins[1]]) / TRUE_PIP_SIZE,
                    )
                    print(
                        "gt max bin mean: ",
                        np.mean(sample_array[sample_array >= bins[-2]]) / TRUE_PIP_SIZE,
                    )
                    _, bins = pd.qcut(point_array, q=13, retbins=True)
                    print(f"pd 13 qcut point: {(bins / TRUE_PIP_SIZE).round(2)}")
                    print(
                        "lt min bin mean: ",
                        np.mean(point_array[point_array <= bins[1]]) / TRUE_PIP_SIZE,
                    )
                    print(
                        "gt max bin mean: ",
                        np.mean(point_array[point_array >= bins[-2]]) / TRUE_PIP_SIZE,
                    )
                    print()
                    print()
                    print()
                    print()
                except Exception as e:
                    print(e)


def main():
    files_to_process = get_lob_filepaths(DATA_DIR, CURRENCY)

    # create_sharded_webdataset(
    #     files_to_process=files_to_process,
    # )

    create_mosiac_dataset(files_to_process)


if __name__ == "__main__":
    main()
