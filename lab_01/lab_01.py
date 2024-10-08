import os
from itertools import repeat
import fastparquet
import pandas as pd
from filesplit.split import Split
from multiprocessing import Pool, cpu_count
from datetime import datetime

EXTERNAL_DATA_DIR_PATH = '../external_data/'
MERGED_FILE_PATH = '../external_data/merged_data.parquet'
CSV_FILE_PATH = '../external_data/merged_data.csv'


def count_time(func):
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)  # Wywołanie funkcji i zapisanie wyniku
        print(f"Czas wczytywania {func.__name__}: {datetime.now() - start}")
        return result  # Zwracanie wyniku, a nie ponowne wywołanie
    return wrapper


def apply_args_and_kwargs(func, args, kwargs):
    return func(*args, **kwargs)


def starmap_with_kwargs(pool, func, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(func), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)


def split_file(filepath, chunksize, destination):
    split = Split(filepath, destination)
    split.bylinecount(linecount=chunksize, includeheader=True)


@count_time
def load_files(directory):
    files = [[f"{directory}/{f}"] for f in os.listdir(directory) if f.endswith(".parquet")]

    kwargs_list = [
        {
            'columns': None,
        }
        for n in range(len(files))
    ]

    pool = Pool(processes=5)
    args_iter = files

    results = starmap_with_kwargs(pool, pd.read_parquet, args_iter, kwargs_list)
    results = pd.concat(results)

    return results


def optimize_types(df):
    df['sid'] = pd.to_numeric(df['sid'], downcast='integer')
    df['sid_profile'] = pd.to_numeric(df['sid_profile'], downcast='integer')
    df['profile_id'] = pd.to_numeric(df['profile_id'], downcast='integer')
    df['post_type'] = df['post_type'].astype('int8')
    df['likes'] = pd.to_numeric(df['likes'], downcast='integer')
    df['comments'] = pd.to_numeric(df['comments'], downcast='integer')
    df['following'] = pd.to_numeric(df['following'], downcast='integer')
    df['followers'] = pd.to_numeric(df['followers'], downcast='integer')
    df['num_posts'] = pd.to_numeric(df['num_posts'], downcast='integer')

    df['lang'] = df['lang'].astype('category')
    df['category'] = df['category'].astype('category')

    return df


@count_time
def group_and_aggregate(df):
    return df.groupby('post_type').agg({'likes': 'sum', 'comments': 'mean'}).reset_index()


@count_time
def filter_data(df):
    return df[df['likes'] > 1000]


@count_time
def sort_data(df):
    return df.sort_values(by='followers', ascending=False)


# Funkcja do wykonania operacji na oryginalnych i zoptymalizowanych danych
def perform_operations(original_df, optimized_df):
    print("=== Oryginalne dane ===")
    original_grouped = group_and_aggregate(original_df)
    original_filtered = filter_data(original_df)
    original_sorted = sort_data(original_df)

    print("\n=== Zoptymalizowane dane ===")
    optimized_grouped = group_and_aggregate(optimized_df)
    optimized_filtered = filter_data(optimized_df)
    optimized_sorted = sort_data(optimized_df)

    return original_grouped, original_filtered, original_sorted, optimized_grouped, optimized_filtered, optimized_sorted


@count_time
def save_to_csv(df, csv_file_path):
    df.to_csv(csv_file_path, index=False)
    print(f"Dane zapisane do pliku {csv_file_path}")


@count_time
def load_csv(file_path):
    # engine='python' zamiast 'c', ponieważ:
    # Error tokenizing data. C error: Buffer overflow caught - possible malformed input file.
    df = pd.read_csv(file_path, engine='python', encoding='utf-8')
    print(f"Wczytano dane z pliku CSV ({file_path})")
    return df


@count_time
def load_csv_with_chunksize(file_path, chunksize):
    chunk_list = []
    for chunk in pd.read_csv(file_path, chunksize=chunksize, engine='python', encoding='utf-8'):
        chunk_list.append(chunk)
    df = pd.concat(chunk_list, ignore_index=True)
    print(f"Wczytano dane z pliku CSV ({file_path}) z użyciem chunksize = {chunksize}")
    return df


def read_csv_chunk(file_path, skiprows, chunksize):
    return pd.read_csv(file_path, skiprows=skiprows, chunksize=chunksize, engine='python').get_chunk()


@count_time
def load_csv_multiprocessing(file_path, n_processes):
    file_size = os.path.getsize(file_path)
    chunk_size = file_size // n_processes

    with Pool(processes=n_processes) as pool:
        results = pool.starmap(
            read_csv_chunk,
            [(file_path, i * chunk_size, chunk_size) for i in range(n_processes)]
        )

    df = pd.concat(results, ignore_index=True)
    print(f"Wczytano dane z pliku CSV ({file_path}) z multiprocessingiem ({n_processes} procesy)")
    return df


def sum_likes_sequential(file_paths):
    total_likes = 0
    for file_path in file_paths:
        df = load_csv(file_path)
        total_likes += df['likes'].sum()
    return total_likes


def sum_likes_multiprocessing(file_paths, n_processes):
    def load_and_sum(file_path):
        df = load_csv(file_path)
        return df['likes'].sum()

    with Pool(processes=n_processes) as pool:
        results = pool.map(load_and_sum, file_paths)
    return sum(results)


if __name__ == "__main__":
    # Zadanie 1: Wczytywanie danych
    if os.path.exists(MERGED_FILE_PATH):
        print(f"Plik {MERGED_FILE_PATH} już istnieje. Wczytywanie istniejącej ramki danych.")
        merged_df = pd.read_parquet(MERGED_FILE_PATH)
    else:
        print("Plik nie istnieje. Wczytywanie i scalanie plików .parquet.")
        merged_df = load_files(EXTERNAL_DATA_DIR_PATH)
        merged_df.to_parquet(MERGED_FILE_PATH)
        print(f"Zapisano scaloną ramkę danych jako {MERGED_FILE_PATH}.")

    print("Typy danych w ramce DataFrame:")
    print(merged_df.dtypes)

    # Zadanie 2: Optymalizacja typów
    df_optimized = optimize_types(merged_df.copy())

    # Zmierzenie parametrów zużycia pamięci
    original_memory_usage = merged_df.memory_usage(deep=True).sum() / (1024 ** 2)  # w MB
    optimized_memory_usage = df_optimized.memory_usage(deep=True).sum() / (1024 ** 2)  # w MB

    # Wyświetlenie zużycia pamięci
    print('===============================')
    print(f"Zużycie pamięci przed optymalizacją: {original_memory_usage:.2f} MB")
    print(f"Zużycie pamięci po optymalizacji: {optimized_memory_usage:.2f} MB")

    # Zadanie 3
    perform_operations(merged_df, df_optimized)

    # Zadanie 4: Zapis danych do CSV
    save_to_csv(df_optimized, CSV_FILE_PATH)

    # Zadanie 5: Różne sposoby wczytywania danych
    # 1. Wczytywanie całego pliku na raz
    df_csv = load_csv(CSV_FILE_PATH)

    if df_csv is not None:
        df_csv.info()

    # 2. Wczytywanie pliku z chunksize
    chunk_size = 10 ** 6
    df_csv_chunksize = load_csv_with_chunksize(CSV_FILE_PATH, chunksize=chunk_size)

    # 3. Wczytywanie pliku z multiprocessingiem
    n_cores = cpu_count() - 2
    df_csv_multiproc = load_csv_multiprocessing(CSV_FILE_PATH, n_cores)

    # 4. Wczytywanie z większą ilością procesów (2x więcej niż ilość rdzeni - 2)
    df_csv_multiproc_more = load_csv_multiprocessing(CSV_FILE_PATH, n_cores * 2)

    # Zadanie 6: Sumowanie likes
    file_paths = [os.path.join(EXTERNAL_DATA_DIR_PATH, f) for f in os.listdir(EXTERNAL_DATA_DIR_PATH) if f.endswith('.csv')]

    # Sekwencyjne sumowanie likes
    total_likes_sequential = sum_likes_sequential(file_paths)
    print(f"Suma likes (sekwencyjnie): {total_likes_sequential}")

    # Równoległe sumowanie likes
    total_likes_multiprocessing = sum_likes_multiprocessing(file_paths, n_cores)
    print(f"Suma likes (multiprocessing): {total_likes_multiprocessing}")
