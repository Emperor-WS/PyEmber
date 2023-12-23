import requests
import tarfile
import zipfile
import gzip
import shutil
import os
import time


def get_time(start_time, end_time):
    """
    Calculates the elapsed time between a start and end time.

    Args:
        start_time (float): The starting time in seconds.
        end_time (float): The ending time in seconds.

    Returns:
        tuple: A tuple containing (elapsed_mins, elapsed_secs), where:
            - elapsed_mins (int): The elapsed minutes.
            - elapsed_secs (int): The elapsed seconds.
    """

    elapsed_time = end_time - start_time  # Calculate total elapsed time in seconds

    # Extract minutes and seconds from the elapsed time
    elapsed_mins = int(elapsed_time / 60)  # Divide elapsed time by 60 to get minutes
    # Subtract minutes * 60 to get remaining seconds
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


def progress_bar(current_index, max_index, prefix=None, suffix=None, start_time=None):
    """
    Prints a progress bar to the console.

    Args:
        current_index (int): The current index of the progress.
        max_index (int): The maximum index of the progress.
        prefix (str, optional): A prefix string to display before the progress bar. Defaults to None.
        suffix (str, optional): A suffix string to display after the progress bar. Defaults to None.
        start_time (float, optional): The starting time of the process in seconds. Used for calculating elapsed time.
    """

    # Set prefix to empty string if not provided
    prefix = "" if prefix is None else str(prefix) + " "

    # Calculate percentage progress and create visual progress bar
    percentage = current_index * 100 // max_index
    # Create bar with "=" for progress
    loading = "[" + "=" * (percentage // 2) + " " * (50 - percentage // 2) + "]"
    progress_display = "\r{0}{1:3d}% | {2}".format(
        prefix, percentage, loading)  # Format progress display string

    # Add suffix if provided
    progress_display += "" if suffix is None else " | " + str(suffix)

    # Add elapsed time if start_time is provided
    if start_time is not None:
        time_min, time_sec = get_time(start_time, time.time())  # Get elapsed time
        time_display = " | Time: {0}m {1}s".format(
            time_min, time_sec)  # Format time display
        progress_display += time_display

    # Print the progress display and manage end-of-progress formatting
    print(progress_display, end="{}".format(
        "" if current_index < max_index else " | Done !\n"))


def download_from_url(url, save_path, chunk_size=128):
    """
    Downloads a file from a given URL and saves it to the specified path.

    Args:
        url (str): The URL of the file to download.
        save_path (str): The path to save the downloaded file.
        chunk_size (int, optional): The size of chunks to download in bytes. Defaults to 128.
    """

    response = requests.get(url, stream=True)  # Start streaming download
    total = response.headers.get('content-length')  # Get total file size from headers

    with open(save_path, 'wb') as f:  # Open file for writing in binary mode
        if total is None:  # If total size is not available, download in one go
            f.write(response.content)
        else:  # If total size is available, download in chunks
            downloaded = 0
            total = int(total)  # Convert total size to integer
            # Iterate through chunks
            for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                downloaded += len(data)  # Update downloaded bytes with each chunk size
                f.write(data)  # Write the downloaded chunk to the file
                # Show progress bar with downloaded and total size
                progress_bar(downloaded, total, "Downloading...")


def extract_to_dir(filename, dirpath='.'):
    """
    Extracts the contents of an archive file to a specified directory.

    Args:
        filename (str): The path to the archive file.
        dirpath (str, optional): The directory to extract to. Defaults to the current directory '.'.

    Returns:
        str: The absolute path to the extraction directory.
    """

    name, ext = os.path.splitext(filename)  # Split filename into name and extension

    print(dirpath)
    print("Extracting...", end="")

    # Handle different archive types
    if tarfile.is_tarfile(filename):
        tarfile.open(filename, 'r').extractall(dirpath)  # Extract TAR archive
    elif zipfile.is_zipfile(filename):
        zipfile.ZipFile(filename, 'r').extractall(dirpath)  # Extract ZIP archive
    elif ext == '.gz':
        # Handle gzip files differently (not extracted, just moved)
        if not os.path.exists(dirpath):  # Create the directory if it doesn't exist
            os.mkdir(dirpath)
        # Move gzip file to directory
        shutil.move(filename, os.path.join(dirpath, os.path.basename(filename)))
        print(f" | NOTE: gzip files are not extracted, and moved to {dirpath}", end="")

    print(" | Done !")
    # Return the absolute path to the extraction directory
    return os.path.abspath(dirpath)
