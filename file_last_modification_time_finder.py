# finding the file creatin time in miliseconds
import os
import time


def file_last_modification_time(file_path):
    return os.path.getmtime(file_path)


def is_file_last_modification_time_consecutive(file_path1, file_path2):
    """Check if the difference between file modification time is one second using datetime objects"""
    return file_last_modification_time_minutes_seconds(file_path1) != file_last_modification_time_minutes_seconds(file_path2)


# get the minutes and seconds from file last_modification time
def file_last_modification_time_minutes_seconds(file_path):
    return time.strftime('%H:%M:%S', time.localtime(file_last_modification_time(file_path)))


# get the file names of all json files in the folder
def get_json_files(json_file_folder):
    json_files = []
    for file in os.listdir(json_file_folder):
        if file.endswith(".json"):
            json_files.append(file)
    return json_files


# sort files by filename
def sort_files_by_filename(files):
    return sorted(files)


# find consecutive 2 json files where the creating second increases by 1
def find_consecutive_files_with_increasing_seconds(files):
    """Looping through all files in a folder and identifying those where the creating time (in second precision)
    switched seconds. These are saved in second_starters. The number of files between these files are saved in
    num_files_between."""

    second_starters = []
    num_files_between = []

    num_files_between_count = 0
    for i in range(len(files) - 1):
        if is_file_last_modification_time_consecutive(files[i], files[i + 1]):
            # print(f"Checking files: {files[i]} and {files[i + 1]}")
            # print(
            #     f"Stamps: {file_last_modification_time_minutes_seconds(files[i])} and {file_last_modification_time_minutes_seconds(files[i + 1])}")
            #print(f"Found consecutive files: {files[i]} and {files[i + 1]}")
            #print(f"Stamps: {file_last_modification_time_minutes_seconds(files[i])} and {file_last_modification_time_minutes_seconds(files[i + 1])}")
            second_starters.append(files[i])
            num_files_between.append(num_files_between_count)
            # print(second_starters, num_files_between)
            num_files_between_count = 0
        elif i == len(files) - 2:
            num_files_between_count += 1
            num_files_between.append(num_files_between_count)
            second_starters.append(files[i + 1])
        else:
            num_files_between_count += 1

    return second_starters, num_files_between
