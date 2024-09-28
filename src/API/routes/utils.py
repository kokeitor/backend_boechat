import os


def list_files_in_directory(directory_path):
    files = []
    try:
        # List all files and directories in the specified directory
        with os.scandir(directory_path) as entries:
            for entry in entries:
                if entry.is_file():  # Check if it's a file
                    files.append(entry.name)  # Print the file name
                    print(entry.name)  # Print the file name
    except FileNotFoundError:
        print(f"Error: The directory {directory_path} does not exist.")
    except PermissionError:
        print(f"Error: You do not have permission to access {directory_path}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return files
