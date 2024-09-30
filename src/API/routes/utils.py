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


def delete_pdf_files(directory: str):
    """
    Iterates through a directory and deletes all .pdf files.

    :param directory: The path to the directory where .pdf files will be deleted.
    """
    try:
        # Check if the directory exists
        if not os.path.exists(directory):
            print(f"Directory '{directory}' does not exist.")
            return

        # Iterate through the files in the directory
        fileNames = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)

            # Check if the file is a PDF
            if filename.lower().endswith(".pdf"):
                try:
                    fileNames.append(filename)
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
            else:
                print(f"Skipped: {file_path} (not a PDF)")
    except Exception as e:
        print(f"Error while deleting files: {e}")
    return fileNames
