import os
import shutil


def copy_and_rename_sparql_files(source_directory, target_directory):
    """
    Copies all 'query.sparql' files from a source directory to a target directory.
    Each copied file is renamed to its grandparent directory name with the '.sparql' extension.

    Args:
    source_directory (str): The root directory to search for 'query.sparql' files.
    target_directory (str): The directory to copy and rename the files to.
    """
    # Ensure the target directory exists
    os.makedirs(target_directory, exist_ok=True)
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(source_directory):
        for file in files:
            # Check if the current file is 'query.sparql'
            if file == 'query.sparql':
                # Construct the full path to the file
                full_file_path = os.path.join(root, file)
                
                # Get the parent directory name (this will be the new file name)
                parent_dir = os.path.basename(root)
                new_file_name = f"{parent_dir}.sparql"
                
                # Construct the full path for the new file location
                new_file_path = os.path.join(target_directory, new_file_name)
                
                # Copy and rename the file
                shutil.copy(full_file_path, new_file_path)
                print(f"Copied and renamed '{full_file_path}' to '{new_file_path}'")


def main():
    copy_and_rename_sparql_files('xp_blaze', 'data/queries/wdbench/ppaths/opt_blaze/')

if __name__ == "__main__":
    main()