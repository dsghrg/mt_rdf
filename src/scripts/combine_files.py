import os


def combine_files(original_dir, target_dirs, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all filenames from the original directory
    filenames = os.listdir(original_dir)

    for filename in filenames:
        # Create the output file path
        output_path = os.path.join(output_dir, filename)

        with open(output_path, 'w') as output_file:
            # Write content from the original file
            original_path = os.path.join(original_dir, filename)
            output_file.write(f"# original\n")
            output_file.write(read_file(original_path))

            # Write content from each target directory
            for target_dir in target_dirs:
                target_name = os.path.basename(target_dir)
                target_path = os.path.join(target_dir, filename)
                output_file.write(f"\n# {target_name}\n")
                output_file.write(read_file(target_path))

def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return ""

if __name__ == "__main__":
    base_path = "data/queries/wdbench/"
    original_dir = f"{base_path}ppaths/original"
    target_dirs = [
        f"{base_path}ppaths/nl",
        f"{base_path}ppaths/opt_blaze",
        f"{base_path}ppaths/opt_virt"
    ]
    output_dir = f"{base_path}ppaths/combined"

    combine_files(original_dir, target_dirs, output_dir)
    print("Files combined successfully!")