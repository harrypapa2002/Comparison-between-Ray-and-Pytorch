import os
import zipfile

def unzip_twitter_mtx(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Extracted to: {extract_to}")
    except zipfile.BadZipFile:
        print("Error: The zip file is corrupted or invalid.")
    except Exception as e:
        print(f"An error occurred: {e}")

def read_after_comments(file_path, num_lines=50):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        non_comment_lines = [line.strip() for line in lines if not line.startswith('%')]

        print(f"\nFirst {num_lines} Non-Comment Lines of {file_path}:\n")
        for i, line in enumerate(non_comment_lines[:num_lines]):
            print(line)

    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

def reduce_file_size(input, data, target_size_gb):
    target_size = target_size_gb * (1024**3)  
    size = 0

    os.makedirs(os.path.dirname(data), exist_ok=True)

    try:
        with open(input, 'r') as input_file, open(data, 'w') as data_file:
            data_file.write("node1,node2\n")

            first_line_skipped = False  # Track if the first 3-value line is skipped

            for line in input_file:
                # Skip comment lines and the first data line with 3 values
                if line.startswith('%') or line.startswith('#'):
                    continue

                # Check for three-column first line (e.g., 41652230 41652230 1468365182)
                if not first_line_skipped and len(line.strip().split()) == 3:
                    first_line_skipped = True
                    continue  # Skip the first line with 3 columns

                # Process the remaining two-column lines
                csv_line = ','.join(line.strip().split()[:2]) + '\n'
                data_file.write(csv_line)

                size += len(line.encode('utf-8'))
                if size >= target_size:
                    break

        file_size_mb = round(os.path.getsize(data) / (1024**2), 2)
        file_size_gb = round(file_size_mb / 1024, 2)
        print(f"Reduced file saved as {data} ({file_size_mb} MB, {file_size_gb} GB)")

    except FileNotFoundError:
        print(f"Error: {input} not found.")
    except Exception as e:
        print(f"An error occurred during resizing: {e}")


# Get file size in MB and GB
def get_file_size(file_path):
    try:
        file_size = os.path.getsize(file_path)
        size_mb = round(file_size / (1024 ** 2), 2)
        size_gb = round(file_size / (1024 ** 3), 2)
        print(f"\nFile Size: {size_mb} MB ({size_gb} GB)")
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
    except Exception as e:
        print(f"An error occurred while getting file size: {e}")


# Paths (modify these based on your setup)
zip_path = r'C:\Users\anton\Downloads\twitter7.mtx.zip'  # Path to the zip file
extract_to = r'C:\Users\anton\Desktop\Pytorch data'      # Extraction directory
mtx_file = os.path.join(extract_to, 'twitter7.mtx')      # Extracted .mtx file path

# File paths for resized data
data_file = r'C:\Users\anton\Desktop\Pytorch data\twitter7_10gb.csv'
data_file_1 = r'C:\Users\anton\Desktop\Pytorch data\twitter7_1gb.csv'
data_file_2 = r'C:\Users\anton\Desktop\Pytorch data\twitter7_2.5gb.csv'
data_test = r'C:\Users\anton\Desktop\Pytorch data\twitter7_100mb.csv'
data_file_3 = r'C:\Users\anton\Desktop\Pytorch data\twitter7_1mb.csv'

# Uncomment to extract and inspect the file
# unzip_twitter_mtx(zip_path, extract_to)
# read_after_comments(mtx_file)
# get_file_size(mtx_file)

# Resize the graph to different sizes and save to individual files

#reduce_file_size(mtx_file, data_file, 10)      # 10GB version
#reduce_file_size(mtx_file, data_file_1, 1)     # 1GB version
#reduce_file_size(mtx_file, data_file_2, 2.5)   # 2.5GB version
reduce_file_size(mtx_file, data_test, 0.1)     # 100MB test version


#reduce_file_size(mtx_file, data_file_3, 0.001)     # 1MB test version

read_after_comments(data_file_3, 30)
