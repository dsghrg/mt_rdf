import collections
import os

from tqdm import tqdm


def count_triples(file_path):
    subject_counts = collections.Counter()
    predicate_counts = collections.Counter()
    object_counts = collections.Counter()
    file_size = os.path.getsize(file_path)
    with open(file_path, 'r', buffering=1_048_576) as file:  # Use a 1MB buffer
        for line in tqdm(file, total=file_size, unit='B', unit_scale=True, desc="Processing"):
            parts = line.split()
            if len(parts) >= 3:
                subject = parts[0]
                predicate = parts[1]
                obj = parts[2]
                
                subject_counts[subject] += 1
                predicate_counts[predicate] += 1
                object_counts[obj] += 1

    return subject_counts, predicate_counts, object_counts

def save_counts_to_file(counts, output_file):
    with open(output_file, 'w') as file:
        for item, count in counts.items():
            file.write(f"{item}\t{count}\n")

def load_counts_from_file(output_file):
    counts = collections.Counter()
    with open(output_file, 'r') as file:
        for line in file:
            item, count = line.strip().split('\t')
            counts[item] = int(count)
    return counts

def get_file_size(file_path):
    return os.path.getsize(file_path)

def main(input_file):
    file_size = get_file_size(input_file)
    print(f"Processing file: {input_file} ({file_size / (1024 * 1024 * 1024):.2f} GB)")
    
    subject_counts, predicate_counts, object_counts = count_triples(input_file)
    
    save_counts_to_file(subject_counts, 'subject_counts.txt')
    save_counts_to_file(predicate_counts, 'predicate_counts.txt')
    save_counts_to_file(object_counts, 'object_counts.txt')
    
    print("Subject counts have been saved to subject_counts.txt")
    print("Predicate counts have been saved to predicate_counts.txt")
    print("Object counts have been saved to object_counts.txt")

if __name__ == "__main__":
    input_file = '../truthy_direct_properties.nt'
    
    main(input_file)
    
    # Load counts back from file (for demonstration purposes)
    loaded_subject_counts = load_counts_from_file('subject_counts.txt')
    loaded_predicate_counts = load_counts_from_file('predicate_counts.txt')
    loaded_object_counts = load_counts_from_file('object_counts.txt')
    
    print("Loaded subject counts from file:")
    for subject, count in list(loaded_subject_counts.items())[:10]:
        print(f"{subject}: {count}")
        
    print("Loaded predicate counts from file:")
    for predicate, count in list(loaded_predicate_counts.items())[:10]:
        print(f"{predicate}: {count}")
        
    print("Loaded object counts from file:")
    for obj, count in list(loaded_object_counts.items())[:10]:
        print(f"{obj}: {count}")