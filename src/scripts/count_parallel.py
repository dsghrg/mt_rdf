import collections
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def process_lines(lines):
    subject_counts = collections.Counter()
    predicate_counts = collections.Counter()
    object_counts = collections.Counter()

    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            subject = parts[0]
            predicate = parts[1]
            obj = parts[2]
            
            subject_counts[subject] += 1
            predicate_counts[predicate] += 1
            object_counts[obj] += 1

    return subject_counts, predicate_counts, object_counts

def save_partial_counts(counts, filename):
    with open(filename, 'a') as file:
        for item, count in counts.items():
            file.write(f"{item}\t{count}\n")

def count_triples(file_path, chunk_size=100000):
    subject_counts = collections.Counter()
    predicate_counts = collections.Counter()
    object_counts = collections.Counter()
    
    with open(file_path, 'r', buffering=1_048_576) as file:
        chunk = []
        for line in tqdm(file, unit='lines', desc="Processing"):
            chunk.append(line)
            if len(chunk) >= chunk_size:
                sub_counts, pred_counts, obj_counts = process_lines(chunk)
                save_partial_counts(sub_counts, 'partial_subject_counts.txt')
                save_partial_counts(pred_counts, 'partial_predicate_counts.txt')
                save_partial_counts(obj_counts, 'partial_object_counts.txt')
                chunk = []

        if chunk:
            sub_counts, pred_counts, obj_counts = process_lines(chunk)
            save_partial_counts(sub_counts, 'partial_subject_counts.txt')
            save_partial_counts(pred_counts, 'partial_predicate_counts.txt')
            save_partial_counts(obj_counts, 'partial_object_counts.txt')

def combine_partial_counts(partial_file):
    counts = collections.Counter()
    with open(partial_file, 'r') as file:
        for line in file:
            item, count = line.strip().split('\t')
            counts[item] += int(count)
    return counts

def save_counts_to_file(counts, output_file):
    with open(output_file, 'w') as file:
        for item, count in counts.items():
            file.write(f"{item}\t{count}\n")

def main(input_file):
    print(f"Processing file: {input_file}")
    
    count_triples(input_file)
    
    subject_counts = combine_partial_counts('partial_subject_counts.txt')
    predicate_counts = combine_partial_counts('partial_predicate_counts.txt')
    object_counts = combine_partial_counts('partial_object_counts.txt')

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
    loaded_subject_counts = combine_partial_counts('subject_counts.txt')
    loaded_predicate_counts = combine_partial_counts('predicate_counts.txt')
    loaded_object_counts = combine_partial_counts('object_counts.txt')
    
    print("Loaded subject counts from file:")
    for subject, count in list(loaded_subject_counts.items())[:10]:
        print(f"{subject}: {count}")
        
    print("Loaded predicate counts from file:")
    for predicate, count in list(loaded_predicate_counts.items())[:10]:
        print(f"{predicate}: {count}")
        
    print("Loaded object counts from file:")
    for obj, count in list(loaded_object_counts.items())[:10]:
        print(f"{obj}: {count}")