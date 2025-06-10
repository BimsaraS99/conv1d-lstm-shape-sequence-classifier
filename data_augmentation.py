import csv

def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        return list(reader)

def write_csv(filename, data):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

def reverse_sequence(sequence):
    """Reverse the sequence while keeping the label at position 0"""
    label = sequence[0]
    points = sequence[1:]
    return [label] + points[::-1]

def cut_and_join(sequence):
    """Cut sequence from middle and join first part with last part"""
    label = sequence[0]
    points = sequence[1:]
    
    # Split in middle
    mid = len(points) // 2
    first_part = points[:mid]
    second_part = points[mid:]
    
    # Join second part + first part
    return [label] + second_part + first_part

def augment_dataset(input_file, output_file):
    data = read_csv(input_file)
    augmented_data = []
    
    for sequence in data:
        # Original sequence
        augmented_data.append(sequence)
        
        # Reversed sequence
        reversed_seq = reverse_sequence(sequence)
        augmented_data.append(reversed_seq)
        
        # Cut and joined sequence
        cut_joined_seq = cut_and_join(sequence)
        augmented_data.append(cut_joined_seq)
        
        # Reversed then cut and joined sequence
        reversed_then_cut = cut_and_join(reversed_seq)
        augmented_data.append(reversed_then_cut)
    
    write_csv(output_file, augmented_data)

# Usage
augment_dataset('dataset/raw_sequences.csv', 'augmented_data.csv')