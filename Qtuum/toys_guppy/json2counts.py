import json
from collections import Counter


def read_bitstrings(file_path):
    counts = Counter()
    
    # Load the entire top-level JSON array
    with open(file_path, 'r') as f:
        all_shots = json.load(f)
        
    for shot in all_shots:
        # Convert each internal list like [["b0", 1], ["b1", 0], ...] into a dictionary
        bit_dict = {item[0]: item[1] for item in shot}
        
        # Combine bits into a single string (Qiskit style: MSB on the left -> b23 ... b0)
        nq = len(bit_dict)
        bitstr = "".join(str(bit_dict[f"b{i}"]) for i in reversed(range(nq)))
        
        counts[bitstr] += 1
            
    # Counter.most_common() automatically sorts descending by count.
    # Convert it back to a standard Python dictionary
    return dict(counts.most_common())

# --- Example Usage ---
if __name__ == "__main__":
    inpF='result_24q.json'
    result_dict = read_bitstrings(inpF)
    for bitstr, count in result_dict.items():
         print(f"  {bitstr} (dec={int(bitstr, 2):2d}): {count}")
    

