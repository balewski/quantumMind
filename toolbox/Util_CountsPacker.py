import numpy as np


def pack_counts_for_npz(data):
    """
    Pack a list of bitstring-frequency dictionaries into arrays suitable for npz storage.
    Supports arbitrary bitstring lengths (e.g. 200+ qubits).

    Args:
        data: list of dicts, e.g. [{'000': 52, '111': 48}, {'0001': 12, '1010': 48}]

    Returns:
        dict with keys:
            'meas_num_bits':  int32 array of shape (num_dicts,)
            'meas_dict_ids':  int32 array
            'meas_counts':    int32 array
            'meas_bitstring_bytes': uint8 2D array of shape (num_entries, max_bytes)
    """
    dict_ids = []
    counts = []
    num_bits = []
    raw_bytes_list = []

    for i, d in enumerate(data):
        if not d:
            num_bits.append(0)
            continue
        first_key = next(iter(d))
        nb = len(first_key)
        num_bits.append(nb)
        n_bytes = (nb + 7) // 8
        for bitstring, count in d.items():
            dict_ids.append(i)
            counts.append(count)
            raw_bytes_list.append(
                int(bitstring, 2).to_bytes(n_bytes, byteorder='big')
            )

    # Pad all byte representations to the same length for a rectangular array
    if raw_bytes_list:
        max_bytes = max(len(b) for b in raw_bytes_list)
        bitstring_bytes = np.zeros((len(raw_bytes_list), max_bytes), dtype=np.uint8)
        for j, b in enumerate(raw_bytes_list):
            # Right-align: pad on the left with zeros
            offset = max_bytes - len(b)
            bitstring_bytes[j, offset:] = list(b)
    else:
        bitstring_bytes = np.zeros((0, 0), dtype=np.uint8)

    return {
        'meas_num_bits': np.array(num_bits, dtype=np.int32),
        'meas_dict_ids': np.array(dict_ids, dtype=np.int32),
        'meas_counts': np.array(counts, dtype=np.int32),
        'meas_bitstring_bytes': bitstring_bytes,
    }


def unpack_counts_from_npz(packed):
    """
    Reconstruct list of bitstring-frequency dictionaries from packed arrays.

    Args:
        packed: dict (or NpzFile) with keys 'meas_num_bits', 'meas_dict_ids',
                'meas_counts', 'meas_bitstring_bytes'

    Returns:
        list of dicts, e.g. [{'000': 52, '111': 48}, {'0001': 12, '1010': 48}]
    """
    num_bits = packed['meas_num_bits']
    dict_ids = packed['meas_dict_ids']
    counts = packed['meas_counts']
    bs_bytes = packed['meas_bitstring_bytes']

    num_dicts = len(num_bits)
    data = [{} for _ in range(num_dicts)]

    for j in range(len(dict_ids)):
        di = int(dict_ids[j])
        nb = int(num_bits[di])
        # Convert bytes back to int, then to zero-padded bitstring
        val = int.from_bytes(bs_bytes[j].tobytes(), byteorder='big')
        bitstring = format(val, f'0{nb}b')
        data[di][bitstring] = int(counts[j])

    return data


# ---- test ----
if __name__ == '__main__':
    import random

    # Small mixed-width test
    original = [
        {'000': 52, '111': 48},
        {'0001': 12, '1010': 48},
        {},
        {'10': 7, '01': 3, '11': 90},
    ]

    # Simulate a 200-qubit dictionary
    big_dict = {}
    for _ in range(1000):
        bs = ''.join(random.choice('01') for _ in range(200))
        big_dict[bs] = big_dict.get(bs, 0) + 1
    original.append(big_dict)

    packed = pack_counts_for_npz(original)
    print("Packed arrays:")
    for k, v in packed.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    reconstructed = unpack_counts_from_npz(packed)

    for i, (orig, reco) in enumerate(zip(original, reconstructed)):
        assert orig == reco, f"Mismatch at index {i}"
        print(f"  dict[{i}] ✓  ({len(reco)} entries)")

    # NPZ round-trip
    np.savez_compressed('/tmp/test_counts.npz', **packed)
    loaded = np.load('/tmp/test_counts.npz')
    reconstructed2 = unpack_counts_from_npz(loaded)
    assert reconstructed2 == original
    print("NPZ round-trip ✓")
