"""Generate N-by-N multiplication dataset with reversed-digit CoT.

Usage:
    python generate_mult_dataset.py --digits 12 --output_dir data/12_by_12_mult
    python generate_mult_dataset.py --digits 7 --output_dir data/7_by_7_mult --train_samples 100000
"""

import argparse
import os
import random


def generate_sample(n_digits, rng):
    """Generate one multiplication sample with CoT in reversed-digit format."""
    lo = 10 ** (n_digits - 1)
    hi = 10 ** n_digits - 1
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)

    a_digits = [int(d) for d in str(a)]
    b_digits = [int(d) for d in str(b)]

    # Reverse digits (LSB first)
    a_digits_rev = list(reversed(a_digits))
    b_digits_rev = list(reversed(b_digits))

    # Question
    question = ' '.join(str(d) for d in a_digits_rev) + ' * ' + ' '.join(str(d) for d in b_digits_rev)

    # CoT: multiply a by each digit of b_digits_rev (left to right = LSB to MSB)
    running = 0
    cot_parts = []
    for i, d in enumerate(b_digits_rev):
        p = a * d * (10 ** i)
        running += p
        pad_len = n_digits + 1 + i
        p_rev = str(p)[::-1].ljust(pad_len, '0')
        r_rev = str(running)[::-1].ljust(pad_len, '0')
        cot_parts.append((p_rev, r_rev))

    # Build CoT string
    cot = ' '.join(cot_parts[0][0])
    for i in range(1, len(cot_parts)):
        cot += ' + ' + ' '.join(cot_parts[i][0])
        if i < len(cot_parts) - 1:
            cot += ' ( ' + ' '.join(cot_parts[i][1]) + ' )'

    # Answer (reversed)
    answer = ' '.join(str(a * b)[::-1])

    return question + '||' + cot + ' #### ' + answer


def generate_file(path, n_digits, n_samples, seed):
    rng = random.Random(seed)
    with open(path, 'w') as f:
        for i in range(n_samples):
            line = generate_sample(n_digits, rng)
            f.write(line + '\n')


def main():
    parser = argparse.ArgumentParser(description='Generate N-by-N multiplication dataset')
    parser.add_argument('--digits', type=int, required=True, help='Number of digits per operand')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--train_samples', type=int, default=800000)
    parser.add_argument('--val_samples', type=int, default=1000)
    parser.add_argument('--test_samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Generating {args.digits}x{args.digits} multiplication dataset in {args.output_dir}/')

    generate_file(os.path.join(args.output_dir, 'train.txt'),
                  args.digits, args.train_samples, args.seed)
    print(f'  train.txt: {args.train_samples} samples')

    generate_file(os.path.join(args.output_dir, 'valid.txt'),
                  args.digits, args.val_samples, args.seed + 1)
    print(f'  valid.txt: {args.val_samples} samples')

    generate_file(os.path.join(args.output_dir, 'test_bigbench.txt'),
                  args.digits, args.test_samples, args.seed + 2)
    print(f'  test_bigbench.txt: {args.test_samples} samples')

    print('Done.')


if __name__ == '__main__':
    main()
