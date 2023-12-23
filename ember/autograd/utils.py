def inv_permutation(permutation):
    """
    Compute the inverse of a permutation.

    Args:
    - permutation (list): List representing a permutation.

    Returns:
    - list: Inverse permutation.

    """
    inverse = [0] * len(permutation)
    for original_idx, permuted_idx in enumerate(permutation):
        inverse[permuted_idx] = original_idx
    return inverse
