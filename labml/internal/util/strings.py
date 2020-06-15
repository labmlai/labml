from typing import Iterable
import numpy as np


def is_pattern_match(key: str, pattern: str):
    dp = np.zeros((len(key) + 1, len(pattern) + 1), dtype=np.bool)
    dp[0, 0] = True

    for i in range(len(key)):
        for j in range(len(pattern)):
            if key[i] == pattern[j]:
                dp[i + 1][j + 1] = dp[i][j]
            elif pattern[j] == '?':
                dp[i + 1][j + 1] = dp[i][j]
            elif pattern[j] == '*':
                dp[i + 1][j + 1] = dp[i][j] or dp[i][j + 1] or dp[i + 1][j]

    return bool(dp[len(key), len(pattern)])


def find_best_pattern(key: str, patterns: Iterable[str]):
    max_score = -1
    best = None
    for p in patterns:
        if is_pattern_match(key, p):
            s = 0
            for c in p:
                if c not in {'*', '?'}:
                    s += 1

            if s > max_score:
                max_score = s
                best = p

    return best, max_score
