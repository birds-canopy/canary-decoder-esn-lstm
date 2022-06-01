import math
import itertools

from collections import defaultdict, Counter

import numpy as np


def remove(seq, labels):
    """
    Remove some labels in a sequence.
    """
    filtered = seq.copy()
    for lbl in labels:
        filtered = filtered[filtered != lbl]

    return filtered


def group(seq, min_frame_nb=0, exclude=("SIL", "TRASH")):
    """
    Group all consecutive equivalent labels in a discrete
    time sequence into a single label,
    with number of frames grouped attached. All groups of
    consecutive predictions
    composed of N < min_frame_nb items will be removed.
    """
    # first grouping
    newseq = [s for s in seq if s not in exclude]
    newseq = [list(g) for k, g in itertools.groupby(newseq)]
    newseq = [rseq for rseq in newseq
              if len(rseq) >= min_frame_nb]

    newseq = flatten(newseq)

    grouped_sequence = [k for k, g in itertools.groupby(newseq)]

    return grouped_sequence


def majority_vote(seq, win_length=1, exclude=("SIL", "TRASH")):

    #ss = [s for s in seq if s not in exclude]

    # majority vote over sliding windows
    new_seq = []
    for win in ngrams(seq, win_length):
        counts = Counter(win).most_common()
        new_seq.append(counts[0][0])

    return new_seq


def lev_dist(s, t):
    """
    Levenshtein distance between two sequences S and T.
    Returns the minimum number of operations (insertions, substitutions, deletions)
    to perform to perfectly match S and T.
    """

    v0 = np.arange(len(t)+1)
    v1 = np.zeros(len(t)+1)

    for i in range(0, len(s)):
        v1[0] = i + 1

        for j in range(0, len(t)):
            delcost = v0[j+1] + 1  # deletions
            incost = v1[j] + 1  # insertions
            if s[i] == t[j]:
                subcost = v0[j]  # substitutions
            else:
                subcost = v0[j] + 1

            v1[j + 1] = min([delcost, incost, subcost])

        v0, v1 = v1, v0

    return v0[len(t)]


def word_error_rate(true, pred):

    edit_dist = lev_dist(pred, true)
    return edit_dist / len(true)


def lev_sim(s, t):
    """
    Similarity score based on minimum Levenshtein distance between two
    sequences S and T.
    Returns:
    .. math::
        \frac{|s| + |t| - lenvenshtein(s, t)}{|s| + |t|}

    """
    return ((len(s)+len(t)) - lev_dist(s, t)) / (len(s)+len(t))


def lcs(s, t):
    """
    Longest Common Subsequence between two sequences. R
    Returns the maximum length matching subsequence between two sequences.
    Similar to the Levenshtein distance but subsitutions are not allowed
    to build the matching subsequences.
    """
    c = np.zeros((len(s)+1, len(t)+1))

    for i in range(1, len(s)+1):
        for j in range(1, len(t)+1):
            if s[i-1] == t[j-1]:
                c[i, j] = c[i-1, j-1] + 1
            else:
                c[i, j] = max(c[i, j-1], c[i-1, j])
    return c[len(s), len(t)]


def lcs_ratio(s, t):
    """
    Similarity score based on LCS between two sequences S and T.
    Returns:
    .. math::
        \frac{lcs(s, t)}{max(|s), |t|)}
    """
    return lcs(s, t) / max(len(s), len(t))


def flatten(lists):
    """
    Flatten a list of lists into a simple list.
    """
    flat = []
    for ls in lists:
        for e in ls:
            flat.append(e)
    return flat


def ngrams(seq, n=1):
    """
    Generate ngrams from a sequence.
    """
    for i in range(len(seq)-n+1):
        yield tuple(seq[i:i+n])


def ngram_occurences(doc, n=1):
    """
    Frequencies of ngrams in set of sequences.
    """

    occurences = defaultdict(lambda: defaultdict(int))
    for seq in doc:
        for ng in ngrams(seq, n=n):
            occurences[ng[:n-2]][ng[n-1]] += 1

    for ngram in occurences:
        total = float(sum(occurences[ngram].values()))
        for s in occurences[ngram]:
            occurences[ngram][s] /= total

    return occurences


def to_seconds(annots, config):

    new_annots = {}
    for m, model_annots in annots.items():
        new_annots[m] = {}
        for s, song_annots in model_annots.item():
            new_annots[m][s] = []
            for i, a in enumerate(song_annots):
                t = config.to_duration(a[1])
                new_a = (a[0], t)
                new_annots.append(new_a)
    return new_annots
