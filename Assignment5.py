import numpy as np
from tqdm import tqdm
from scipy.special import comb


def loadLastCol(filename):
    LastCol = ''.join(np.loadtxt(filename, dtype=str))
    return LastCol

def loadref(filename):
    RefSeq = ''.join(np.loadtxt(filename, dtype=str)[1:])
    return RefSeq

def loadreads(filename):
    Reads = np.loadtxt(filename, dtype=str)
    return Reads

def loadmap(filename):
    MapToRefSeq = np.loadtxt(filename, dtype=int)
    return MapToRefSeq


LastCol = loadLastCol("chrX_bwt/chrX_last_col.txt")
RefSeq = loadref("chrX_bwt/chrX.fa")
Reads = loadreads("chrX_bwt/reads")
Map = loadmap("chrX_bwt/chrX_map.txt")

red_pos = np.array([
    [149249757, 149249868], [149256127, 149256423], [149258412, 149258580],
    [149260048, 149260213], [149261768, 149262007], [149264290, 149264400]
])
green_pos = np.array([
    [149288166, 149288277], [149293258, 149293554], [149295542, 149295710],
    [149297178, 149297343], [149298898, 149299137], [149301420, 149301530]
])
rg_positions = [red_pos[:, 0], red_pos[:, 1], green_pos[:, 0], green_pos[:, 1]]

class Milestones:
    def __init__(self, last_col, k=200):
        length = len(last_col)
        pad_length = (k - length % k) % k  
        padded_col = last_col + '$' * pad_length
        self.k = k
        self.last_col_arr = last_col
        last_col_np = np.array(list(padded_col)).reshape(-1, k)
        unique_chars, counts = np.unique(list(last_col), return_counts=True)
        unique_chars, counts = unique_chars[1:], counts[1:] # ignoring the $
        self.char_start_index = {'A': 0, 'C': counts[0] , 'G': counts[0] + counts[1], 'T': counts[0] + counts[1] + counts[2]}
        self.char_end_index = {'A': self.char_start_index['C'] - 1, 'C': self.char_start_index['G'] - 1, 'G': self.char_start_index['T'] - 1, 'T': sum(counts) - 1}
        self.cumulatives = {}
        
        for c in unique_chars:
            temp = (last_col_np == c).sum(axis=1).cumsum()
            self.cumulatives[c] = temp
            
        del (last_col_np) 
        
    def last_col_rank(self, char, index):
        block_index = (index + 1) // self.k
        rank = self.cumulatives[char][block_index - 1] if block_index > 0 else 0
        start_pos = block_index * self.k
        rank += sum(1 for j in range(start_pos, index + 1) if self.last_col_arr[j] == char)
        return rank

    def col1_index_rank(self, char, rank):
        return self.char_start_index[char] + rank - 1

    def col1_start_index(self, char):
        return self.char_start_index[char]

    def col1_end_index(self, char):
        return self.char_end_index[char]
    
    def get_window(self, input_string):
    
        length = len(input_string)
        i = length - 1
        start, end = self.col1_start_index(input_string[i]), self.col1_end_index(input_string[i])
        while i > 0:
            i -= 1
            c = input_string[i]
            srank, erank = self.last_col_rank(c, start), self.last_col_rank(c, end)
            srank += (1 if srank < erank and self.last_col_arr[start] != c else 0)
            if srank == erank:
                while start <= end and self.last_col_arr[start] != c:
                    start += 1
                if start > end:
                    break
            start, end = self.col1_index_rank(c, srank), self.col1_index_rank(c, erank)
            if start == end:
                return start, end, i

        return start, end, i

def valid(i, string, RefSeq = RefSeq, maxmismatch=2):
    mismatch, j = 0, 0
    length = len(string)
    while j < length and mismatch <= maxmismatch:
        mismatch += (RefSeq[i] != string[j])
        i += 1
        j += 1
    return mismatch <= maxmismatch


def identify_exon(positions, r_left, r_right, g_left, g_right):
    R = np.zeros(6)
    G = np.zeros(6)
    
    for each in positions:
        a = (r_left <= each) & (each <= r_right)
        b = (g_left <= each) & (each <= g_right)
        if np.sum(a) == 1 and np.sum(b) == 1:
            R[a] += 0.5
            G[b] += 0.5
        else:
            R[a] += 1
            G[b] += 1

    return np.concatenate([R, G])


def matching(string, milestones, map = Map, refseq = RefSeq):
    complement = {
        'A': 'T',
        'C': 'G',
        'G': 'C',
        'T': 'A'
    }

    string = string.replace('N', 'A')
    string_reverse = string[::-1]
    string_reverse = ''.join(complement[base] for base in string_reverse)

    start1, end1, offset1 = milestones.get_window(string)
    start2, end2, offset2 = milestones.get_window(string_reverse)
    # print("start,offsets", start1, start2, end1, end2, offset1, offset2)
    
    end1, end2 = end1 + 1, end2 + 1
    string = string[:offset1]
    string_reverse = string_reverse[:offset2]

    matches = []
    for i in range(start1, end1):
        index = map[i] - offset1
        if index > -1 and valid(index, string, refseq):
            matches.append(index)

    for i in range(start2, end2):
        index = map[i] - offset2
        if index > -1 and valid(index, string_reverse, refseq):
            matches.append(index)
    
    return matches

milestone = Milestones(LastCol, k=50)
ExonMatchCounts = np.zeros(12)
length = len(Reads)

# matching('GAGGACAGCACCCAGTCCAGCATCTTCACCTACACCAACAGCAACTCCACCAGAGGTGAGCCAGCAGGCCCGTGGAGGCTGGGTGGCTGCACTGGGGGCCA', milestone, Map, RefSeq)
for i in tqdm(range(length)):
    read = Reads[i]
    positions = matching(read, milestone, Map, RefSeq)
    ExonMatchCounts += identify_exon(positions, *rg_positions)
# [135.  74.  78. 145. 302. 358. 135. 186.  75. 123. 337. 358.]

def ComputeProb(ExonMatchCounts):
    red_exon_cnt = [int(c) for c in ExonMatchCounts[:6]]
    green_exon_cnt = [int(c) for c in ExonMatchCounts[6:]]
    total_cnt = [red_exon_cnt[i] + green_exon_cnt[i] for i in range(6)]

    configs = [
        ([1/3, 1/3, 1/3, 1/3], [2/3, 2/3, 2/3, 2/3]),
        ([1/2, 1/2, 0, 0], [1/2, 1/2, 1.0, 1.0]),
        ([1/4, 1/4, 1/2, 1/2], [3/4, 3/4, 1/2, 1/2]),
        ([1/4, 1/4, 1/4, 1/2], [3/4, 3/4, 3/4, 1/2])
    ]

    probabilities = []
    for config_red, config_green in configs:
        P = 1
        for i in range(4):
            P *= comb(total_cnt[i+1], red_exon_cnt[i+1]) * pow(config_red[i], red_exon_cnt[i+1]) * pow(config_green[i], green_exon_cnt[i+1])
        probabilities.append(P)

    return probabilities


ListProb = ComputeProb(ExonMatchCounts)
MostLikely = np.argmax(ListProb) + 1
print(f"Configuration {MostLikely} is the best match")
print(ListProb, ExonMatchCounts)