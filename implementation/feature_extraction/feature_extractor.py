from collections import OrderedDict
import numpy as np
import wordninja
from math import log
import math

def find_length(name):
    length = len(name)
    return length

def find_max_digit_sequence(name):
    digits = "0123456789"
    max_digit_sequence = 0
    current_digit_sequence = 0
    for character in name:
        if character in digits:
            current_digit_sequence += 1
        else:
            if current_digit_sequence > max_digit_sequence:
                max_digit_sequence = current_digit_sequence
            current_digit_sequence = 0
    if current_digit_sequence > max_digit_sequence:
        max_digit_sequence = current_digit_sequence
    return max_digit_sequence

def find_max_string_sequence(name):
    chars = "abcdefghijklmnopqrstuvwxyz"
    max_string_sequence = 0
    current_string_sequence = 0
    for character in name:
        if character in chars:
            current_string_sequence += 1
        else:
            if current_string_sequence > max_string_sequence:
                max_string_sequence = current_string_sequence
            current_string_sequence = 0
    if current_string_sequence > max_string_sequence:
        max_string_sequence = current_string_sequence
    return max_string_sequence

def find_individual_string_char_frequency(name):
	chars = "abcdefghijklmnopqrstuvwxyz0123456789"
	freq_dict = OrderedDict()
	for char in chars:
		freq_dict[char] = 0
	for char in name:
		if char in chars:
			freq_dict[char] += 1
	csv_part = ""
	for char in freq_dict.keys():
		csv_part = csv_part + str(freq_dict[char]) + ","
	csv_part = csv_part[0:-1]
	return csv_part

def find_special_char_frequency(name):
    special_chars = "-_."
    special_char_freq = 0
    for char in name:
        if char in special_chars:
            special_char_freq += 1
    return special_char_freq

def find_ratio_special_char(name, special_char_freq):
    name_length = len(name)
    ratio = special_char_freq / name_length
    return ratio

def find_integer_frequency(name):
    chars = "0123456789"
    integer_freq = 0
    for char in name:
        if char in chars:
            integer_freq += 1
    return integer_freq

def find_integer_ratio(name, integer_freq):
    name_length = len(name)
    ratio = integer_freq / name_length
    return ratio

def find_vowel_frequency(name):
    chars = "aeiouy"
    vowels_freq = 0
    for char in name:
        if char in chars:
            vowels_freq += 1
    return vowels_freq 

def find_vowels_ratio(name, vowels_freq):
    name_length = len(name)
    ratio = vowels_freq / name_length
    return ratio

def find_maximum_gap_between_dots(name):
    labels = name.split(".")
    if len(labels) <= 2:
        max_length = 0
    else:
        lengths = [len(label) for label in labels[1:-1]]
        max_length = max(lengths)
    return max_length

def find_ngrams(label, ngrams):
	label_length = len(label)
	tri_grams, quad_grams, five_grams, six_grams, seven_grams = [], [], [], [], []
	if label_length >= 3:
		tri_grams = [label[i : (i + 3)] for i in range(0, label_length - 2)]
	if label_length >= 4:
		quad_grams = [label[i : (i + 4)] for i in range(0, label_length - 3)]
	if label_length >= 5:
		five_grams = [label[i : (i + 5)] for i in range(0, label_length - 4)]
	if label_length >= 6:
		six_grams = [label[i : (i + 6)] for i in range(0, label_length - 5)]
	if label_length >= 7:
		seven_grams = [label[i : (i + 7)] for i in range(0, label_length - 6)]
	for item in tri_grams:
		ngrams.add(item)
	for item in quad_grams:
		ngrams.add(item)
	for item in five_grams:
		ngrams.add(item)
	for item in six_grams:
		ngrams.add(item)
	for item in seven_grams:
		ngrams.add(item)
	return ngrams

def get_label(name):
	label = name.split(".")[0]
	return label

def load_tranco_names():
	filename = "../filter_tranco_names/tranco_top100k.txt"
	names = set()
	fdr = open(filename, "r")
	sequence = 0
	for line in fdr:
		name = line.strip()
		if name not in names:
			names.add(name)
			sequence += 1
			if sequence == 100000:
				break
	return names

def get_ngram_whitelist():
	names = load_tranco_names()
	ngrams = set()
	for name in names:
		label = get_label(name)
		ngrams = find_ngrams(label, ngrams)
	return ngrams

def get_ngrams(name):
	ngrams = set()
	label = get_label(name)
	ngrams = find_ngrams(label, ngrams)
	return ngrams

def compare_whitelist(whitelist, ngrams):
	counter = 0
	for ngram in ngrams:
		if ngram in whitelist:
			counter += 1
	return counter

def find_reputation(name, whitelist):
	ngrams = get_ngrams(name)
	reputation = compare_whitelist(whitelist, ngrams)
	return reputation

def find_words_number(name):
    number_of_words = 0
    labels = name.split(".")
    for label in labels:
        words = wordninja.split(label)
        for word in words:
            if len(word) > 2:
                number_of_words += 1
    return number_of_words

def find_words_mean_length(name):
    lengths = []
    labels = name.split(".")
    for label in labels:
        words = wordninja.split(label)
        for word in words:
            if len(word) > 2:
                lengths.append(len(word))
    lengths = np.array(lengths)
    mean = np.mean(lengths)
    if math.isnan(mean):
        mean = 0
    return mean

def calculate_shannon_entropy(string):
    ent = 0.0
    if len(string) < 2:
        return ent
    size = float(len(string))
    for b in range(128):
        freq = string.count(chr(b))
        if freq > 0:
            freq = float(freq) / size
            ent = ent + freq * log(freq, 2)
    return -ent

def get_shannon_entropy(name):
    entropy = calculate_shannon_entropy(name)
    return entropy

def create_csv_line(f1, f2, f3, f4_group, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, name_original, label, family):
    csv_line = ""
    csv_line += str(f1) + ","
    csv_line += str(f2) + ","
    csv_line += str(f3) + ","
    csv_line += str(f4_group) + ","
    csv_line += str(f5) + ","
    csv_line += str(f6) + ","
    csv_line += str(f7) + ","
    csv_line += str(f8) + ","
    csv_line += str(f9) + ","
    csv_line += str(f10) + ","
    csv_line += str(f11) + ","
    csv_line += str(f12) + ","
    csv_line += str(f13) + ","
    csv_line += str(f14) + ","
    csv_line += str(f15) + ","
    csv_line = csv_line + str(name_original) + "," + str(label) + "," + str(family)
    return csv_line

def export_features(name, name_original, label, family, whitelist):
    length = find_length(name)
    max_digit_sequence = find_max_digit_sequence(name)
    max_string_sequence = find_max_string_sequence(name)
    individual_string_char_frequency = find_individual_string_char_frequency(name)
    special_char_freq = find_special_char_frequency(name)
    ratio_special_char = find_ratio_special_char(name, special_char_freq)
    integer_freq = find_integer_frequency(name)
    ratio_integers = find_integer_ratio(name, integer_freq)
    vowels_freq = find_vowel_frequency(name)
    vowels_ratio = find_vowels_ratio(name, vowels_freq)
    max_gap = find_maximum_gap_between_dots(name)
    reputation = find_reputation(name, whitelist)
    words_number = find_words_number(name)
    words_mean = find_words_mean_length(name)
    shannon_entropy = get_shannon_entropy(name)
    csv_line = create_csv_line(length, max_digit_sequence, max_string_sequence, individual_string_char_frequency, special_char_freq, ratio_special_char, integer_freq, ratio_integers, vowels_freq, vowels_ratio, max_gap, reputation, words_number, words_mean, shannon_entropy, name_original, label, family)
    return csv_line

if __name__ == "__main__":
    whitelist = get_ngram_whitelist()
    filename = "../reduction_and_labeling/labeled_dataset.csv"
    filename_out = "./labeled_dataset_features.csv"
    fdr = open(filename, "r")
    fdw = open(filename_out, "w")

    headers = "Length,Max_DeciDig_Seq,Max_Let_Seq,Freq_A,Freq_B,Freq_C,Freq_D,Freq_E,Freq_F,Freq_G,Freq_H,Freq_I,Freq_J,Freq_K,Freq_L,Freq_M,Freq_N,Freq_O,Freq_P,Freq_Q,Freq_R,Freq_S,Freq_T,Freq_U,Freq_V,Freq_W,Freq_X,Freq_Y,Freq_Z,Freq_0,Freq_1,Freq_2,Freq_3,Freq_4,Freq_5,Freq_6,Freq_7,Freq_8,Freq_9,Spec_Char_Freq,Ratio_Spec_Char,DeciDig_Freq,Ratio_DeciDig,Vowel_Freq,Vowel_Ratio,Max_Gap,Reputation,Words_Freq,Words_Mean,Entropy,Name,Label,Family"
    fdw.write(headers + "\n")

    for line in fdr:
        line = line.strip()
        name, name_original, label, family = line.split(",")
        csv_line = export_features(name, name_original, label, family, whitelist)
        fdw.write(csv_line + "\n")
    fdr.close()
    fdw.close()
