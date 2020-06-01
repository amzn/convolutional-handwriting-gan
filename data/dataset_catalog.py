# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

_DATA_ROOT = 'Datasets/'
datasets = {"RIMEScharH32W16": _DATA_ROOT+'RIMES/h32char16to17/tr',
            "RIMEScharH32": _DATA_ROOT+'RIMES/h32/tr',
            "RIMEScharH32te": _DATA_ROOT+'RIMES/h32/te',
            "RIMEScharH32val": _DATA_ROOT+'RIMES/h32/val',
            "IAMcharH32W16rmPunct": _DATA_ROOT+'IAM/words/h32char16to17/tr_removePunc',
            "IAMcharH32rmPunct": _DATA_ROOT+'IAM/words/h32/tr_removePunc',
            "IAMcharH32rmPunct_te": _DATA_ROOT+'IAM/words/h32/te_removePunc',
            "IAMcharH32rmPunct_val1": _DATA_ROOT+'IAM/words/h32/va1',
            "CVLcharH32W16": _DATA_ROOT+'CVL/h32char16to17/tr',
            "CVLtrH32": _DATA_ROOT+'CVL/h32/train_new_partition',
            "CVLteH32": _DATA_ROOT+'CVL/h32/test_unlabeled',
            }

alphabet_dict = {'IAM': 'alphabetEnglish',
                 'RIMES': 'alphabetFrench',
                 'CVL': 'alphabetEnglish',
                 }

lex_dict = {'IAM': _DATA_ROOT + 'Lexicon/english_words.txt',
            'RIMES': _DATA_ROOT + 'Lexicon/Lexique383.tsv',
            'CVL': _DATA_ROOT + 'Lexicon/english_words.txt'}
