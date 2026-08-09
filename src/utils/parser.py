import sys, os, logging, pickle
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
import argparse
import json
from pathlib import Path
import pandas as pd
import random
from collections import defaultdict
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

def get_origin_prot_id(prot_id):
    prot_id_origin = prot_id


    # NEW: handle window-decoy ids like "d4hswa_.1.10" -> "d4hswa_"
    # (two dot-separated integer suffixes)
    m = re.match(r"^(.+)\.(\d+)\.(\d+)$", prot_id)
    if m:
        return m.group(1)

    if '_range' in prot_id: prot_id_origin = prot_id[:prot_id.find('_range')]
    elif '_rev' in prot_id: prot_id_origin = prot_id[:prot_id.find('_rev')]
    elif '_doub_shuf0' in prot_id: prot_id_origin = prot_id[:prot_id.find('_doub_shuf0')]
    elif '_doub_shuf1' in prot_id: prot_id_origin = prot_id[:prot_id.find('_doub_shuf1')]
    elif '_decoy' in prot_id: prot_id_origin = prot_id[:prot_id.find('_decoy')]
    elif '_trunc' in prot_id: prot_id_origin = prot_id[:prot_id.find('_trunc')]
    elif '_doub' in prot_id: prot_id_origin = prot_id[:prot_id.find('_doub')]
    elif '_mut' in prot_id: prot_id_origin = prot_id[:prot_id.find('_mut')]
    elif '_shuf' in prot_id: prot_id_origin = prot_id[:prot_id.find('_shuf')]
    elif '_mkv' in prot_id: prot_id_origin = prot_id[:prot_id.find('_mkv')]
    elif '_mkv2' in prot_id: prot_id_origin = prot_id[:prot_id.find('_mkv2')]
    elif '_PGmsk25pct' in prot_id: prot_id_origin = prot_id[:prot_id.find('_PGmsk25pct')]
    elif '_dplm' in prot_id: prot_id_origin = prot_id[:prot_id.find('_dplm')]
    elif '_adp' in prot_id: prot_id_origin = prot_id[:prot_id.find('_adp')]
    elif '_PGlen' in prot_id: prot_id_origin = prot_id[:prot_id.find('_PGlen')]
    elif '_copy' in prot_id: prot_id_origin = prot_id[:prot_id.find('_copy')]
    elif '_dup' in prot_id: prot_id_origin = prot_id[:prot_id.find('_dup')]

    
    return prot_id_origin

def get_seq_label_map(fasta_url):
    records = list(SeqIO.parse(fasta_url, "fasta"))

    fa_id_label_map = {}
    for record in tqdm(records):
        fa_id = record.id
        desc = record.description
        scop_label = desc.split(' ')[1]
        arr = scop_label.split('.')
        assert len(arr) == 4

        scop_fold = '.'.join(arr[:2])
        scop_supf = '.'.join(arr[:3])
        scop_fam = '.'.join(arr[:4])

        assert fa_id not in fa_id_label_map
        fa_id_label_map[fa_id] = (scop_fold, scop_supf, scop_fam)

    return fa_id_label_map

def check_homolog(src_scop_fold, src_scop_supf, tgt_scop_fold, tgt_scop_supf):
    if src_scop_supf == tgt_scop_supf: return 1
    elif src_scop_fold != tgt_scop_fold: return -1
    else: return 0

def check_muti_level_homolog(src_labels, tgt_labels):
    '''
        -1: non-homolog
        0: same fold, different superfamily
        1: same superfamily, different family
        2: same family
    '''
    src_scop_fold, src_scop_supf, src_scop_fam = src_labels
    tgt_scop_fold, tgt_scop_supf, tgt_scop_fam = tgt_labels
    
    if src_scop_fold != tgt_scop_fold:
        category = -1              
    elif src_scop_supf != tgt_scop_supf:
        category = 0
    elif src_scop_fam != tgt_scop_fam:
        category = 1
    else:
        category = 2
    return category 
    
def parse_plm_data(url):
    res = []
    with open(url, 'r') as fp:
        for cnt, line in enumerate(fp):
            line = line.rstrip().rstrip("\n")
            if len(line) <= 0: continue
            arr = line.split()
            res.append([arr[0], arr[1], float(arr[-1])])
    return res

def parse_tmvec_data(url):
    res = []
    with open(url, 'r') as fp:
        for cnt, line in enumerate(fp):
            line = line.rstrip().rstrip("\n")
            if len(line) <= 0: continue
            if line.startswith('query_id'): continue
            
            arr = line.split()
            res.append([arr[0], arr[2], float(arr[-1])])
    return res

def parse_dhr_data(url):
    res = []
    with open(url, 'r') as fp:
        for cnt, line in enumerate(fp):
            line = line.rstrip().rstrip("\n")
            if len(line) <= 0: continue
            
            arr = line.split(',')
            res.append([arr[0], arr[1], float(arr[-1])])
    return res

def parse_dctdomain_data(url):
    res = []
    with open(url, 'r') as fp:
        for cnt, line in enumerate(fp):
            line = line.rstrip().rstrip("\n")
            if len(line) <= 0: continue
            if not line.startswith('Query'): continue

            arr = line.split(':')
            assert len(arr) == 4

            curr_id = arr[1].strip().split()[0]
            tgt_id = arr[2].strip().split()[0]
            score = float(arr[3].strip())
            res.append([curr_id, tgt_id, score])
    return res

def parse_blast_data(url):
    curr_query_id = ''
    score_zone = False
    with open(url, 'r') as fp:
        for cnt, line in enumerate(fp):
            line = line.rstrip().rstrip("\n")
            if len(line) <= 0: continue

            if line.startswith("Results from"):
                curr_iter = int(line.split()[-1])
                print('curr_iter={}'.format(curr_iter))
                assert curr_iter <= 1

            elif line.startswith("Query="):
                curr_query_id = line.split()[1]

            elif line.startswith("Sequences"): score_zone = True
            elif line.startswith("Lambda"): score_zone = False
            else:
                if not score_zone: continue
                data = line.split()
                sid = data[0]
                evalue = float(data[-1])
                raw_bit = float(data[-2])
                
                yield (curr_query_id, sid, raw_bit, evalue)


ParserFunctionMap = {
    "plm": parse_plm_data,
    "tmvec": parse_tmvec_data,
    "dhr": parse_dhr_data,
    "dctdomain": parse_dctdomain_data,
    "blastp": parse_blast_data
}


def parse_result_by_type(input_path, save_path, data_type):
    assert data_type in ParserFunctionMap.keys()

    fasta_url = "/home/yfnyang/2025_yifan_plmfool/data/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
    fa_id_label_map = get_seq_label_map(fasta_url)
    print('fa_id_label_map={}'.format(len(fa_id_label_map)))
    
    fname = Path(input_path).name
    save_url = os.path.join(save_path, f'{fname}_hit{max_hits}.pkl')

    if os.path.exists(save_url):
        with open(save_url, 'r') as f:
            qt_score = pickle.load(f)                        
    else:
        assert os.path.exists(input_path)
        qt_score_homo={}
        qt_score_nonhomo={}
        
        parse_func = ParserFunctionMap.get(data_type)
        
        for file in tqdm(os.listdir(input_path), desc='Sparsing batch...'):
            if not ('.out' in file or '.txt' in file): continue
            
            url = os.path.join(input_path, file)
            print('url={}'.format(url))

            for curr_id, tgt_id, score in tqdm(parse_func(url), desc='Sparsing file...'):
                curr_id_origin = get_origin_prot_id(curr_id)
                if curr_id not in qt_score_homo: qt_score_homo[curr_id] = []
                if curr_id not in qt_score_nonhomo: qt_score_nonhomo[curr_id] = []
                
                curr_id_scores = qt_score_homo[curr_id] + qt_score_nonhomo[curr_id]
                if len(curr_id_scores) >= max_hits: continue

                is_homolog = -1
                if curr_id_origin in fa_id_label_map:
                    src_scop_fold, src_scop_supf, src_scop_fam = fa_id_label_map[curr_id_origin]
                    
                    tgt_id_origin = get_origin_prot_id(tgt_id)
                    if tgt_id_origin not in fa_id_label_map: print("warning", curr_id, tgt_id, score)
                    tgt_scop_fold, tgt_scop_supf, tgt_scop_fam = fa_id_label_map[tgt_id_origin]
                    
                    is_homolog = check_homolog(src_scop_fold, src_scop_supf, tgt_scop_fold, tgt_scop_supf)

                if is_homolog > 0: 
                    qt_score_homo[curr_id].append((curr_id_origin, tgt_id, score))
                    if data_type == 'blastp': continue
                    curr_id_scores = [s for _, _, s in qt_score_homo[curr_id]]
                    if data_type == 'dhr':
                        assert score == np.max(np.array(curr_id_scores)), f'{curr_id}, {qt_score_homo[curr_id]}, {score}, {np.max(np.array(curr_id_scores))}'
                    else:
                        assert score == np.min(np.array(curr_id_scores)), f'{curr_id}, {qt_score_homo[curr_id]}, {score}, {np.min(np.array(curr_id_scores))}'
                else: 
                    qt_score_nonhomo[curr_id].append(score)
                    if data_type == 'blastp': continue
                    curr_id_scores = qt_score_nonhomo[curr_id]
                    if data_type == 'dhr':
                        assert score == np.max(np.array(curr_id_scores)), f'{curr_id}, {qt_score_nonhomo[curr_id]}, {score}, {np.max(np.array(curr_id_scores))}'
                    else:
                        assert score == np.min(np.array(curr_id_scores)), f'{curr_id}, {qt_score_nonhomo[curr_id]}, {score}, {np.min(np.array(curr_id_scores))}'                    

        qt_score = {'homo': qt_score_homo, 'nonhomo': qt_score_nonhomo}

        with open(save_url, 'wb') as fp:
            pickle.dump(qt_score, fp)
                    
        # ## To save storage
        # pd_data = {}
        # query_set = sorted(list(query_set))
        # tgt_set = sorted(list(tgt_set))
        # tgt_map = {key:i for i, key in enumerate(tgt_set)}
        # fp_type_map = {
        #     "plm": np.float64,
        #     "tmvec": np.float32,
        #     "dhr": np.float64,
        #     "dctdomain": np.float16,
        #     "blastp": np.float16            
        # }
        
        # for key in query_set:
        #     query = np.full(len(tgt_set), np.nan, dtype=fp_type_map[data_type])
        #     for type in ["homo", "nonhomo"]:
        #         if key in qt_score[type]:
        #             for tgt_line in qt_score[type][key]:
        #                 tgt_idx = tgt_map[tgt_line[1]]
        #                 query[tgt_idx] = float(tgt_line[2])
        #     pd_data[key] = query
        #     if data_type != "blastp":
        #         assert not np.isnan(query).any(), f"Error: query={key} does not contain all targets in DB!"
        # data = pd.DataFrame(pd_data, index=tgt_set)
        
        # with open(save_url.replace(".json", ".pkl"), 'wb') as fp:
        #     pickle.dump(data, fp)
            
    print('sid_search_map_homo={}\tsid_search_map_nonhomo={}'.format(len(qt_score_homo), len(qt_score_nonhomo)))
    return qt_score_homo, qt_score_nonhomo

def parse_score_by_type(input_path, save_path, data_type):
    assert data_type in ParserFunctionMap.keys()
    
    fname = Path(input_path).stem
    save_url = os.path.join(save_path, f'{fname}.pkl')
    print(f"saved to {save_url}")

    if os.path.exists(save_url):
        print(f"File already exists: {save_url}, loading...")
        with open(save_url, 'rb') as f:
            qt_score = pickle.load(f)  
            
        qt_score_homo = qt_score['homo']
        qt_score_nonhomo = qt_score['nonhomo']
        
    else:
        assert os.path.exists(input_path)
        with open(input_path, 'r') as f:
            parsed_result = f.readlines()
            
        qt_score_homo={}
        qt_score_nonhomo={}
        
        for row in tqdm(parsed_result[1:], desc='Sparsing batch...'):
            curr_id, tgt_id, score, homo_type, rank, *rest = row.strip().split('\t')
            score, homo_type, rank = float(score), int(homo_type), int(rank)
            
            curr_id_origin = get_origin_prot_id(curr_id)
            if curr_id not in qt_score_homo: qt_score_homo[curr_id] = []
            if curr_id not in qt_score_nonhomo: qt_score_nonhomo[curr_id] = []     
                   
            if homo_type > 0: 
                qt_score_homo[curr_id].append((curr_id_origin, tgt_id, score))
            else:
                qt_score_nonhomo[curr_id].append(score)
                
        qt_score = {'homo': qt_score_homo, 'nonhomo': qt_score_nonhomo}

        with open(save_url, 'wb') as fp:
            pickle.dump(qt_score, fp)
            
    print('sid_search_map_homo={}\tsid_search_map_nonhomo={}'.format(len(qt_score_homo), len(qt_score_nonhomo)))
    return qt_score_homo, qt_score_nonhomo

def parse_mutlevel_homo_result_by_type(input_path, save_path, data_type):
    assert data_type in ParserFunctionMap.keys()

    fasta_url = "/home/yfnyang/2025_yifan_plmfool/data/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
    fa_id_label_map = get_seq_label_map(fasta_url)
    print('fa_id_label_map={}'.format(len(fa_id_label_map)))

    fname = Path(input_path).stem
    save_url = os.path.join(save_path, f'{fname}_mutlevel.pkl')
    print(f"saved to {save_url}")

    if os.path.exists(save_url):
        print(f"File already exists: {save_url}, loading...")
        with open(save_url, 'rb') as f:
            qt_result = pickle.load(f)  
                    
    else:
        assert os.path.exists(input_path)
        with open(input_path, 'r') as f:
            parsed_result = f.readlines()
            
        qt_result = {
            -1: defaultdict(dict),
            0: defaultdict(dict),
            1: defaultdict(dict),
            2: defaultdict(dict),
        }
        
        for row in tqdm(parsed_result[1:], desc='Sparsing batch...'):
            curr_id, tgt_id, score, homo_type, rank, *rest = row.strip().split('\t')
            score, homo_type, rank = float(score), int(homo_type), int(rank)
            qt_result[homo_type][curr_id][tgt_id] = (score, rank)
        
        with open(save_url, 'wb') as fp:
            pickle.dump(qt_result, fp)
    
    print('fam_homo={}\tsupf_homo={}\tfold_homo={}\tnonhomo={}'.format(
        len(qt_result[2]), len(qt_result[1]), len(qt_result[0]), len(qt_result[-1])
        ))
    
    return qt_result

def parse_search_result_by_type(input_path, save_path, data_type, fasta_url):
    assert data_type in ParserFunctionMap.keys()

    fa_id_label_map = get_seq_label_map(fasta_url)
    print('fa_id_label_map={}'.format(len(fa_id_label_map)))
    
    fname = Path(input_path).name
    save_url = os.path.join(save_path, f'parsed_result/{fname}_hit{max_hits}.txt')

    if os.path.exists(save_url):
        with open(save_url, 'r') as f:
            qt_score = f.readlines()
        print(f"Parsed results already exists (line={len(qt_score)}): {save_url}!")    
        print(f"Sample data: \n{qt_score[0]}\n{qt_score[1]}") 
                     
    else:
        assert os.path.exists(input_path)
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        
        parse_func = ParserFunctionMap.get(data_type)
        
        with open(save_url, 'w') as f:
            if data_type == 'blastp':
                f.write(f"qid\ttid\tscore\thomo_type\trank\tevalue\n")
            else:
                f.write(f"qid\ttid\tscore\thomo_type\trank\n")
        
        total_line = 0
        for file in tqdm(os.listdir(input_path), desc='Sparsing batch...'):
            if not ('.out' in file or '.txt' in file): continue
            
            url = os.path.join(input_path, file)
            print('url={}'.format(url))

            query_id_flag = None
            rank = None
            prev_score = None # for sanity check
            for row in tqdm(parse_func(url), desc='Sparsing file...'):
                curr_id, tgt_id, score, *rest = row
                score = float(score)
                
                # get ranking
                if query_id_flag != curr_id:
                    if rank is not None and data_type not in ['dctdomain', 'blastp']: assert rank >= max_hits
                    query_id_flag = curr_id
                    rank = 1
                    prev_score = score
                else:
                    rank += 1
                    if data_type != 'blastp':
                        assert score >= prev_score if data_type == 'dhr' else score <= prev_score
                    prev_score = score
                    
                if rank > max_hits: continue
                
                # get homo type
                curr_id_origin = get_origin_prot_id(curr_id)
                assert curr_id_origin in fa_id_label_map, f"Error: {curr_id_origin} not found in label map!"
                
                tgt_id_origin = get_origin_prot_id(tgt_id)
                assert tgt_id_origin in fa_id_label_map, f"Error: {tgt_id_origin} not found in label map!"
                
                src_labels = fa_id_label_map[curr_id_origin]
                tgt_labels = fa_id_label_map[tgt_id_origin]
                homo_type = check_muti_level_homolog(src_labels, tgt_labels)
        
                # save results
                with open(save_url, 'a') as f:
                    if rest:
                        f.write(f"{curr_id}\t{tgt_id}\t{score}\t{homo_type}\t{rank}\t{rest[0]}\n")
                    else:
                        f.write(f"{curr_id}\t{tgt_id}\t{score}\t{homo_type}\t{rank}\n")
                total_line += 1

        print(f"Parsed results (line={total_line}) saved to: {save_url}.")   

def set_randseed(seed=0):
    np.random.seed(seed)
    random.seed(seed)

def parse_astral_key(fasta_url="../data/astral.fa"):
    astral_key = []
    for record in SeqIO.parse(fasta_url, "fasta"):
        seq_id = record.id
        astral_key.append(seq_id)
    return astral_key

def parse_sequence_lenth(fasta_url):
    length_dict = {}

    for record in SeqIO.parse(fasta_url, "fasta"):
        seq_id = record.id
        seq_length = len(record.seq)
        length_dict[seq_id] = seq_length
    return length_dict

def load_sequence(fasta_url):
    astral_dic = {}
    for record in SeqIO.parse(fasta_url, "fasta"):
        astral_dic[record.id] = str(record.seq)
    return astral_dic
    
max_hits=None
def main(args):
    global max_hits
    max_hits = args.max_hits
    
    if args.basedb == 'SCOPe40':
        fasta_url = "/home/yfnyang/2025_yifan_plmfool/data/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
    elif args.basedb == 'SCOPe95':
        fasta_url = "/home/yfnyang/2025_yifan_plmfool/data/astral-scopedom-seqres-gd-sel-gs-bib-95-2.08.fa"
    else:
        raise ValueError(f"Unsupported basedb: {args.basedb}")
    # parse_result_by_type(args.input_path, args.save_path, args.data_type)
    parse_search_result_by_type(args.input_path, args.save_path, args.data_type, fasta_url)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser for PLMs")
    parser.add_argument("--data_type", type=str, help="PLMs method")
    parser.add_argument("--input_path", type=str, help="Input path to be parsed")
    parser.add_argument("--save_path", type=str, help="Path to save parsed result")
    parser.add_argument("--max_hits", type=int, default=1000, help="Keep how many hits while parsing")
    parser.add_argument("--basedb", type=str, help="SCOPe40/SCOPe95")

    args = parser.parse_args()
    main(args)

# command
# python utils/parser.py --data_type blastp --input_path ../data/temp/result_blastp_astral_blosum62_Q11R1 --save_path ../data