import sys, os, logging, pickle
import numpy as np
import random
from tqdm import tqdm
from Bio import SeqIO
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

def get_origin_prot_id(prot_id):
    prot_id_origin = prot_id
    if '_range' in prot_id: prot_id_origin = prot_id[:prot_id.find('_range')]
    elif '_rev' in prot_id: prot_id_origin = prot_id[:prot_id.find('_rev')]
    elif '_doub_shuf0' in prot_id: prot_id_origin = prot_id[:prot_id.find('_doub_shuf0')]
    elif '_doub_shuf1' in prot_id: prot_id_origin = prot_id[:prot_id.find('_doub_shuf1')]
    elif '_decoy' in prot_id: prot_id_origin = prot_id[:prot_id.find('_decoy')]
    elif '_trunc' in prot_id: prot_id_origin = prot_id[:prot_id.find('_trunc')]
    elif '_doub' in prot_id: prot_id_origin = prot_id[:prot_id.find('_doub')]
    elif '_mut' in prot_id: prot_id_origin = prot_id[:prot_id.find('_mut')]
    elif '_shuf' in prot_id: prot_id_origin = prot_id[:prot_id.find('_shuf')]
    elif '_mkv2' in prot_id: prot_id_origin = prot_id[:prot_id.find('_mkv2')]
    elif '_mkv1' in prot_id: prot_id_origin = prot_id[:prot_id.find('_mkv1')]
    
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
    curr_search_list = []
    score_zone = False
    res = []
    with open(url, 'r') as fp:
        for cnt, line in enumerate(fp):
            line = line.rstrip().rstrip("\n")
            if len(line) <= 0: continue

            if line.startswith("Results from"):
                curr_iter = int(line.split()[-1])
                print('curr_iter={}'.format(curr_iter))
                assert curr_iter <= 1

            elif line.startswith("Query="):
                if len(curr_query_id) > 0: 
                    res += [[curr_query_id, sid, raw_bit] for sid, _, raw_bit in curr_search_list]
                curr_query_id = line.split()[1]
                curr_search_list = []

            elif line.startswith("Sequences"): score_zone = True
            elif line.startswith("Lambda"): score_zone = False
            else:
                if not score_zone: continue
                data = line.split()
                sid = data[0]
                evalue = float(data[-1])
                raw_bit = float(data[-2])
                curr_search_list.append((sid, evalue, raw_bit))  # sid, Evalue, Rawbit

        if len(curr_query_id) > 0:
            res += [[curr_query_id, sid, raw_bit] for sid, _, raw_bit in curr_search_list]
    return res

ParserFunctionMap = {
    "plm": parse_plm_data,
    "tmvec": parse_tmvec_data,
    "dhr": parse_dhr_data,
    "dctdomain": parse_dctdomain_data,
    "blastp": parse_blast_data
}


def parse_result_by_type(prev_data_url, data_type, n_subDB):
    assert data_type in ParserFunctionMap.keys()

    fasta_url = "../data/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
    fa_id_label_map = get_seq_label_map(fasta_url)
    print('fa_id_label_map={}'.format(len(fa_id_label_map)))
    
    subDB_url = f'../data/astral_s{n_subDB}.fa'
    subDB_key = parse_astral_key(subDB_url)
    
    save_dir = '../data'
    
    fname_homo = os.path.basename(prev_data_url).replace('.txt', '_homo.pkl')
    fname_nonhomo = os.path.basename(prev_data_url).replace('.txt', '_nonhomo.pkl')
    fname_homo = fname_homo.replace('astral', f'astrals{n_subDB}')
    fname_nonhomo = fname_nonhomo.replace('astral', f'astrals{n_subDB}')
    saved_url_homo = os.path.join(save_dir, fname_homo)
    saved_url_nonhomo = os.path.join(save_dir, fname_nonhomo)

    if os.path.exists(saved_url_homo) and os.path.exists(saved_url_nonhomo):
        file1 = open(saved_url_homo, 'rb')
        sid_search_map_homo = pickle.load(file1)
        file1.close()

        file2 = open(saved_url_nonhomo, 'rb')
        sid_search_map_nonhomo = pickle.load(file2)
        file2.close()
    else:
        assert os.path.exists(prev_data_url)
        sid_search_map_homo = {}
        sid_search_map_nonhomo = {}

        parse_func = ParserFunctionMap.get(data_type)
        
        for curr_id, tgt_id, score in parse_func(prev_data_url):
            # if get_origin_prot_id(curr_id) not in set(subDB_key): continue ## check if curr_id is in subDB
            # if get_origin_prot_id(tgt_id) not in set(subDB_key): continue ## check if tgt_id is in subDB
            
            curr_id_origin = get_origin_prot_id(curr_id)
            if curr_id not in sid_search_map_nonhomo: sid_search_map_nonhomo[curr_id] = []
            if curr_id not in sid_search_map_homo: sid_search_map_homo[curr_id] = []

            is_homolog = -1
            if curr_id_origin in fa_id_label_map:
                src_scop_fold, src_scop_supf, src_scop_fam = fa_id_label_map[curr_id_origin]
                
                tgt_id_origin = get_origin_prot_id(tgt_id)
                if tgt_id_origin not in fa_id_label_map: print("warning", curr_id, tgt_id, score)
                tgt_scop_fold, tgt_scop_supf, tgt_scop_fam = fa_id_label_map[tgt_id_origin]
                
                is_homolog = check_homolog(src_scop_fold, src_scop_supf, tgt_scop_fold, tgt_scop_supf)

            if is_homolog > 0: sid_search_map_homo[curr_id].append((curr_id_origin, tgt_id, score))
            else: sid_search_map_nonhomo[curr_id].append(score)

        with open(saved_url_homo, 'wb') as fp:
            pickle.dump(sid_search_map_homo, fp)

        with open(saved_url_nonhomo, 'wb') as fp:
            pickle.dump(sid_search_map_nonhomo, fp)

    print('sid_search_map_homo={}\tsid_search_map_nonhomo={}'.format(len(sid_search_map_homo), len(sid_search_map_nonhomo)))
    return sid_search_map_homo, sid_search_map_nonhomo

def set_randseed(seed=0):
    np.random.seed(seed)
    random.seed(seed)


def parse_astral_key(fasta_url="../data/astral.fa"):
    astral_key = []
    for record in SeqIO.parse(fasta_url, "fasta"):
        seq_id = record.id
        astral_key.append(seq_id)
    return astral_key

def main(args):
    prev_data_path = '/home/yfnyang/2025_yifan_plmfool/data'
    save_dir = '../data'
    
    if args.data_type == 'blastp':
        prev_data_url = f'{prev_data_path}/results_blastp_astral_{args.decoy_type}_to_astral_blosum62_Q11R1.txt'
    else:
        prev_data_url = f"{prev_data_path}/result_{args.data_type}_astral_{args.decoy_type}_to_astral.txt"
    assert os.path.exists(prev_data_url), f'{prev_data_url} not exist'
    
    sid_search_map_homo, sid_search_map_nonhomo = parse_result_by_type(prev_data_url, args.data_type, args.n_subDB)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract subsample results")
    parser.add_argument("--data_type", type=str, help="PLMs method")
    parser.add_argument("--decoy_type", type=str, help="Decoy type")
    parser.add_argument("--n_subDB", type=int, help="number of subsample in database")
    
    args = parser.parse_args()
    main(args)

# Command
# python utils/extract_prev_result.py --n_subDB 500 --data_type blastp --decoy_type shuf
# python utils/extract_prev_result.py --n_subDB 1000 --data_type blastp --decoy_type shuf