import re
import stanza
import numpy as np
import itertools
from itertools import product
from seq2seq.relation_infusion.constants import MAX_RELATIVE_DIST
from seq2seq.relation_infusion.get_relation2id_dict import get_relation2id_dict
from seq2seq.relation_infusion.get_relation_name import Relation_Name
from transformers import T5Tokenizer, T5ForConditionalGeneration

RELATION2ID_DICT, ID2RELATION_DICT, relation_num = get_relation2id_dict()

class Relation_Id(object):
    def __init__(self):
        self.nlp_tokenize = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse', 
                                    tokenize_pretokenized = False, use_gpu=True)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')



    def get_word2ids(self):
        question_input = self.question_input
        question_tok = self.question_tok
        schema = self.schema
    
        question2id = {tok:i for i,tok in enumerate(question_tok)}

        phrase2tokid = {}
        begin = 0
        question_list = question_tok + schema.split()
        
        ent2id, ent_items = {}, ' '.join(self.ents).split()
        rel2id, rel_items = {}, ' '.join(self.rels).split()
        
        for it_index,item in enumerate(question_list):
            tok_id = self.tokenizer(item,padding=True,max_length=512, truncation=True)['input_ids'][:-1]
            if len(tok_id)>0 and tok_id[0] == 3 :
                #index = question_input.find(item)

                question_test = ' '+ question_input + ' ' 
                if question_test.find(f' {item} ')==-1:
                    tok_id = tok_id[1:]  

            item_name = item
            i = 0
            while item_name in phrase2tokid:
                item_name = item + f'_{i}'
                i+=1
                
            phrase2tokid[item_name] = [begin, begin+len(tok_id)]
            begin += len(tok_id)
            
            if it_index>= len(question_tok) and item in ent_items:
                if '<extra_id' in item or bool(re.match('[pq][0-9]', item)):
                    ent2id[item_name] = len(ent2id)
                else:
                    label_type = re.findall('<extra_id_[0-9][0-9]>',' '.join(question_list[:it_index]))[-1]
                    if label_type == '<extra_id_53>':
                        ent2id[item_name] = len(ent2id)
                
            if it_index >= len(question_tok) and item in rel_items:
                if '<extra_id' in item or bool(re.match('[pq][0-9]', item)):
                    rel2id[item_name] = len(rel2id)
                else:
                    label_type = re.findall('<extra_id_[0-9][0-9]>',' '.join(question_list[:it_index]))[-1]
                    if label_type != '<extra_id_53>':
                        rel2id[item_name] = len(rel2id)
    
        return begin, phrase2tokid, question2id, ent2id, rel2id



    def get_relation_id(self, data):
        self.raw_question = data['raw_question']
        self.gold_query = data['sparql_process']
        self.question_input = data['question_process']
        
        if isinstance(data['question_toks'], list):
            self.question_tok = data['question_toks'] 
        elif isinstance(data['question_toks'], str):
            self.question_tok = eval(data['question_toks'])
        self.schema = data['schema']

        rel_name = Relation_Name(self.gold_query, self.question_input, self.question_tok, self.raw_question, self.nlp_tokenize)
        
        q_q_depend = rel_name.get_qq_dependency()
        q_q_dist = rel_name.get_qq_distance()
        q_e_mat, e_q_mat = rel_name.get_qer_relation('entity')
        q_r_mat, r_q_mat = rel_name.get_qer_relation('relation')
        e_r_mat, r_e_mat = rel_name.get_er_relation()
        e_e_mat = rel_name.get_ee_rr_relation('entity')
        r_r_mat = rel_name.get_ee_rr_relation('relation')

        split_list = self.question_input.split(' <extra_id_59> ')
        question = split_list[0]
        self.ents, self.rels = [], []
        for item in split_list[1:]:
            #if bool(re.match(r'q[0-9]+',item.split()[1])):
            if item.split()[0] == '<extra_id_53>':
                self.ents.append(item)
            else: self.rels.append(item)

        begin, phrase2tokid, question2id, ent2id, rel2id = self.get_word2ids()
        #tok_len = len(self.tokenizer(self.question_input,padding=True,max_length=512, truncation=True)['input_ids'])
        tok_len = len(data['input_ids'])
        assert tok_len == begin+1

        relation_mat = np.array([[0] * tok_len for _ in range(tok_len)], dtype=np.float64)

        # add single E/R relation
        for (ent1,ent2) in itertools.product(ent2id.keys(),ent2id.keys()) :
            relation_name = e_e_mat[ent2id[ent1], ent2id[ent2]]
            range1 = phrase2tokid[ent1]
            range2 = phrase2tokid[ent2]
            relation_mat[range1[0]:range1[1], range2[0]:range2[1]] = RELATION2ID_DICT[relation_name]
            
        for (rel1,rel2) in itertools.product(rel2id.keys(),rel2id.keys()) :
            relation_name = r_r_mat[rel2id[rel1], rel2id[rel2]]
            range1 = phrase2tokid[rel1]
            range2 = phrase2tokid[rel2]
            relation_mat[range1[0]:range1[1], range2[0]:range2[1]] = RELATION2ID_DICT[relation_name]

        # add Q-Q relation
        for (q1,q2) in itertools.product(question2id.keys(),question2id.keys()) :
            relation_name = q_q_dist[question2id[q1], question2id[q2]]
            range1 = phrase2tokid[q1]
            range2 = phrase2tokid[q2]
            relation_mat[range1[0]:range1[1], range2[0]:range2[1]] = RELATION2ID_DICT[relation_name]

        for (q1,q2) in itertools.product(question2id.keys(),question2id.keys()) :
            relation_name = q_q_depend[question2id[q1], question2id[q2]]
            range1 = phrase2tokid[q1]
            range2 = phrase2tokid[q2]
            relation_mat[range1[0]:range1[1], range2[0]:range2[1]] = RELATION2ID_DICT[relation_name]

        # add E-Q/Q-E relation
        for (e,q) in itertools.product(ent2id.keys(),question2id.keys()) :
            relation_name = e_q_mat[ent2id[e], question2id[q]]
            range1 = phrase2tokid[e]
            range2 = phrase2tokid[q]
            relation_mat[range1[0]:range1[1], range2[0]:range2[1]] = RELATION2ID_DICT[relation_name]

        for (q,e) in itertools.product(question2id.keys(),ent2id.keys()) :
            relation_name = q_e_mat[question2id[q], ent2id[e]]
            range1 = phrase2tokid[q]
            range2 = phrase2tokid[e]         
            relation_mat[range1[0]:range1[1], range2[0]:range2[1]] = RELATION2ID_DICT[relation_name]

        # add R-Q/Q-R relation
        for (r,q) in itertools.product(rel2id.keys(),question2id.keys()) :
            relation_name = r_q_mat[rel2id[r], question2id[q]]
            range1 = phrase2tokid[r]
            range2 = phrase2tokid[q]
            relation_mat[range1[0]:range1[1], range2[0]:range2[1]] = RELATION2ID_DICT[relation_name]

        for (q,r) in itertools.product(question2id.keys(),rel2id.keys()) :
            relation_name = q_r_mat[question2id[q], rel2id[r]]
            range1 = phrase2tokid[q]
            range2 = phrase2tokid[r]          
            relation_mat[range1[0]:range1[1], range2[0]:range2[1]] = RELATION2ID_DICT[relation_name]

        # add R-E/E-R relation
        for (e,r) in itertools.product(ent2id.keys(),rel2id.keys()) :
            relation_name = e_r_mat[ent2id[e], rel2id[r]]
            range1 = phrase2tokid[e]
            range2 = phrase2tokid[r]           
            relation_mat[range1[0]:range1[1], range2[0]:range2[1]] = RELATION2ID_DICT[relation_name]

        for (r,e) in itertools.product(rel2id.keys(),ent2id.keys()) :
            relation_name = r_e_mat[rel2id[r], ent2id[e]]
            range1 = phrase2tokid[r]
            range2 = phrase2tokid[e]           
            relation_mat[range1[0]:range1[1], range2[0]:range2[1]] = RELATION2ID_DICT[relation_name]

        return relation_mat