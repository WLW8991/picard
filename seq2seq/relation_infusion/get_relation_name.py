import re
import stanza
import numpy as np
import itertools
from nltk.corpus import stopwords
from itertools import product, combinations
from seq2seq.relation_infusion.constants import MAX_RELATIVE_DIST
from seq2seq.relation_infusion.get_relation2id_dict import get_relation2id_dict
#from transformers import T5Tokenizer, T5ForConditionalGeneration


class Tree(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.children_relation = dict()
        self._size = -1
        self._depth = -1
        self.parents = []


    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)
    
    def add_parent(self, parent):
        self.parents.extend(parent)

    def add_relation(self, child_node, relation):
        self.children_relation[child_node] = relation

    def size(self):
        if getattr(self, '_size') != -1:
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth') != -1:
            return self._depth
        count = 1
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth


class Relation_Name(object):  
    
    def __init__(self, gold_query, question_input, question_toks, raw_question, nlp_tokenize):     
        
        self.nlp_tokenize = nlp_tokenize
        
        # get question ents rels
        split_list = question_input.split(' <extra_id_59> ')
        self.raw_question = raw_question
        self.ents, self.rels = [], []

        for item in split_list[1:]:
            #if bool(re.match(r'q[0-9]+',item.split()[1])):
            if item.split()[0] == '<extra_id_53>':
                self.ents.append(item)
            else:
                self.rels.append(item)

        self.dtype = '<U100'

        self.question_toks = question_toks
        self.q_num = len(self.question_toks)

        ## get ent rel
        index_begin = gold_query.find('<extra_id_33>')+len('<extra_id_33>')
        index_end = gold_query.find('<extra_id_15')
        query_tuples = gold_query[index_begin:index_end].split(' <extra_id_38> ')
        self.label_tuples = [re.findall(r'[pq][0-9]+',item) for item in query_tuples]

        ent_item = {'labels':[], 'ranges':[]}
        ent_nums = 0
        for item in self.ents:  
            label = re.findall(r'[pq][0-9]+',item)
            if len(label) == 0:
                label = ['q0']
            assert(len(label) == 1)
            ent_item['labels'].append(label[0])
            ent_item['ranges'].append([ent_nums, ent_nums+len(item.split())-1])
            ent_nums = ent_nums + len(item.split())
            
            
        rel_item = {'labels':[], 'ranges':[]}
        rel_nums = 0
        for item in self.rels:  
            label = re.findall(r'[pq][0-9]+',item)
            if len(label) == 0:
                label = ['p0']
            assert(len(label) == 1)
            rel_item['labels'].append(label[0])
            rel_item['ranges'].append([rel_nums, rel_nums+len(item.split())-1])
            rel_nums = rel_nums + len(item.split())
        
        self.ent_nums = ent_nums
        self.rel_nums = rel_nums
        self.ent_item = ent_item
        self.rel_item = rel_item
            
            
    '''Question-Question Relation'''
    def read_tree(self, tree_mat):
        question_doc = self.nlp_tokenize(self.raw_question)
        trees = dict()
        root = None
            
        bias = 0
        for sent in question_doc.sentences: 
            for word in sent.words:
                tree = Tree()
                tree.idx = word.id -1 + bias 
                trees[tree.idx] = tree
            bias += len(sent.words)
        bias = 0
        for idx, sent in enumerate(question_doc.sentences): 
            for word in sent.words:
                head_id = word.head - 1 + bias
                word_id = word.id - 1 + bias
                if word.head - 1 == -1:
                    root = trees[word_id]
                    continue
                trees[head_id].add_child(trees[word_id])
                tree_mat[head_id, word_id] = "Forward-Syntax"
                tree_mat[word_id, head_id] = "Backward-Syntax"
            bias += len(sent.words)
        return tree_mat

    def get_qq_dependency(self,):
        
        q_depend_mat = np.array([["None-Syntax"] * (self.q_num) \
                        for _ in range(self.q_num)], dtype=self.dtype)
        q_depend_mat = self.read_tree(q_depend_mat)
        return q_depend_mat

    def get_qq_distance(self,):
        if self.q_num <= MAX_RELATIVE_DIST + 1:
            dist_vec = ['question-question-dist' + str(i) if i != 0 else 'question-question-identity'
                        for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1, 1)]
            starting = MAX_RELATIVE_DIST
        else:
            dist_vec = ['question-question-generic'] * (self.q_num - MAX_RELATIVE_DIST - 1) + \
                ['question-question-dist' + str(i) if i != 0 else 'question-question-identity' 
                for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1, 1)] + \
                ['question-question-generic'] * (self.q_num - MAX_RELATIVE_DIST - 1)
            starting = self.q_num - 1
    
        q_dist_mat = np.array([dist_vec[starting - i: starting - i + self.q_num] \
                                for i in range(self.q_num)], dtype=self.dtype)

        return q_dist_mat


        
    '''Question-Entity/Relation Relation'''

    def get_qer_relation(self, mode):
        if mode =='entity':
            ent_rel = self.ents
        elif mode == 'relation':
            ent_rel = self.rels
        else:
            raise NotImplementedError

        entrel_toks = []
        for item in ent_rel:
            entrel_toks += item.split()

        entrel_item = {'names':[],'ranges':[]}
        begin = 0
        for item in ent_rel:
            item_doc = self.nlp_tokenize(' '.join(item.split()[2:]))
            try:
                ent_rel_tok = [w.lemma.lower() for s in item_doc.sentences for w in s.words]
            except:
                ent_rel_tok = [ w.lower() for w in item.split()[2:]]
            entrel_item['names'].append( ' '.join(ent_rel_tok))
            entrel_item['ranges'].append([begin, begin+len(item.split())-1])
            begin = begin+len(item.split())

        q_num, t_num = self.q_num, len(entrel_toks)

        q_er_mat = np.array([[f'question-{mode}-nomatch'] * t_num  \
                             for _ in range(q_num)], dtype=self.dtype)
        er_q_mat = np.array([[f'{mode}-question-nomatch'] * q_num  \
                             for _ in range(t_num)], dtype=self.dtype)
        max_len = max([len(t) for t in entrel_toks])

        index_pairs = list(filter(lambda x: x[1] - x[0] <= max_len, combinations(range(q_num + 1), 2)))
        index_pairs = sorted(index_pairs, key=lambda x: x[1] - x[0])

        for i, j in index_pairs:
            phrase = ' '.join(self.question_toks[i: j])
            if phrase in stopwords.words("english"): continue
            for idx, name in enumerate(entrel_item['names']):
                item_begin = entrel_item['ranges'][idx][0]
                item_end = entrel_item['ranges'][idx][1]+1
                if phrase == name: # fully match will overwrite partial match due to sort
                    q_er_mat[i:j, item_begin:item_end] = f'question-{mode}-exactmatch'
                    er_q_mat[item_begin:item_end, i:j] = f'{mode}-question-exactmatch'
                elif (j - i == 1 and phrase in name.split()) or (j - i > 1 and phrase in name):
                    q_er_mat[i:j, item_begin:item_end] = f'question-{mode}-partialmatch'
                    er_q_mat[item_begin:item_end, i:j] = f'{mode}-question-partialmatch'


        return q_er_mat, er_q_mat



    '''Entity-Relation Relation'''

    def get_er_relation(self,):

        ent_item = self.ent_item
        rel_item = self.rel_item

        e_r_mat = np.array([['entity-relation-generic'] * self.rel_nums \
                            for _ in range(self.ent_nums)], dtype=self.dtype)
        r_e_mat = np.array([['relation-entity-generic'] * self.ent_nums \
                            for _ in range(self.rel_nums)], dtype=self.dtype)
        
        for (i,j) in itertools.product(range(len(ent_item['labels'])), range(len(rel_item['labels']))):                   
            for label_t in self.label_tuples:
                if (ent_item['labels'][i] in label_t) and (rel_item['labels'][j] in label_t):
                    e_r_mat[ent_item['ranges'][i][0]:ent_item['ranges'][i][1]+1,\
                            rel_item['ranges'][j][0]:rel_item['ranges'][j][1]+1] \
                            = 'entity-relation-cooccurence'
                
                    r_e_mat[rel_item['ranges'][j][0]:rel_item['ranges'][j][1]+1, \
                            ent_item['ranges'][i][0]:ent_item['ranges'][i][1]+1,] \
                            = 'relation-entity-cooccurence'
                    
                    break

        return e_r_mat, r_e_mat



    '''Entity-Entity / Relation-Relation Relation'''
    def get_ee_rr_relation(self,mode):
        if mode =='entity':
            entrel = self.ents
            entrel_nums = self.ent_nums
            entrel_item = self.ent_item
        elif mode == 'relation':
            entrel = self.rels
            entrel_nums = self.rel_nums
            entrel_item = self.rel_item
        else:
            raise NotImplementedError
        

        er_er_mat = np.array([[f'{mode}-{mode}-generic'] * entrel_nums \
                                for _ in range(entrel_nums)], dtype=self.dtype)

        length = len(entrel_item['labels'])
        
        for (i,j) in itertools.product(range(length), range(length)):
            if i == j:
                begin = entrel_item['ranges'][i][0]
                end = entrel_item['ranges'][i][1]
                er_er_mat[begin:begin+2+1,begin:begin+2+1] = 'label-label-identity'
                er_er_mat[begin+2:end+1,begin+2:end+1] = 'name-name-identity'
                er_er_mat[begin:begin+2+1,begin+2:end+1] = 'label-name-decode'
                er_er_mat[begin+2:end+1,begin:begin+2+1] = 'name-label-encode'
                continue
                                
            for label_t in self.label_tuples:
                if (entrel_item['labels'][i] == entrel_item['labels'][j]):
                    er_er_mat[entrel_item['ranges'][i][0]:entrel_item['ranges'][i][1]+1,\
                            entrel_item['ranges'][j][0]:entrel_item['ranges'][j][1]+1] \
                            = f'{mode}-{mode}-identity'
                    
                elif (entrel_item['labels'][i] in label_t) and (entrel_item['labels'][j] in label_t):
                    er_er_mat[entrel_item['ranges'][i][0]:entrel_item['ranges'][i][1]+1,\
                            entrel_item['ranges'][j][0]:entrel_item['ranges'][j][1]+1] \
                            = f'{mode}-{mode}-cooccurence'

        return er_er_mat