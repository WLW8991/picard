import os
import re
import json

class Preprocess(object):
    def __init__(self):
        
        path = './seq2seq/datasets/lc_quad_1'
        ent_rel_labels = json.load(open(os.path.join(path, 'ent_rels.json'), 'rb'))
        #ent_rel_labels = json.load(open('./ent_rels.json', 'rb'))
        
        vocab = ['?x','{','}','?uri','SELECT', 'DISTINCT', 'COUNT', '(', ')', 'WHERE',
               '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>', '.','ASK','[DEF]',
               '<http://dbpedia.org/ontology/','<http://dbpedia.org/property/',
               '<http://dbpedia.org/resource/']

        vocab_dict = {}
        for i,text in enumerate(vocab):
            vocab_dict[text] = '<extra_id_'+str(i)+'>'

        self.ent_rel_labels = ent_rel_labels
        self.vocab_dict = vocab_dict
        
    
    def _replace_ent_rel(self,sparql):
        variable = ['<http://dbpedia.org/ontology/','<http://dbpedia.org/property/',
                    '<http://dbpedia.org/resource/']
        
        for item in variable:
            sparql = sparql.replace(item,self.vocab_dict[item]+' ')
        
        res = ''
        for t in range(len(sparql)):
            if sparql[t]=='>' and sparql[t-10:t-5]!='extra' and sparql[t-11:t-6]!='extra': 
                continue
            else: res += sparql[t]
        
        return res

    
    def _preprocess(self, data):
        dbsparql = data['sparql_query']
        question = data['corrected_question']

        sparql = dbsparql.replace('{', '{ ').replace('}', ' }').replace('?uri.','?uri .') \
        .replace('COUNT(?uri)', 'COUNT ( ?uri )')
        #sparql = ' '.join(sparql.split())
        
        # get processed input 
        _ent_rel = re.findall(r'<http://dbpedia.org/[a-z]+/.*?>', sparql)
        for item in _ent_rel:
            question_label = self._replace_ent_rel(item +' '+self.ent_rel_labels[item])
            question = question+self.vocab_dict['[DEF]']+' '+question_label
        question_input = ' '.join(question.split()).strip()
        
        # get processed gold query
        split = sparql.split()
        for idx,item in enumerate(split):
            if item in self.vocab_dict:
                split[idx] = self.vocab_dict[item]
        extra_sparql = ' '.join(split).strip()
        
        gold_query = self._replace_ent_rel(extra_sparql)
        

        return gold_query, question_input