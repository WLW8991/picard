import re

def process(string):
    vocab=['?x','{','}','?uri','SELECT', 'DISTINCT', 'COUNT', '(', ')','WHERE', 
           '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>', '.','ASK','[DEF]',
           '<http://dbpedia.org/ontology/', '<http://dbpedia.org/property/', 
           '<http://dbpedia.org/resource/']

    vocab_dict = {}
    for i,text in enumerate(vocab):
        vocab_dict['<extra_id_'+str(i)+'>'] = text
                        
    for key in vocab_dict:
        string = string.replace(key, ' '+vocab_dict[key]+' ')
    
    variable = ['<http://dbpedia.org/ontology/', '<http://dbpedia.org/property/',
                '<http://dbpedia.org/resource/']
    
    vals = string.split()
    for i,val in enumerate(vals):
        if val in variable:
            if i < len(vals)-1:
                vals[i] = val+vals[i+1]+'>'
                vals[i+1] = ''
                  
    string = ' '.join(vals).strip()
    
    return ' '.join(string.split())