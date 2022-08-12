import os
MAX_RELATIVE_DIST = 2 

RELATIONS = ['name-label-encode', 'label-name-decode', 'name-name-identity', 'label-label-identity'] + \
    ['entity-entity-identity', 'relation-relation-identity'] + \
    ['question-question-dist' + str(i) if i != 0 else 'question-question-identity' for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1)] + \
    ['Forward-Syntax', 'Backward-Syntax', 'None-Syntax'] + \
    ['entity-entity-cooccurence', 'entity-relation-cooccurence', 'relation-entity-cooccurence', 'relation-relation-cooccurence'] + \
    ['question-entity-exactmatch', 'question-entity-partialmatch', 'question-entity-nomatch', 
    'entity-question-exactmatch', 'entity-question-partialmatch', 'entity-question-nomatch'] + \
    ['question-relation-exactmatch', 'question-relation-partialmatch', 'question-relation-nomatch',
    'relation-question-exactmatch', 'relation-question-partialmatch', 'relation-question-nomatch'] + \
    ['question-question-generic', 'entity-entity-generic', 'relation-relation-generic','relation-entity-generic', 'entity-relation-generic']
