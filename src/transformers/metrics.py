from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
import sys

def intersection(lst1, lst2): 
    # Use of hybrid method 
    temp = set(lst2) 
    lst3 = [value for value in lst1 if value in temp] 
    return lst3 

def calculate_fidelity(preds, labels):
        labels_excellent, labels_good = labels[0], labels[1]

        print('preds count{}:{}'.format(len(preds), preds[:10]))
        print('label excellent count{}:{}'.format(len(labels_excellent), labels_excellent[:10]))
        print('label good count{}:{}'.format(len(labels_good), labels_good[:10]))
        # calculation
        score = 0.0
        score_ideal = 0.0
        for docs, docs_excellent, docs_good in zip(preds, labels_excellent, labels_good):
            score = score + 15.0 * len(intersection(docs, docs_excellent)) + 7.0 * len(intersection(docs, docs_good))
            score_ideal = score_ideal + 15.0 * len(docs_excellent) + 7.0 * len(docs_good)
        print("score: {}".format(score))
        print("score_ideal: {}".format(score_ideal))
        return score/score_ideal

def parse_label_dict(labels_ids, labels):
    #label_id: 'a_id'_'b_id'
    labels_excellent_dict, labels_good_dict = {}, {}
    for labels_id, label in zip(labels_ids, labels):
        a_id = labels_id.split('-')[0]
        b_id = labels_id.split('-')[1]
        if label== 1.0:
            if a_id not in labels_excellent_dict:
                labels_excellent_dict[a_id]=[]
            labels_excellent_dict[a_id].append(b_id)
        if label== 2.0:
            if a_id not in labels_good_dict:
                labels_good_dict[a_id]=[]
            labels_good_dict[a_id].append(b_id)
    print("labels_excellent_dict:", labels_excellent_dict)
    print("labels_good_dict:", labels_good_dict)
    return labels_excellent_dict, labels_good_dict

def get_fidelity(a_ids, a_preds, b_ids, b_preds, labels_ids, labels):
    #
    #step1: ann search
    #
    # Dimension of our vector space
    dimension = 64

    # Create a random binary hash with 10 bits
    rbp = RandomBinaryProjections('rbp', 10)

    # Create engine with pipeline configuration
    engine = Engine(dimension, lshashes=[rbp])

    # Index 1000000 random vectors (set their data to a unique string)
    if len(a_ids) != len(a_preds):
        sys.exit("length doesn't match: a_ids: {}, a_preds: {}".format(len(a_ids),len(a_predss)))
    if len(b_ids) != len(b_preds):
        sys.exit("length doesn't match: b_ids: {}, b_preds: {}".format(len(b_ids),len(b_predss)))
    if len(labels_ids) != len(labels):
        sys.exit("length doesn't match: labels_ids: {}, labels: {}".format(len(labels_ids),len(labels)))
    for b_id, b_pred in zip(b_ids, b_preds):
        engine.store_vector(b_pred, '{}'.format(b_id))

    labels_excellent_dict, labels_good_dict = parse_label_dict(labels_ids, labels)
    # Get nearest neighbours
    preds = []
    labels_excellent, labels_good = [], []
    for a_id, a_pred in zip(a_ids, a_preds):
        results = get_neighbours()
        results = engine.neighbours(a_pred)
        preds.append([result[1] for result in results])
        labels_excellent.append(labels_excellent_dict[a_id] if a_id in labels_excellent_dict else [])
        labels_good.append(labels_good_dict[a_id] if a_id in labels_good_dict else [])


    #
    # calculate fidelity
    #
    fidelity = calculate_fidelity(preds, [labels_excellent, labels_good])
    return fidelity