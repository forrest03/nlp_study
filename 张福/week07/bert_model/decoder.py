def decode_entities(tokens, pred_labels, id2label):
    entities = []
    current_entity = None
    
    for i, (token, label_id) in enumerate(zip(tokens, pred_labels)):
        label = id2label[label_id]
        
        if label.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                'text': token,
                'label': label[2:],
                'start': i,
                'end': i + 1
            }
        elif label.startswith('I-') and current_entity:
            current_entity['text'] += token
            current_entity['end'] = i + 1
        elif label.startswith('E-') and current_entity:
            current_entity['text'] += token
            current_entity['end'] = i + 1
            entities.append(current_entity)
            current_entity = None
        elif label == 'O':
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    
    return entities

def decode_batch(batch_outputs, batch_tokens, id2label):
    decoded_results = []
    for i in range(len(batch_tokens)):
        pred_labels = batch_outputs[i]
        tokens = batch_tokens[i]
        entities = decode_entities(tokens, pred_labels, id2label)
        decoded_results.append({
            'tokens': tokens,
            'entities': entities
        })
    return decoded_results