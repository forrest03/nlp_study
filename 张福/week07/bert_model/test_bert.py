import torch
from transformers import BertModel, BertTokenizer

print("Testing BERT model loading...")

# Test loading model
try:
    model = BertModel.from_pretrained('bert-base-chinese')
    print(f"Model type: {type(model)}")
    print(f"Model loaded successfully")
    
    # Test forward pass
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    inputs = tokenizer("测试文本", return_tensors="pt")
    print(f"Input keys: {inputs.keys()}")
    
    # Test model output
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"Output type: {type(outputs)}")
    print(f"Output keys: {outputs.keys() if hasattr(outputs, 'keys') else 'N/A'}")
    
    if hasattr(outputs, 'last_hidden_state'):
        print(f"last_hidden_state shape: {outputs.last_hidden_state.shape}")
    else:
        print("No last_hidden_state attribute")
        print(f"Output is tuple: {isinstance(outputs, tuple)}")
        if isinstance(outputs, tuple):
            print(f"Tuple length: {len(outputs)}")
            for i, o in enumerate(outputs):
                print(f"  Element {i} type: {type(o)}, shape: {o.shape if hasattr(o, 'shape') else 'N/A'}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
