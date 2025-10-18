from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


model.eval()


input_text = "This autumn is the" 


input_ids = tokenizer.encode(input_text, return_tensors='pt')


output_ids = model.generate(
    input_ids,
    max_new_tokens=1,       
    do_sample=True,        
    top_k=50,              
    top_p=0.95,             
    temperature=0.8      
)

output_text1 = tokenizer.decode(output_ids[0], skip_special_tokens=True)

input_ids2 = tokenizer.encode(output_text1, return_tensors='pt')

output_ids2 = model.generate(
    input_ids2,
    max_new_tokens=1,       
    do_sample=True,        
    top_k=50,              
    top_p=0.95,             
    temperature=0.8      
)
output_text2 = tokenizer.decode(output_ids2[0], skip_special_tokens=True)

print(f"Input sequence: {input_text}")
print(f"Predicted continuation: {output_text2}")
