import numpy as np

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def get_yeast_proteins():
    def get_protein_sequence(systematic_name):
        try:
            # Construct the URL for the SGD protein page
            
            url = f"https://www.yeastgenome.org/run_seqtools?format=fasta&type=protein&genes={systematic_name}&strains=S288C"

            # Send GET request to the URL
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for invalid responses

            if response.text:
                # The protein sequence is contained within a <pre> tag
                sequence = "".join(response.text.split("\n")[1:]).replace("*","")
                return sequence
            else:
                return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    # Example systematic names
    systematic_names = np.loadtxt("../data/yeast_network/yeast_string_genes.txt", delimiter="\n", dtype=str)
    print(systematic_names)
    print(systematic_names.shape)

    found_sequences = []
    not_found = []

    count = 0
    # Retrieve protein sequences for each systematic name
    for systematic_name in tqdm(systematic_names):
        protein_sequence = get_protein_sequence(systematic_name)
        if protein_sequence:
            found_sequences.append(protein_sequence)
            print(f"Systematic Name: {systematic_name}\nProtein Sequence:\n{protein_sequence}\n")
        else:
            not_found.append(systematic_name)
            print(f"No protein sequence found for systematic name: {systematic_name}\n")
            
        if count %25 == 0:
            np.save("../data/yeast_string_proteins.npy", np.array(found_sequences))
            np.save("../data/not_found_yeast_string_proteins.npy", np.array(not_found))
        count += 1

    np.save("../data/yeast_string_proteins.npy", np.array(found_sequences))
    print(f"{len(not_found)} not found")
    np.save("../data/not_found_yeast_string_proteins.npy", np.array(not_found))

    print(np.load("../data/yeast_string_proteins.npy").shape)
    

def get_esm_embeddings():
    from transformers import EsmTokenizer, EsmModel
    import torch

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
    seqs =list(np.load("../data/yeast_string_proteins.npy"))
    outs = np.zeros(shape=(1,320))
    print(f"Processing {len(seqs)} proteins")
    batch_size = 5
    for i in tqdm(range(0, len(seqs), batch_size)):
        curr = i
        next = min(i + batch_size,len(seqs))
        batch = seqs[curr:next]
        
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        x = last_hidden_states.detach()
        x = x.mean(axis=1)
        outs = np.concatenate([outs,x],axis=0)
        
    print(outs.shape)
    np.save("../data/esm_embeddings.npy",outs)
    assert  batch[-1] == seqs[-1]
    
    output = np.load("../data/esm_embeddings.npy")


   
    print(output.shape)
    print(output)

get_esm_embeddings()