import torch # we use PyTorch: https://pytorch.org
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

#
#
#
# DO NOT EDIT
#
#
#
def test_tokenization(text_data):
    """Tests the tokenization functions"""
    vocab = get_vocab(text_data)
    encode_map = create_encode_mapping(vocab)
    decode_map = create_decode_mapping(vocab)
    word_to_encode = "hobbit"
    encoded_word = encode(word_to_encode, encode_map)
    assert(encoded_word, [61, 68, 55, 55, 62, 73])
    decoded_word = decode(encoded_word, decode_map)
    assert(decoded_word, word_to_encode)


#
#
#
# Start here
#
#
#

print("Hello World")

# Todo: load the data here
text = ""

print(text[:1000])

# todo: implement
# text_data is a string containing the text from the input file
# return a sorted list of the unique characters in the text
def get_vocab(text_data):
    """Takes a string and returns the vocabulary of the string, a list of unique characters"""
    # start with an empty list, this will be our vocabulary list, where each item is a character
    vocab = []
    # iterate through text_data and add all the unique characters in the data to vocab, with no repeat characters. Note: we care about punctuation, spaces, new lines, and more as well, not just numbers and letters.
    # there's less expensive ways to do this if you don't want to iterate through the text_data or add to the list; if you're interested, take a look at type conversion and sets in python

    # sort the list

    # The length of your vocabulary should be 80, this will throw an error if your list isn't 80 characters long
    assert(len(vocab) == 80)

    # return the vocabulary list
    return vocab

# todo: implement
# vocab is a sorted list of unique characters in the text
# returns a dictionary mapping characters to integers
def create_encode_mapping(vocab):
    """Takes a vocabulary and creates a mapping from characters to integers"""
    # Look up python dictionaries. This will store our mappings for encoding characters
    encode_map = {}
    # for each character in the vocabulary, create a mapping in encode_map
    # The mapping should be the index of the character in the vocab list. For example, if our vocabulary was ["a", "b", "c", "."], our mapping should look like "a" -> 0, "b" -> 1, "c" -> 2, "." -> 3.
    
    return encode_map

# todo: implement
# vocab is a sorted list of unique characters in the text
# returns the reverse of create_encode_mapping
def create_decode_mapping(vocab):
    """Takes a vocabulary and creates a mapping from integers to characters"""
    # This will store our mappings for decoding characters
    decode_map = {}
    # This is similar to the create_encode_mapping function, but reversed...

    return decode_map

# todo: implement
# string is a string of characters
# returns the encoded list of integers
def encode(string, encode_mapping):
    """Takes a string and encode_mapping and outputs a list of integers"""
    # start with an empty list, this will store our encoded sequence
    encoded = []
    # for each character in the string, find the integer that corresponds to the character in encode_mapping and add that integer to the encoded list (hint: remember this is a python dictionary)
    
    return encoded

# todo: implement
# integers is an encoded list of integers
# decodes and returns the string (reverse of encode)
def decode(integers, decode_mapping):
    """Takes a list of integers and decode_mapping and outputs a string of the integer sequence decoded"""
    # start with an empty list, this will store our encoded sequence
    decoded = []
    # for each integer in the list of integers, find the character that corresponds to the integer in decode_mapping and concatenate that character to the string (hint: remember this is a python dictionary)
    return decoded

# todo: implement
def split_data(data):
    """splits the data into 90% training and 10% validation"""
    # determine the index of the data split (where the 90% index is)

    # split the data along the index, so that the first 90% is the training data and the last 10% is the validation data
    # (hint: look up python list slicing)

    # return a tuple of the training data, validation data
    
    # return training_data, valid_data


#
#
# You don't need to edit these
#
#

test_tokenization(text)

# Let's now split up the data into train and validation sets
# done, no need to edit
def encode_all(text_data, encode_mapping):
    """Encodes the entire text dataset and stores into a torch.Tensor"""
    encoded_data = torch.tensor(encode(text_data, encode_mapping), dtype=torch.long)
    return encoded_data
vocab = get_vocab(text)
encode_map = create_encode_mapping(vocab)
decode_map = create_decode_mapping(vocab)
data = encode_all(text, encode_map)
train_data, val_data = split_data(data)

#
#
# Todo edit these
#
#


batch_size = 0 # how many independent sequences will we process in parallel?
context_length = 0 # what is the maximum context length for predictions?

# todo: implement
def get_batch(data):
    # randomly pick 4 contexts of length 8 each

    # pick 4 corresponding targets also of length 8


    # call torch.stack on contexts to get a 4x8 matrix


    # call torch.stack on targets to get a 4x8 matrix

    # return context, targets
    pass # remove when implemented

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.weights = nn.Embedding(vocab_size, vocab_size)

    # context is the context matrix (1 batch of 4 contexts)
    # T is the corresponding target matrix
    def forward(self, context, T=None):
        # Use self.weights() to perform C x W
        T_predict = None 
        
        # If T is None, return T_predict, None

        # If T is not None, flatten T_predict

        # Then flatten T

        # Use F.cross_entropy(T_predict, T) to compute the loss

        # Return T_predict, loss

        pass # remove this line when you write this function

    def generate(self, context, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # call self.forward(context) to get T_predict, you can ignore the loss variable
            
            # Use F.softmax(T_predict, dim=-1) to get the probabilities

            # use torch.multinomial(probs, num_samples=1) to sample the next token index

            # append the sampled index to the context
            # update context = torch.cat((context, idx_next), dim=1) to do this
            pass # remove when you fill out this function

        return context

#
# Running the model
#

# todo: uncomment & implement m.generate
# m = BigramLanguageModel(vocab_size)
# logits, loss = m(xb, yb)
# print(logits.shape)
# print(loss)