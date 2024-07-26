import json
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.text  import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences

with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

model=load_model('sentiment.h5')

t = "Просто вкусно, этого достаточно, спасибо за рецепт".lower()
data = tokenizer.texts_to_sequences([t])
data_pad = pad_sequences(data, maxlen=30)

res = model.predict(data_pad)
print(res, np.argmax(res), t, data, data_pad)

if res[0][1] > 0.55:
    print (1)
elif res[0][1] > 0.45:
    print (0)
else:
    print (-1)