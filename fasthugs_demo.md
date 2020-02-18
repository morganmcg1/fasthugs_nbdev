# Title
> summary


This notebook to demo's the use of the pretrained HuggingFace tokenizers and transformers with the new **fastai-v2** library.

### Pretrained Transformers only for now
Initially, this notebook will only deal with finetuning HuggingFace's pretrained models. It covers BERT, DistilBERT, RoBERTa and ALBERT pretrained classification models only. These are the core transformer model architectures where HuggingFace have added a classification head. HuggingFace also has other versions of these model architectures such as the core model architecture and language model model architectures.

If you'd like to try train a model from scratch HuggingFace just recently published an article on [How to train a new language model from scratch using Transformers and Tokenizers](https://huggingface.co/blog/how-to-train). Its well worth reading to see how their `tokenizers` library can be used, independent of their pretrained transformer models.

### Read these first 👇
This notebooks heavily borrows from [this notebook](https://www.kaggle.com/melissarajaram/roberta-fastai-huggingface-transformers) , which in turn is based off of this [tutorial](https://www.kaggle.com/maroberti/fastai-with-transformers-bert-roberta) and accompanying [article](https://towardsdatascience.com/fastai-with-transformers-bert-roberta-xlnet-xlm-distilbert-4f41ee18ecb2). Huge thanks to  Melissa Rajaram and Maximilien Roberti for these great resources, if you're not familiar with the HuggingFace library please 

### fastai-v2
[This paper](https://www.fast.ai/2020/02/13/fastai-A-Layered-API-for-Deep-Learning/) introduces the v2 version of the fastai library and you can follow and contribute to v2's progress [on the forums](https://forums.fast.ai/). This notebook is based off the [fastai-v2 ULMFiT tutorial](http://dev.fast.ai/tutorial.ulmfit). Huge thanks to Jeremy, Sylvain, Rachel and the fastai community for making this library what it is. I'm super excited about the additinal flexibility v2 brings.

### Dependencies
If you haven't already, install HuggingFace's `transformers` library with: `pip install transformers`

# Lets Go!

```python
path = untar_data(URLs.IMDB_SAMPLE)
model_path = Path('models')
df = pd.read_csv(path/'texts.csv')
```

### Prepare HuggingFace model classes

```python
models_dict = {'bert_classification': (BertForSequenceClassification, BertTokenizer, BertConfig),
                'roberta_classification': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),
                'distilbert_classification': (DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig),
                'albert_classification': (AlbertForSequenceClassification, AlbertTokenizer, AlbertConfig)
              }
```

```python
model_type = 'bert_classification'   
pretrained_model_name = 'bert-base-uncased'  # roberta-base

# model_type = 'roberta_classification'   
# pretrained_model_name = 'roberta-base'

# model_type = 'distilbert_classification'   
# pretrained_model_name = 'distilbert-base-uncased'

model_class, tokenizer_class, config_class = models_dict[model_type]
```

### Get Tokenizer from HuggingFace
Intialise the tokenizer needed for the pretrained model, this will download the `vocab.json` and `merges.txt` files needed. Specifying `cache_dir` will allow us easily access them, otherwise they will be saved to a Torch cache folder here `~/.cache/torch/transformers`. 

```python
transformer_tokenizer = tokenizer_class.from_pretrained(pretrained_model_name, 
                                                        cache_dir=model_path/f'{pretrained_model_name}')
```

 Model and vocab files will be saved with files names as a long string of digits and letters (e.g. `d9fc1956a0....f4cfdb5feda.json` generated from the etag from the AWS S3 bucket as described [here in the HuggingFace repo](https://github.com/huggingface/transformers/issues/2157). For readability I prefer to save the files in a specified directory and model name so that it can be easily found and accessed in future.
 
To avoid saving these files twice you could look at the `from_pretrained` and `cached_path` functions in HuggingFace's `PreTrainedTokenizer` class definition to find the code that downloads the files and maybe modify them to download directly to your specified directory withe desired name

```python
# TODO: Figure out how to give files model-specific names in case of using different transformers
transformer_tokenizer.save_vocabulary(model_path/f'{pretrained_model_name}')
```

Load vocab file into a list as expected by fastai-v2. BERT's vocab is saved as a .txt file where as RoBERTa's is saved as a .json

```python
if pretrained_model_name=='bert-base-uncased': 
    suff = 'txt'
else: 
    suff = 'json'

with open(model_path/f'{pretrained_model_name}/vocab.{suff}', 'r') as f: 
    if pretrained_model_name=='bert-base-uncased':
        transformer_vocab = f.read()
    else:
        transformer_vocab = list(json.load(f).keys()) 
```

### Custom FastHugs Tokenizer
Now that we have our vocab list, can look to incorporate HuggingFaces pretrained transformer tokenizers into fastai-v2's framework

```python
class FastHugsTokenizer():
    def __init__(self, pretrained_tokenizer = transformer_tokenizer, model_type = 'roberta', **kwargs):        
        self.tok = transformer_tokenizer
        self.max_seq_len = self.tok.max_len
        self.model_type = model_type
        self.pad_token_id = self.tok.pad_token_id
        
    def do_tokenize(self, t:str):
        """Limits the maximum sequence length and add the special tokens"""
        CLS = self.tok.cls_token
        SEP = self.tok.sep_token
        if 'roberta' in model_type:
            tokens = self.tok.tokenize(t, add_prefix_space=True)[:self.max_seq_len - 2]
        else:
            tokens = self.tok.tokenize(t)[:self.max_seq_len - 2]
        return [CLS] + tokens + [SEP]

    def __call__(self, items): 
        for t in items: yield self.do_tokenize(t)
```

```python
# Pass empty `rules` and `post_rules` list so fastai rules are not applied

fasthugstok = partial(FastHugsTokenizer, pretrained_tokenizer = transformer_tokenizer, model_type=model_type)

tok_fn = Tokenizer.from_df(text_cols='text', res_col_name='text', 
                           tok_func=fasthugstok,
                           rules=[], post_rules=[])
```

```python
jj = FastHugsTokenizer(pretrained_tokenizer = transformer_tokenizer, model_type=model_type)
```

```python
jj.max_seq_len
```

### Create Dataset
Lets add our custom tokenizer function (`tok_fn`) and `transformer_vocab` here

```python
splits = ColSplitter()(df)
x_tfms = [attrgetter("text"), tok_fn, Numericalize(vocab=transformer_vocab)]
dsets = Datasets(df, splits=splits, tfms=[x_tfms, [attrgetter("label"), Categorize()]], dl_type=SortedDL)
```

### Create Dataloaders
**Padding**: BERT, Roberta prefers padding to the right, so we set `pad_first=False`

```python
def transformer_padding(transformer_tokenizer): 
    if transformer_tokenizer.padding_side == 'right': 
        pad_first=False
    return partial(pad_input_chunk, pad_first=pad_first, pad_idx=transformer_tokenizer.pad_token_id)
```

```python
transformer_tokenizer.pad_token_id
```

Load you dataloader

```python
bs = 2
dls = dsets.dataloaders(bs=bs, device='cuda', before_batch=transformer_padding(transformer_tokenizer))
#dls = dsets.dataloaders(bs=bs, before_batch=pad_input_chunk)
```

```python
dls.show_batch()
```

```python
dsets.train.items.iloc[58:69] #head()
```

```python
dls.show_batch(max_n=2, trunc_at=60)
```

```python
y = dls.one_batch()
y[0].size(), y[1].size()
```

```python
y[0][1]
```

Factory dataloader. Here we set:
- `tok_tfm=tok_fn` to get our custom tokenizer
- `text_vocab=transformer_vocab` to load our pretrained vocab
- `before_batch=transformer_padding(transformer_tokenizer)` to make sure we set padding as the pretrained model expects

```python
# Factory
fct_dls = TextDataLoaders.from_df(df, text_col="text", tok_tfm=tok_fn, text_vocab=transformer_vocab,
                              before_batch=transformer_padding(transformer_tokenizer),
                              label_col='label', valid_col='is_valid', bs=bs)
```





```python
fct_dls.show_batch(max_n=2, trunc_at=60)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[ i [ [ [ [ [ [ [ ( [ [ [ [ [ : t [ [ [ [ [ [ [ [ ) [ a [ [ [ [ [ , [ [ [ [ [ [ [ [ [ [ " [ " . [ [ [ [ , [ [ [ [ [ [ [</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[ [ [ [ [ [ [ [ [ . [ [ [ , [ [ , [ [ [ , [ [ [ [ [ [ [ [ [ [ [ [ [ . [ [ [ [ [ [ [ [ , [ [ [ [ [ [ [ [ [ a [ [ [ [ [ [</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>


### Load HuggingFace models

```python
# More or less copy past from https://www.kaggle.com/melissarajaram/roberta-fastai-huggingface-transformers/data
class FastHugsModel(nn.Module):
    def __init__(self, transformer_model):
        super(FastHugsModel, self).__init__()
        self.transformer = transformer_model
        
    def forward(self, input_ids, attention_mask=None):
        attention_mask = (input_ids!=1).type(input_ids.type()) # attention_mask for RoBERTa
        print(input_ids.size())
        logits = self.transformer(input_ids, attention_mask = attention_mask)[0] 
        print(logits.size())
        return logits
```

### Model Setup - Model Splitters
HuggingFace's models with names such as: `RobertaForSequenceClassification` are core trasnformer models with a classification head. Lets split the classification head from the core transformer backbone to enable us use progressive unfreezing and differential learning rates.

**Classification Head Differences**

Interestingly, BERT's classification head is different to RoBERTa's

BERT's:

`
(dropout): Dropout(p=0.1, inplace=False)
(classifier): Linear(in_features=768, out_features=2, bias=True)
`

DistilBERT's has a "pre-classifier" layer:

`
(pre_classifier): Linear(in_features=768, out_features=768, bias=True)
(classifier): Linear(in_features=768, out_features=2, bias=True)
(dropout): Dropout(p=0.2, inplace=False)`

RoBERTa's:

`(classifier): RobertaClassificationHead(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
    (out_proj): Linear(in_features=768, out_features=2, bias=True))`

```python
def bert_clas_splitter(m):
    "Split the classifier head from the backbone"
    groups = [nn.Sequential(m.transformer.bert.embeddings,
                m.transformer.bert.encoder.layer[0],
                m.transformer.bert.encoder.layer[1],
                m.transformer.bert.encoder.layer[2],
                m.transformer.bert.encoder.layer[3],
                m.transformer.bert.encoder.layer[4],
                m.transformer.bert.encoder.layer[5],
                m.transformer.bert.encoder.layer[6],
                m.transformer.bert.encoder.layer[7],
                m.transformer.bert.encoder.layer[8],
                m.transformer.bert.encoder.layer[9],
                m.transformer.bert.encoder.layer[10],
                m.transformer.bert.encoder.layer[11],
                m.transformer.bert.pooler)]
    groups = L(groups + [m.transformer.classifier]) 
    return groups.map(params)
```

```python
def distilbert_clas_splitter(m):
    groups = [nn.Sequential(m.embeddings,
                m.transformer.layer[0],
                m.transformer.layer[1],
                m.transformer.layer[2],
                m.transformer.layer[3],
                m.transformer.layer[4],
                m.transformer.layer[6],
                m.pre_classifier)]
    groups = L(groups + [m.classifier]) 
    return groups.map(params)
```

```python
def roberta_clas_splitter(m):
    "Split the classifier head from the backbone"
    groups = [nn.Sequential(m.transformer.roberta.embeddings,
                  m.transformer.roberta.encoder.layer[0],
                  m.transformer.roberta.encoder.layer[1],
                  m.transformer.roberta.encoder.layer[2],
                  m.transformer.roberta.encoder.layer[3],
                  m.transformer.roberta.encoder.layer[4],
                  m.transformer.roberta.encoder.layer[5],
                  m.transformer.roberta.encoder.layer[6],
                  m.transformer.roberta.encoder.layer[7],
                  m.transformer.roberta.encoder.layer[8],
                  m.transformer.roberta.encoder.layer[9],
                  m.transformer.roberta.encoder.layer[10],
                  m.transformer.roberta.encoder.layer[11],
                  m.transformer.roberta.pooler)]
    groups = L(groups + [m.transformer.classifier])
    return groups.map(params)
```

### Load Model with configs

Here we can tweak the HuggingFace model's config file before loading the model

```python
config = config_class.from_pretrained(pretrained_model_name)
config.num_labels = 2
config.save_pretrained(model_path/f'{pretrained_model_name}') 
```

```python
model = model_class.from_pretrained(pretrained_model_name, config = config, 
                                    cache_dir=model_path/f'{pretrained_model_name}')
model.save_pretrained(model_path/f'{pretrained_model_name}')
```

```python
model
```




    BertForSequenceClassification(
      (bert): BertModel(
        (embeddings): BertEmbeddings(
          (word_embeddings): Embedding(30522, 768, padding_idx=0)
          (position_embeddings): Embedding(512, 768)
          (token_type_embeddings): Embedding(2, 768)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (encoder): BertEncoder(
          (layer): ModuleList(
            (0): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (1): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (2): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (3): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (4): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (5): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (6): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (7): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (8): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (9): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (10): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (11): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
        (pooler): BertPooler(
          (dense): Linear(in_features=768, out_features=768, bias=True)
          (activation): Tanh()
        )
      )
      (dropout): Dropout(p=0.1, inplace=False)
      (classifier): Linear(in_features=768, out_features=2, bias=True)
    )



```python
dd = [p[0] for p in model.named_parameters() if p[1].requires_grad]
dd
```




    ['bert.embeddings.word_embeddings.weight',
     'bert.embeddings.position_embeddings.weight',
     'bert.embeddings.token_type_embeddings.weight',
     'bert.embeddings.LayerNorm.weight',
     'bert.embeddings.LayerNorm.bias',
     'bert.encoder.layer.0.attention.self.query.weight',
     'bert.encoder.layer.0.attention.self.query.bias',
     'bert.encoder.layer.0.attention.self.key.weight',
     'bert.encoder.layer.0.attention.self.key.bias',
     'bert.encoder.layer.0.attention.self.value.weight',
     'bert.encoder.layer.0.attention.self.value.bias',
     'bert.encoder.layer.0.attention.output.dense.weight',
     'bert.encoder.layer.0.attention.output.dense.bias',
     'bert.encoder.layer.0.attention.output.LayerNorm.weight',
     'bert.encoder.layer.0.attention.output.LayerNorm.bias',
     'bert.encoder.layer.0.intermediate.dense.weight',
     'bert.encoder.layer.0.intermediate.dense.bias',
     'bert.encoder.layer.0.output.dense.weight',
     'bert.encoder.layer.0.output.dense.bias',
     'bert.encoder.layer.0.output.LayerNorm.weight',
     'bert.encoder.layer.0.output.LayerNorm.bias',
     'bert.encoder.layer.1.attention.self.query.weight',
     'bert.encoder.layer.1.attention.self.query.bias',
     'bert.encoder.layer.1.attention.self.key.weight',
     'bert.encoder.layer.1.attention.self.key.bias',
     'bert.encoder.layer.1.attention.self.value.weight',
     'bert.encoder.layer.1.attention.self.value.bias',
     'bert.encoder.layer.1.attention.output.dense.weight',
     'bert.encoder.layer.1.attention.output.dense.bias',
     'bert.encoder.layer.1.attention.output.LayerNorm.weight',
     'bert.encoder.layer.1.attention.output.LayerNorm.bias',
     'bert.encoder.layer.1.intermediate.dense.weight',
     'bert.encoder.layer.1.intermediate.dense.bias',
     'bert.encoder.layer.1.output.dense.weight',
     'bert.encoder.layer.1.output.dense.bias',
     'bert.encoder.layer.1.output.LayerNorm.weight',
     'bert.encoder.layer.1.output.LayerNorm.bias',
     'bert.encoder.layer.2.attention.self.query.weight',
     'bert.encoder.layer.2.attention.self.query.bias',
     'bert.encoder.layer.2.attention.self.key.weight',
     'bert.encoder.layer.2.attention.self.key.bias',
     'bert.encoder.layer.2.attention.self.value.weight',
     'bert.encoder.layer.2.attention.self.value.bias',
     'bert.encoder.layer.2.attention.output.dense.weight',
     'bert.encoder.layer.2.attention.output.dense.bias',
     'bert.encoder.layer.2.attention.output.LayerNorm.weight',
     'bert.encoder.layer.2.attention.output.LayerNorm.bias',
     'bert.encoder.layer.2.intermediate.dense.weight',
     'bert.encoder.layer.2.intermediate.dense.bias',
     'bert.encoder.layer.2.output.dense.weight',
     'bert.encoder.layer.2.output.dense.bias',
     'bert.encoder.layer.2.output.LayerNorm.weight',
     'bert.encoder.layer.2.output.LayerNorm.bias',
     'bert.encoder.layer.3.attention.self.query.weight',
     'bert.encoder.layer.3.attention.self.query.bias',
     'bert.encoder.layer.3.attention.self.key.weight',
     'bert.encoder.layer.3.attention.self.key.bias',
     'bert.encoder.layer.3.attention.self.value.weight',
     'bert.encoder.layer.3.attention.self.value.bias',
     'bert.encoder.layer.3.attention.output.dense.weight',
     'bert.encoder.layer.3.attention.output.dense.bias',
     'bert.encoder.layer.3.attention.output.LayerNorm.weight',
     'bert.encoder.layer.3.attention.output.LayerNorm.bias',
     'bert.encoder.layer.3.intermediate.dense.weight',
     'bert.encoder.layer.3.intermediate.dense.bias',
     'bert.encoder.layer.3.output.dense.weight',
     'bert.encoder.layer.3.output.dense.bias',
     'bert.encoder.layer.3.output.LayerNorm.weight',
     'bert.encoder.layer.3.output.LayerNorm.bias',
     'bert.encoder.layer.4.attention.self.query.weight',
     'bert.encoder.layer.4.attention.self.query.bias',
     'bert.encoder.layer.4.attention.self.key.weight',
     'bert.encoder.layer.4.attention.self.key.bias',
     'bert.encoder.layer.4.attention.self.value.weight',
     'bert.encoder.layer.4.attention.self.value.bias',
     'bert.encoder.layer.4.attention.output.dense.weight',
     'bert.encoder.layer.4.attention.output.dense.bias',
     'bert.encoder.layer.4.attention.output.LayerNorm.weight',
     'bert.encoder.layer.4.attention.output.LayerNorm.bias',
     'bert.encoder.layer.4.intermediate.dense.weight',
     'bert.encoder.layer.4.intermediate.dense.bias',
     'bert.encoder.layer.4.output.dense.weight',
     'bert.encoder.layer.4.output.dense.bias',
     'bert.encoder.layer.4.output.LayerNorm.weight',
     'bert.encoder.layer.4.output.LayerNorm.bias',
     'bert.encoder.layer.5.attention.self.query.weight',
     'bert.encoder.layer.5.attention.self.query.bias',
     'bert.encoder.layer.5.attention.self.key.weight',
     'bert.encoder.layer.5.attention.self.key.bias',
     'bert.encoder.layer.5.attention.self.value.weight',
     'bert.encoder.layer.5.attention.self.value.bias',
     'bert.encoder.layer.5.attention.output.dense.weight',
     'bert.encoder.layer.5.attention.output.dense.bias',
     'bert.encoder.layer.5.attention.output.LayerNorm.weight',
     'bert.encoder.layer.5.attention.output.LayerNorm.bias',
     'bert.encoder.layer.5.intermediate.dense.weight',
     'bert.encoder.layer.5.intermediate.dense.bias',
     'bert.encoder.layer.5.output.dense.weight',
     'bert.encoder.layer.5.output.dense.bias',
     'bert.encoder.layer.5.output.LayerNorm.weight',
     'bert.encoder.layer.5.output.LayerNorm.bias',
     'bert.encoder.layer.6.attention.self.query.weight',
     'bert.encoder.layer.6.attention.self.query.bias',
     'bert.encoder.layer.6.attention.self.key.weight',
     'bert.encoder.layer.6.attention.self.key.bias',
     'bert.encoder.layer.6.attention.self.value.weight',
     'bert.encoder.layer.6.attention.self.value.bias',
     'bert.encoder.layer.6.attention.output.dense.weight',
     'bert.encoder.layer.6.attention.output.dense.bias',
     'bert.encoder.layer.6.attention.output.LayerNorm.weight',
     'bert.encoder.layer.6.attention.output.LayerNorm.bias',
     'bert.encoder.layer.6.intermediate.dense.weight',
     'bert.encoder.layer.6.intermediate.dense.bias',
     'bert.encoder.layer.6.output.dense.weight',
     'bert.encoder.layer.6.output.dense.bias',
     'bert.encoder.layer.6.output.LayerNorm.weight',
     'bert.encoder.layer.6.output.LayerNorm.bias',
     'bert.encoder.layer.7.attention.self.query.weight',
     'bert.encoder.layer.7.attention.self.query.bias',
     'bert.encoder.layer.7.attention.self.key.weight',
     'bert.encoder.layer.7.attention.self.key.bias',
     'bert.encoder.layer.7.attention.self.value.weight',
     'bert.encoder.layer.7.attention.self.value.bias',
     'bert.encoder.layer.7.attention.output.dense.weight',
     'bert.encoder.layer.7.attention.output.dense.bias',
     'bert.encoder.layer.7.attention.output.LayerNorm.weight',
     'bert.encoder.layer.7.attention.output.LayerNorm.bias',
     'bert.encoder.layer.7.intermediate.dense.weight',
     'bert.encoder.layer.7.intermediate.dense.bias',
     'bert.encoder.layer.7.output.dense.weight',
     'bert.encoder.layer.7.output.dense.bias',
     'bert.encoder.layer.7.output.LayerNorm.weight',
     'bert.encoder.layer.7.output.LayerNorm.bias',
     'bert.encoder.layer.8.attention.self.query.weight',
     'bert.encoder.layer.8.attention.self.query.bias',
     'bert.encoder.layer.8.attention.self.key.weight',
     'bert.encoder.layer.8.attention.self.key.bias',
     'bert.encoder.layer.8.attention.self.value.weight',
     'bert.encoder.layer.8.attention.self.value.bias',
     'bert.encoder.layer.8.attention.output.dense.weight',
     'bert.encoder.layer.8.attention.output.dense.bias',
     'bert.encoder.layer.8.attention.output.LayerNorm.weight',
     'bert.encoder.layer.8.attention.output.LayerNorm.bias',
     'bert.encoder.layer.8.intermediate.dense.weight',
     'bert.encoder.layer.8.intermediate.dense.bias',
     'bert.encoder.layer.8.output.dense.weight',
     'bert.encoder.layer.8.output.dense.bias',
     'bert.encoder.layer.8.output.LayerNorm.weight',
     'bert.encoder.layer.8.output.LayerNorm.bias',
     'bert.encoder.layer.9.attention.self.query.weight',
     'bert.encoder.layer.9.attention.self.query.bias',
     'bert.encoder.layer.9.attention.self.key.weight',
     'bert.encoder.layer.9.attention.self.key.bias',
     'bert.encoder.layer.9.attention.self.value.weight',
     'bert.encoder.layer.9.attention.self.value.bias',
     'bert.encoder.layer.9.attention.output.dense.weight',
     'bert.encoder.layer.9.attention.output.dense.bias',
     'bert.encoder.layer.9.attention.output.LayerNorm.weight',
     'bert.encoder.layer.9.attention.output.LayerNorm.bias',
     'bert.encoder.layer.9.intermediate.dense.weight',
     'bert.encoder.layer.9.intermediate.dense.bias',
     'bert.encoder.layer.9.output.dense.weight',
     'bert.encoder.layer.9.output.dense.bias',
     'bert.encoder.layer.9.output.LayerNorm.weight',
     'bert.encoder.layer.9.output.LayerNorm.bias',
     'bert.encoder.layer.10.attention.self.query.weight',
     'bert.encoder.layer.10.attention.self.query.bias',
     'bert.encoder.layer.10.attention.self.key.weight',
     'bert.encoder.layer.10.attention.self.key.bias',
     'bert.encoder.layer.10.attention.self.value.weight',
     'bert.encoder.layer.10.attention.self.value.bias',
     'bert.encoder.layer.10.attention.output.dense.weight',
     'bert.encoder.layer.10.attention.output.dense.bias',
     'bert.encoder.layer.10.attention.output.LayerNorm.weight',
     'bert.encoder.layer.10.attention.output.LayerNorm.bias',
     'bert.encoder.layer.10.intermediate.dense.weight',
     'bert.encoder.layer.10.intermediate.dense.bias',
     'bert.encoder.layer.10.output.dense.weight',
     'bert.encoder.layer.10.output.dense.bias',
     'bert.encoder.layer.10.output.LayerNorm.weight',
     'bert.encoder.layer.10.output.LayerNorm.bias',
     'bert.encoder.layer.11.attention.self.query.weight',
     'bert.encoder.layer.11.attention.self.query.bias',
     'bert.encoder.layer.11.attention.self.key.weight',
     'bert.encoder.layer.11.attention.self.key.bias',
     'bert.encoder.layer.11.attention.self.value.weight',
     'bert.encoder.layer.11.attention.self.value.bias',
     'bert.encoder.layer.11.attention.output.dense.weight',
     'bert.encoder.layer.11.attention.output.dense.bias',
     'bert.encoder.layer.11.attention.output.LayerNorm.weight',
     'bert.encoder.layer.11.attention.output.LayerNorm.bias',
     'bert.encoder.layer.11.intermediate.dense.weight',
     'bert.encoder.layer.11.intermediate.dense.bias',
     'bert.encoder.layer.11.output.dense.weight',
     'bert.encoder.layer.11.output.dense.bias',
     'bert.encoder.layer.11.output.LayerNorm.weight',
     'bert.encoder.layer.11.output.LayerNorm.bias',
     'bert.pooler.dense.weight',
     'bert.pooler.dense.bias',
     'classifier.weight',
     'classifier.bias']



Initialise everything our Learner

```python
fasthugs_model = FastHugsModel(transformer_model = model)

opt_func = partial(Adam, decouple_wd=True)

cbs = [MixedPrecision(clip=0.1), SaveModelCallback()]

loss = CrossEntropyLossFlat() #LabelSmoothingCrossEntropy
```

### Create our learner

```python
#dls.one_batch()
```

```python
learn = Learner(dls, fasthugs_model, opt_func=opt_func, # splitter=bert_clas_splitter, # roberta_clas_splitter,
                   loss_func=loss, metrics=[accuracy])
```

```python
#learn.show_training_loop()
```

```python
dsets.c
```




    2



```python
learn.lr_find(suggestions=True)
```





    torch.Size([2, 512])



    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    ~/fastai2/fastai2/learner.py in fit(self, n_epoch, lr, wd, cbs, reset_opt)
        293                         self.epoch=epoch;          self('begin_epoch')
    --> 294                         self._do_epoch_train()
        295                         self._do_epoch_validate()


    ~/fastai2/fastai2/learner.py in _do_epoch_train(self)
        268             self.dl = self.dls.train;                        self('begin_train')
    --> 269             self.all_batches()
        270         except CancelTrainException:                         self('after_cancel_train')


    ~/fastai2/fastai2/learner.py in all_batches(self)
        246         self.n_iter = len(self.dl)
    --> 247         for o in enumerate(self.dl): self.one_batch(*o)
        248 


    ~/fastai2/fastai2/learner.py in one_batch(self, i, b)
        252             self._split(b);                                  self('begin_batch')
    --> 253             self.pred = self.model(*self.xb);                self('after_pred')
        254             if len(self.yb) == 0: return


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        531         else:
    --> 532             result = self.forward(*input, **kwargs)
        533         for hook in self._forward_hooks.values():


    <ipython-input-22-a7e5dd8ac4a9> in forward(self, input_ids, attention_mask)
          9         print(input_ids.size())
    ---> 10         logits = self.transformer(input_ids, attention_mask = attention_mask)[0]
         11         print(logits.size())


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        531         else:
    --> 532             result = self.forward(*input, **kwargs)
        533         for hook in self._forward_hooks.values():


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/transformers/modeling_bert.py in forward(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels)
       1191             head_mask=head_mask,
    -> 1192             inputs_embeds=inputs_embeds,
       1193         )


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        531         else:
    --> 532             result = self.forward(*input, **kwargs)
        533         for hook in self._forward_hooks.values():


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/transformers/modeling_bert.py in forward(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, encoder_hidden_states, encoder_attention_mask)
        798         embedding_output = self.embeddings(
    --> 799             input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        800         )


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        531         else:
    --> 532             result = self.forward(*input, **kwargs)
        533         for hook in self._forward_hooks.values():


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/transformers/modeling_bert.py in forward(self, input_ids, token_type_ids, position_ids, inputs_embeds)
        189             inputs_embeds = self.word_embeddings(input_ids)
    --> 190         position_embeddings = self.position_embeddings(position_ids)
        191         token_type_embeddings = self.token_type_embeddings(token_type_ids)


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        531         else:
    --> 532             result = self.forward(*input, **kwargs)
        533         for hook in self._forward_hooks.values():


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/modules/sparse.py in forward(self, input)
        113             input, self.weight, self.padding_idx, self.max_norm,
    --> 114             self.norm_type, self.scale_grad_by_freq, self.sparse)
        115 


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/functional.py in embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
       1483         _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)
    -> 1484     return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
       1485 


    RuntimeError: CUDA error: device-side assert triggered

    
    During handling of the above exception, another exception occurred:


    RuntimeError                              Traceback (most recent call last)

    <ipython-input-37-35d7aa25ab99> in <module>
    ----> 1 learn.lr_find(suggestions=True)
    

    ~/fastai2/fastai2/callback/schedule.py in lr_find(self, start_lr, end_lr, num_it, stop_div, show_plot, suggestions)
        195     n_epoch = num_it//len(self.dls.train) + 1
        196     cb=LRFinder(start_lr=start_lr, end_lr=end_lr, num_it=num_it, stop_div=stop_div)
    --> 197     with self.no_logging(): self.fit(n_epoch, cbs=cb)
        198     if show_plot: self.recorder.plot_lr_find()
        199     if suggestions:


    ~/fastai2/fastai2/learner.py in fit(self, n_epoch, lr, wd, cbs, reset_opt)
        298 
        299             except CancelFitException:             self('after_cancel_fit')
    --> 300             finally:                               self('after_fit')
        301 
        302     def validate(self, ds_idx=1, dl=None, cbs=None):


    ~/fastai2/fastai2/learner.py in __call__(self, event_name)
        226     def ordered_cbs(self, cb_func): return [cb for cb in sort_by_run(self.cbs) if hasattr(cb, cb_func)]
        227 
    --> 228     def __call__(self, event_name): L(event_name).map(self._call_one)
        229     def _call_one(self, event_name):
        230         assert hasattr(event, event_name)


    ~/fastcore/fastcore/foundation.py in map(self, f, *args, **kwargs)
        360              else f.format if isinstance(f,str)
        361              else f.__getitem__)
    --> 362         return self._new(map(g, self))
        363 
        364     def filter(self, f, negate=False, **kwargs):


    ~/fastcore/fastcore/foundation.py in _new(self, items, *args, **kwargs)
        313     @property
        314     def _xtra(self): return None
    --> 315     def _new(self, items, *args, **kwargs): return type(self)(items, *args, use_list=None, **kwargs)
        316     def __getitem__(self, idx): return self._get(idx) if is_indexer(idx) else L(self._get(idx), use_list=None)
        317     def copy(self): return self._new(self.items.copy())


    ~/fastcore/fastcore/foundation.py in __call__(cls, x, *args, **kwargs)
         39             return x
         40 
    ---> 41         res = super().__call__(*((x,) + args), **kwargs)
         42         res._newchk = 0
         43         return res


    ~/fastcore/fastcore/foundation.py in __init__(self, items, use_list, match, *rest)
        304         if items is None: items = []
        305         if (use_list is not None) or not _is_array(items):
    --> 306             items = list(items) if use_list else _listify(items)
        307         if match is not None:
        308             if is_coll(match): match = len(match)


    ~/fastcore/fastcore/foundation.py in _listify(o)
        240     if isinstance(o, list): return o
        241     if isinstance(o, str) or _is_array(o): return [o]
    --> 242     if is_iter(o): return list(o)
        243     return [o]
        244 


    ~/fastcore/fastcore/foundation.py in __call__(self, *args, **kwargs)
        206             if isinstance(v,_Arg): kwargs[k] = args.pop(v.i)
        207         fargs = [args[x.i] if isinstance(x, _Arg) else x for x in self.pargs] + args[self.maxi+1:]
    --> 208         return self.fn(*fargs, **kwargs)
        209 
        210 # Cell


    ~/fastai2/fastai2/learner.py in _call_one(self, event_name)
        229     def _call_one(self, event_name):
        230         assert hasattr(event, event_name)
    --> 231         [cb(event_name) for cb in sort_by_run(self.cbs)]
        232 
        233     def _bn_bias_state(self, with_bias): return bn_bias_params(self.model, with_bias).map(self.opt.state)


    ~/fastai2/fastai2/learner.py in <listcomp>(.0)
        229     def _call_one(self, event_name):
        230         assert hasattr(event, event_name)
    --> 231         [cb(event_name) for cb in sort_by_run(self.cbs)]
        232 
        233     def _bn_bias_state(self, with_bias): return bn_bias_params(self.model, with_bias).map(self.opt.state)


    ~/fastai2/fastai2/learner.py in __call__(self, event_name)
         23         _run = (event_name not in _inner_loop or (self.run_train and getattr(self, 'training', True)) or
         24                (self.run_valid and not getattr(self, 'training', False)))
    ---> 25         if self.run and _run: getattr(self, event_name, noop)()
         26         if event_name=='after_fit': self.run=True #Reset self.run to True at each end of fit
         27 


    ~/fastai2/fastai2/callback/schedule.py in after_fit(self)
        168         tmp_f = self.path/self.model_dir/'_tmp.pth'
        169         if tmp_f.exists():
    --> 170             self.learn.load('_tmp')
        171             os.remove(tmp_f)
        172 


    ~/fastai2/fastai2/learner.py in load(self, file, with_opt, device, strict)
        372         distrib_barrier()
        373         file = join_path_file(file, self.path/self.model_dir, ext='.pth')
    --> 374         load_model(file, self.model, self.opt, with_opt=with_opt, device=device, strict=strict)
        375         return self
        376 


    ~/fastai2/fastai2/learner.py in load_model(file, model, opt, with_opt, device, strict)
        163     if isinstance(device, int): device = torch.device('cuda', device)
        164     elif device is None: device = 'cpu'
    --> 165     state = torch.load(file, map_location=device)
        166     hasopt = set(state)=={'model', 'opt'}
        167     model_state = state['model'] if hasopt else state


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/serialization.py in load(f, map_location, pickle_module, **pickle_load_args)
        527             with _open_zipfile_reader(f) as opened_zipfile:
        528                 return _load(opened_zipfile, map_location, pickle_module, **pickle_load_args)
    --> 529         return _legacy_load(opened_file, map_location, pickle_module, **pickle_load_args)
        530 
        531 


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/serialization.py in _legacy_load(f, map_location, pickle_module, **pickle_load_args)
        700     unpickler = pickle_module.Unpickler(f, **pickle_load_args)
        701     unpickler.persistent_load = persistent_load
    --> 702     result = unpickler.load()
        703 
        704     deserialized_storage_keys = pickle_module.load(f, **pickle_load_args)


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/serialization.py in persistent_load(saved_id)
        663                 obj = data_type(size)
        664                 obj._torch_load_uninitialized = True
    --> 665                 deserialized_objects[root_key] = restore_location(obj, location)
        666             storage = deserialized_objects[root_key]
        667             if view_metadata is not None:


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/serialization.py in restore_location(storage, location)
        735     elif isinstance(map_location, _string_classes):
        736         def restore_location(storage, location):
    --> 737             return default_restore_location(storage, map_location)
        738     elif isinstance(map_location, torch.device):
        739         def restore_location(storage, location):


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/serialization.py in default_restore_location(storage, location)
        154 def default_restore_location(storage, location):
        155     for _, _, fn in _package_registry:
    --> 156         result = fn(storage, location)
        157         if result is not None:
        158             return result


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/serialization.py in _cuda_deserialize(obj, location)
        134             storage_type = getattr(torch.cuda, type(obj).__name__)
        135             with torch.cuda.device(device):
    --> 136                 return storage_type(obj.size())
        137         else:
        138             return obj.cuda(device)


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/cuda/__init__.py in _lazy_new(cls, *args, **kwargs)
        478     # We may need to call lazy init again if we are a forked child
        479     # del _CudaBase.__new__
    --> 480     return super(_CudaBase, cls).__new__(cls, *args, **kwargs)
        481 
        482 


    RuntimeError: CUDA error: device-side assert triggered


### Stage 1 Training
Lets freeze the model backbone and only train the classifier head. `freeze_to(1)` means that only the classifier head is trainable

```python
#learn.model.transformer.
```

```python
learn.freeze_to(1)  
```

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(input_ids, labels=labels)

#loss, logits = outputs[:2]

outputs
```

    





    (tensor(0.8255, grad_fn=<NllLossBackward>),
     tensor([[0.4643, 0.2150]], grad_fn=<AddmmBackward>))



```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base')
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(input_ids, labels=labels)
outputs
#loss, logits = outputs[:2]
```




    (tensor(0.7342, grad_fn=<NllLossBackward>),
     tensor([[-0.0375, -0.1180]], grad_fn=<AddmmBackward>))



```python
learn.summary()
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-30-bc39e9e85f86> in <module>
    ----> 1 learn.summary()
    

    ~/fastai2/fastai2/callback/hook.py in summary(self)
        186     "Print a summary of the model, optimizer and loss function."
        187     xb = self.dls.train.one_batch()[:self.dls.train.n_inp]
    --> 188     res = self.model.summary(*xb)
        189     res += f"Optimizer used: {self.opt_func}\nLoss function: {self.loss_func}\n\n"
        190     if self.opt is not None:


    ~/fastai2/fastai2/callback/hook.py in summary(self, *xb)
        161 def summary(self:nn.Module, *xb):
        162     "Print a summary of `self` using `xb`"
    --> 163     sample_inputs,infos = layer_info(self, *xb)
        164     n,bs = 64,find_bs(xb)
        165     inp_sz = _print_shapes(apply(lambda x:x.shape, xb), bs)


    ~/fastai2/fastai2/callback/hook.py in layer_info(model, *xb)
        149     layers = [m for m in flatten_model(model)]
        150     with Hooks(layers, _track) as h:
    --> 151         _ = model.eval()(*apply(lambda o:o[:1], xb))
        152         return xb,h.stored
        153 


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        539             result = self._slow_forward(*input, **kwargs)
        540         else:
    --> 541             result = self.forward(*input, **kwargs)
        542         for hook in self._forward_hooks.values():
        543             hook_result = hook(self, input, result)


    <ipython-input-18-273fd30ceadc> in forward(self, input_ids, attention_mask)
          7     def forward(self, input_ids, attention_mask=None):
          8         attention_mask = (input_ids!=1).type(input_ids.type()) # attention_mask for RoBERTa
    ----> 9         logits = self.transformer(input_ids, attention_mask = attention_mask)[0]
         10 
         11         return logits


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        539             result = self._slow_forward(*input, **kwargs)
        540         else:
    --> 541             result = self.forward(*input, **kwargs)
        542         for hook in self._forward_hooks.values():
        543             hook_result = hook(self, input, result)


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/transformers/modeling_bert.py in forward(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels)
       1190             position_ids=position_ids,
       1191             head_mask=head_mask,
    -> 1192             inputs_embeds=inputs_embeds,
       1193         )
       1194 


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        539             result = self._slow_forward(*input, **kwargs)
        540         else:
    --> 541             result = self.forward(*input, **kwargs)
        542         for hook in self._forward_hooks.values():
        543             hook_result = hook(self, input, result)


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/transformers/modeling_bert.py in forward(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, encoder_hidden_states, encoder_attention_mask)
        804             head_mask=head_mask,
        805             encoder_hidden_states=encoder_hidden_states,
    --> 806             encoder_attention_mask=encoder_extended_attention_mask,
        807         )
        808         sequence_output = encoder_outputs[0]


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        539             result = self._slow_forward(*input, **kwargs)
        540         else:
    --> 541             result = self.forward(*input, **kwargs)
        542         for hook in self._forward_hooks.values():
        543             hook_result = hook(self, input, result)


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/transformers/modeling_bert.py in forward(self, hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
        421 
        422             layer_outputs = layer_module(
    --> 423                 hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask
        424             )
        425             hidden_states = layer_outputs[0]


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        539             result = self._slow_forward(*input, **kwargs)
        540         else:
    --> 541             result = self.forward(*input, **kwargs)
        542         for hook in self._forward_hooks.values():
        543             hook_result = hook(self, input, result)


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/transformers/modeling_bert.py in forward(self, hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
        382         encoder_attention_mask=None,
        383     ):
    --> 384         self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        385         attention_output = self_attention_outputs[0]
        386         outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        539             result = self._slow_forward(*input, **kwargs)
        540         else:
    --> 541             result = self.forward(*input, **kwargs)
        542         for hook in self._forward_hooks.values():
        543             hook_result = hook(self, input, result)


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/transformers/modeling_bert.py in forward(self, hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
        328     ):
        329         self_outputs = self.self(
    --> 330             hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
        331         )
        332         attention_output = self.output(self_outputs[0], hidden_states)


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        539             result = self._slow_forward(*input, **kwargs)
        540         else:
    --> 541             result = self.forward(*input, **kwargs)
        542         for hook in self._forward_hooks.values():
        543             hook_result = hook(self, input, result)


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/transformers/modeling_bert.py in forward(self, hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
        230         encoder_attention_mask=None,
        231     ):
    --> 232         mixed_query_layer = self.query(hidden_states)
        233 
        234         # If this is instantiated as a cross-attention module, the keys


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        539             result = self._slow_forward(*input, **kwargs)
        540         else:
    --> 541             result = self.forward(*input, **kwargs)
        542         for hook in self._forward_hooks.values():
        543             hook_result = hook(self, input, result)


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/modules/linear.py in forward(self, input)
         85 
         86     def forward(self, input):
    ---> 87         return F.linear(input, self.weight, self.bias)
         88 
         89     def extra_repr(self):


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/functional.py in linear(input, weight, bias)
       1370         ret = torch.addmm(bias, input, weight.t())
       1371     else:
    -> 1372         output = input.matmul(weight.t())
       1373         if bias is not None:
       1374             output += bias


    RuntimeError: cublas runtime error : library not initialized at /opt/conda/conda-bld/pytorch_1573049306803/work/aten/src/THC/THCGeneral.cpp:216


Lets find a learning rate to train our classifier head

```python
learn.lr_find(suggestions=True)
```





    torch.Size([2, 512])



    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    ~/fastai2/fastai2/learner.py in fit(self, n_epoch, lr, wd, cbs, reset_opt)
        293                         self.epoch=epoch;          self('begin_epoch')
    --> 294                         self._do_epoch_train()
        295                         self._do_epoch_validate()


    ~/fastai2/fastai2/learner.py in _do_epoch_train(self)
        268             self.dl = self.dls.train;                        self('begin_train')
    --> 269             self.all_batches()
        270         except CancelTrainException:                         self('after_cancel_train')


    ~/fastai2/fastai2/learner.py in all_batches(self)
        246         self.n_iter = len(self.dl)
    --> 247         for o in enumerate(self.dl): self.one_batch(*o)
        248 


    ~/fastai2/fastai2/learner.py in one_batch(self, i, b)
        252             self._split(b);                                  self('begin_batch')
    --> 253             self.pred = self.model(*self.xb);                self('after_pred')
        254             if len(self.yb) == 0: return


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        540         else:
    --> 541             result = self.forward(*input, **kwargs)
        542         for hook in self._forward_hooks.values():


    <ipython-input-19-a7e5dd8ac4a9> in forward(self, input_ids, attention_mask)
          9         print(input_ids.size())
    ---> 10         logits = self.transformer(input_ids, attention_mask = attention_mask)[0]
         11         print(logits.size())


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        540         else:
    --> 541             result = self.forward(*input, **kwargs)
        542         for hook in self._forward_hooks.values():


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/transformers/modeling_bert.py in forward(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels)
       1191             head_mask=head_mask,
    -> 1192             inputs_embeds=inputs_embeds,
       1193         )


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        540         else:
    --> 541             result = self.forward(*input, **kwargs)
        542         for hook in self._forward_hooks.values():


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/transformers/modeling_bert.py in forward(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, encoder_hidden_states, encoder_attention_mask)
        805             encoder_hidden_states=encoder_hidden_states,
    --> 806             encoder_attention_mask=encoder_extended_attention_mask,
        807         )


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        540         else:
    --> 541             result = self.forward(*input, **kwargs)
        542         for hook in self._forward_hooks.values():


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/transformers/modeling_bert.py in forward(self, hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
        422             layer_outputs = layer_module(
    --> 423                 hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask
        424             )


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        540         else:
    --> 541             result = self.forward(*input, **kwargs)
        542         for hook in self._forward_hooks.values():


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/transformers/modeling_bert.py in forward(self, hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
        383     ):
    --> 384         self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        385         attention_output = self_attention_outputs[0]


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        540         else:
    --> 541             result = self.forward(*input, **kwargs)
        542         for hook in self._forward_hooks.values():


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/transformers/modeling_bert.py in forward(self, hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
        329         self_outputs = self.self(
    --> 330             hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
        331         )


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        540         else:
    --> 541             result = self.forward(*input, **kwargs)
        542         for hook in self._forward_hooks.values():


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/transformers/modeling_bert.py in forward(self, hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
        231     ):
    --> 232         mixed_query_layer = self.query(hidden_states)
        233 


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        540         else:
    --> 541             result = self.forward(*input, **kwargs)
        542         for hook in self._forward_hooks.values():


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/modules/linear.py in forward(self, input)
         86     def forward(self, input):
    ---> 87         return F.linear(input, self.weight, self.bias)
         88 


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/nn/functional.py in linear(input, weight, bias)
       1371     else:
    -> 1372         output = input.matmul(weight.t())
       1373         if bias is not None:


    RuntimeError: cublas runtime error : library not initialized at /opt/conda/conda-bld/pytorch_1573049306803/work/aten/src/THC/THCGeneral.cpp:216

    
    During handling of the above exception, another exception occurred:


    RuntimeError                              Traceback (most recent call last)

    <ipython-input-29-35d7aa25ab99> in <module>
    ----> 1 learn.lr_find(suggestions=True)
    

    ~/fastai2/fastai2/callback/schedule.py in lr_find(self, start_lr, end_lr, num_it, stop_div, show_plot, suggestions)
        195     n_epoch = num_it//len(self.dls.train) + 1
        196     cb=LRFinder(start_lr=start_lr, end_lr=end_lr, num_it=num_it, stop_div=stop_div)
    --> 197     with self.no_logging(): self.fit(n_epoch, cbs=cb)
        198     if show_plot: self.recorder.plot_lr_find()
        199     if suggestions:


    ~/fastai2/fastai2/learner.py in fit(self, n_epoch, lr, wd, cbs, reset_opt)
        298 
        299             except CancelFitException:             self('after_cancel_fit')
    --> 300             finally:                               self('after_fit')
        301 
        302     def validate(self, ds_idx=1, dl=None, cbs=None):


    ~/fastai2/fastai2/learner.py in __call__(self, event_name)
        226     def ordered_cbs(self, cb_func): return [cb for cb in sort_by_run(self.cbs) if hasattr(cb, cb_func)]
        227 
    --> 228     def __call__(self, event_name): L(event_name).map(self._call_one)
        229     def _call_one(self, event_name):
        230         assert hasattr(event, event_name)


    ~/fastcore/fastcore/foundation.py in map(self, f, *args, **kwargs)
        360              else f.format if isinstance(f,str)
        361              else f.__getitem__)
    --> 362         return self._new(map(g, self))
        363 
        364     def filter(self, f, negate=False, **kwargs):


    ~/fastcore/fastcore/foundation.py in _new(self, items, *args, **kwargs)
        313     @property
        314     def _xtra(self): return None
    --> 315     def _new(self, items, *args, **kwargs): return type(self)(items, *args, use_list=None, **kwargs)
        316     def __getitem__(self, idx): return self._get(idx) if is_indexer(idx) else L(self._get(idx), use_list=None)
        317     def copy(self): return self._new(self.items.copy())


    ~/fastcore/fastcore/foundation.py in __call__(cls, x, *args, **kwargs)
         39             return x
         40 
    ---> 41         res = super().__call__(*((x,) + args), **kwargs)
         42         res._newchk = 0
         43         return res


    ~/fastcore/fastcore/foundation.py in __init__(self, items, use_list, match, *rest)
        304         if items is None: items = []
        305         if (use_list is not None) or not _is_array(items):
    --> 306             items = list(items) if use_list else _listify(items)
        307         if match is not None:
        308             if is_coll(match): match = len(match)


    ~/fastcore/fastcore/foundation.py in _listify(o)
        240     if isinstance(o, list): return o
        241     if isinstance(o, str) or _is_array(o): return [o]
    --> 242     if is_iter(o): return list(o)
        243     return [o]
        244 


    ~/fastcore/fastcore/foundation.py in __call__(self, *args, **kwargs)
        206             if isinstance(v,_Arg): kwargs[k] = args.pop(v.i)
        207         fargs = [args[x.i] if isinstance(x, _Arg) else x for x in self.pargs] + args[self.maxi+1:]
    --> 208         return self.fn(*fargs, **kwargs)
        209 
        210 # Cell


    ~/fastai2/fastai2/learner.py in _call_one(self, event_name)
        229     def _call_one(self, event_name):
        230         assert hasattr(event, event_name)
    --> 231         [cb(event_name) for cb in sort_by_run(self.cbs)]
        232 
        233     def _bn_bias_state(self, with_bias): return bn_bias_params(self.model, with_bias).map(self.opt.state)


    ~/fastai2/fastai2/learner.py in <listcomp>(.0)
        229     def _call_one(self, event_name):
        230         assert hasattr(event, event_name)
    --> 231         [cb(event_name) for cb in sort_by_run(self.cbs)]
        232 
        233     def _bn_bias_state(self, with_bias): return bn_bias_params(self.model, with_bias).map(self.opt.state)


    ~/fastai2/fastai2/learner.py in __call__(self, event_name)
         23         _run = (event_name not in _inner_loop or (self.run_train and getattr(self, 'training', True)) or
         24                (self.run_valid and not getattr(self, 'training', False)))
    ---> 25         if self.run and _run: getattr(self, event_name, noop)()
         26         if event_name=='after_fit': self.run=True #Reset self.run to True at each end of fit
         27 


    ~/fastai2/fastai2/callback/schedule.py in after_fit(self)
        168         tmp_f = self.path/self.model_dir/'_tmp.pth'
        169         if tmp_f.exists():
    --> 170             self.learn.load('_tmp')
        171             os.remove(tmp_f)
        172 


    ~/fastai2/fastai2/learner.py in load(self, file, with_opt, device, strict)
        372         distrib_barrier()
        373         file = join_path_file(file, self.path/self.model_dir, ext='.pth')
    --> 374         load_model(file, self.model, self.opt, with_opt=with_opt, device=device, strict=strict)
        375         return self
        376 


    ~/fastai2/fastai2/learner.py in load_model(file, model, opt, with_opt, device, strict)
        163     if isinstance(device, int): device = torch.device('cuda', device)
        164     elif device is None: device = 'cpu'
    --> 165     state = torch.load(file, map_location=device)
        166     hasopt = set(state)=={'model', 'opt'}
        167     model_state = state['model'] if hasopt else state


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/serialization.py in load(f, map_location, pickle_module, **pickle_load_args)
        424         if sys.version_info >= (3, 0) and 'encoding' not in pickle_load_args.keys():
        425             pickle_load_args['encoding'] = 'utf-8'
    --> 426         return _load(f, map_location, pickle_module, **pickle_load_args)
        427     finally:
        428         if new_fd:


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/serialization.py in _load(f, map_location, pickle_module, **pickle_load_args)
        611     unpickler = pickle_module.Unpickler(f, **pickle_load_args)
        612     unpickler.persistent_load = persistent_load
    --> 613     result = unpickler.load()
        614 
        615     deserialized_storage_keys = pickle_module.load(f, **pickle_load_args)


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/serialization.py in persistent_load(saved_id)
        574                 obj = data_type(size)
        575                 obj._torch_load_uninitialized = True
    --> 576                 deserialized_objects[root_key] = restore_location(obj, location)
        577             storage = deserialized_objects[root_key]
        578             if view_metadata is not None:


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/serialization.py in restore_location(storage, location)
        441     elif isinstance(map_location, _string_classes):
        442         def restore_location(storage, location):
    --> 443             return default_restore_location(storage, map_location)
        444     elif isinstance(map_location, torch.device):
        445         def restore_location(storage, location):


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/serialization.py in default_restore_location(storage, location)
        153 def default_restore_location(storage, location):
        154     for _, _, fn in _package_registry:
    --> 155         result = fn(storage, location)
        156         if result is not None:
        157             return result


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/serialization.py in _cuda_deserialize(obj, location)
        133             storage_type = getattr(torch.cuda, type(obj).__name__)
        134             with torch.cuda.device(device):
    --> 135                 return storage_type(obj.size())
        136         else:
        137             return obj.cuda(device)


    ~/anaconda3/envs/fastai2_me/lib/python3.7/site-packages/torch/cuda/__init__.py in _lazy_new(cls, *args, **kwargs)
        632     # We need this method only for lazy init, so we can remove it
        633     del _CudaBase.__new__
    --> 634     return super(_CudaBase, cls).__new__(cls, *args, **kwargs)
        635 
        636 


    RuntimeError: CUDA error: device-side assert triggered


```python
learn.recorder.plot_lr_find()
plt.vlines(1.318e-07, 0.6, 1.2)
plt.vlines(0.00524, 0.6, 1.2)
```




    <matplotlib.collections.LineCollection at 0x7f72e459d150>




![png](/images/fasthugs_demo_files/output_59_1.png)


```python
learn.fit_one_cycle(3, lr_max=1e-3, div=10)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.703972</td>
      <td>0.685084</td>
      <td>0.465000</td>
      <td>00:10</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.595791</td>
      <td>0.555975</td>
      <td>0.790000</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.497009</td>
      <td>0.511307</td>
      <td>0.820000</td>
      <td>00:09</td>
    </tr>
  </tbody>
</table>


```python
learn.save('roberta-fasthugs-stg1-1e-3')
```

```python
learn.recorder.plot_loss()
```


![png](/images/fasthugs_demo_files/output_62_0.png)


### Stage 2 Training
And now lets train the full model with differential learning rates

```python
learn.unfreeze()
```

```python
learn.lr_find(suggestions=True)
```








    (1.584893179824576e-05, 0.0008317637839354575)




![png](/images/fasthugs_demo_files/output_65_2.png)


```python
learn.recorder.plot_lr_find()
plt.vlines(1.584e-05, 0.3, 1.0)
plt.vlines(0.0008317, 0.3, 1.0)
```




    <matplotlib.collections.LineCollection at 0x7f72e24b12d0>




![png](/images/fasthugs_demo_files/output_66_1.png)


```python
learn.fit_one_cycle(3, lr_max=slice(1e-6, 3e-5))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.417611</td>
      <td>0.400168</td>
      <td>0.880000</td>
      <td>00:28</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.276386</td>
      <td>0.295171</td>
      <td>0.900000</td>
      <td>00:28</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.234433</td>
      <td>0.282703</td>
      <td>0.900000</td>
      <td>00:29</td>
    </tr>
  </tbody>
</table>


```python
learn.save('roberta-fasthugs-stg2-3e-5')
```

```python
learn.recorder.plot_loss()
```


![png](/images/fasthugs_demo_files/output_69_0.png)


## Lets Look at the model's predictions

```python
learn.predict("This was a good movie")
```








    ('positive', tensor(1), tensor([0.1585, 0.8415]))



```python
from fastai2.interpret import *
interp = Interpretation.from_learner(learn)
```





```python
interp.plot_top_losses(5)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>input</th>
      <th>target</th>
      <th>predicted</th>
      <th>probability</th>
      <th>loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;s&gt; ĠIn Ġ17 th ĠCentury ĠJapan , Ġthere Ġlived Ġa Ġsamurai Ġwho Ġwould Ġset Ġthe Ġstandard Ġfor Ġthe Ġages . ĠHis Ġname Ġwas ĠMay eda . ĠHe Ġis Ġsent Ġon Ġan Ġepic Ġjourney Ġacross Ġthe Ġworld Ġto Ġacquire Ġ5 , 000 Ġmus cats Ġfrom Ġthe ĠKing Ġof ĠSpain . ĠWhilst Ġat Ġsea Ġa Ġviolent Ġstorm Ġswall ows Ġtheir Ġprecious Ġgold Ġintended Ġto Ġbuy Ġthe Ġweapons Ġand Ġalmost Ġtakes Ġtheir Ġlives . ĠMay eda Ġmust Ġbattle Ġall Ġodds Ġto Ġsurvive Ġand Ġthe Ġsecure Ġthe Ġfate Ġof Ġhis Ġbeloved ĠJapan . ĠShogun ĠMay eda Ġis Ġa Ġmulti Ġmillion Ġdollar Ġaction Ġadventure Ġepic Ġset Ġacross Ġthree Ġcontinents .&lt; br Ġ/ &gt;&lt; br Ġ/&gt; Star ring Ġcinema Ġlegends ĠSho ĠKos ugi Ġ( T ench u : ĠStealth ĠAssassins ), ĠChristopher ĠLee Ġ( Star ĠWars , ĠLord Ġof Ġthe ĠRings ĠTrilogy ), ĠJohn ĠRh ys ĠDavies Ġ( Lord Ġof Ġthe ĠRings ĠTrilogy , ĠIndiana ĠJones</td>
      <td>negative</td>
      <td>positive</td>
      <td>0.9690422415733337</td>
      <td>3.475131034851074</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&lt;s&gt; ĠI 'm Ġgonna Ġtip Ġthe Ġscales Ġhere Ġa Ġbit Ġand Ġsay ĠI Ġenjoyed Ġthis . ĠHowever , Ġthe Ġcartoon Ġis Ġreally Ġonly Ġgoing Ġto Ġappeal Ġto Ġthose Ġwho Ġhave Ġvery Ġabsurd ist Ġtendencies . ĠIt 's Ġdefinitely Ġsomething Ġthat Ġmost Ġpeople Ġwill Ġnot Ġget , Ġas Ġis Ġthe Ġnature Ġof Ġabsurd ism .&lt; br Ġ/ &gt;&lt; br Ġ/&gt; the Ġanimation Ġis Ġhorrible , Ġbut Ġyes , Ġthat 's Ġthe Ġpoint . ĠThe Ġmain Ġcharacter Ġis Ġfoul Ġm out hed , Ġviolent , Ġand Ġstupid . Ġno Ġredeem ing Ġqualities Ġwhatsoever . Ġhis Ġwife Ġshri eks Ġand Ġw ails , Ġapparently Ġjust Ġbarely Ġcapable Ġof Ġthe Ġmost Ġbasic Ġcommunication Ġskills . Ġmost Ġof Ġthese Ġstories Ġcompletely Ġlack Ġany Ġkind Ġof Ġpoint .&lt; br Ġ/ &gt;&lt; br Ġ/&gt; but Ġagain , Ġthat 's Ġthe Ġpoint Ġ;) &lt; br Ġ/ &gt;&lt; br Ġ/&gt; If Ġnon Ġsequ it ers , Ġfoul Ġlanguage ,</td>
      <td>positive</td>
      <td>negative</td>
      <td>0.9560778737068176</td>
      <td>3.125335693359375</td>
    </tr>
    <tr>
      <th>2</th>
      <td>&lt;s&gt; ĠIn Ġorder Ġto Ġhold Ġthe Ġpublic 's Ġattention Ġfor Ġthree Ġhours , Ġwe Ġwere Ġtreated Ġnot Ġso Ġmuch Ġto Ġa Ġfamily 's Ġr omp Ġthrough Ġfour Ġgenerations Ġand Ġ120 Ġyears Ġof ĠHungarian Ġhistory , Ġas Ġto Ġsexual Ġlia isons Ġwith Ġa Ġsister , Ġa Ġsister - in - law Ġand Ġother Ġadul ter ies . ĠOh Ġyes , Ġthere Ġwas Ġalso Ġa Ġtotally Ġgrat uitous Ġrape . ĠHaving Ġsaid Ġall Ġthis , Ġthe Ġfirst Ġstory Ġof Ġthe Ġrelationship Ġamong Ġthe Ġchildren Ġof Ġthe Ġpatriarch Ġwas Ġfresh Ġand Ġsens ual Ġ- Ġthanks Ġto ĠJennifer ĠEh le . &lt;/s&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt; &lt;pad&gt;</td>
      <td>negative</td>
      <td>positive</td>
      <td>0.9183984398841858</td>
      <td>2.505906581878662</td>
    </tr>
    <tr>
      <th>3</th>
      <td>&lt;s&gt; ĠThis Ġmovie Ġis Ġhorrible - Ġin Ġa Ġ' so Ġbad Ġit 's Ġgood ' Ġkind Ġof Ġway .&lt; br Ġ/ &gt;&lt; br Ġ/&gt; The Ġstoryline Ġis Ġre h ashed Ġfrom Ġso Ġmany Ġother Ġfilms Ġof Ġthis Ġkind , Ġthat ĠI 'm Ġnot Ġgoing Ġto Ġeven Ġbother Ġdescribing Ġit . ĠIt 's Ġa Ġsword / s or cery Ġpicture , Ġhas Ġa Ġkid Ġhoping Ġto Ġrealize Ġhow Ġimportant Ġhe Ġis Ġin Ġthis Ġworld , Ġhas Ġa Ġ" nom adic " Ġadventurer , Ġan Ġevil Ġaide / s orce rer , Ġa Ġprincess , Ġa Ġhairy Ġcreature .... you Ġget Ġthe Ġpoint .&lt; br Ġ/ &gt;&lt; br Ġ/&gt; The Ġfirst Ġtime ĠI Ġcaught Ġthis Ġmovie Ġwas Ġduring Ġa Ġvery Ġharsh Ġwinter . ĠI Ġdon 't Ġknow Ġwhy ĠI Ġdecided Ġto Ġcontinue Ġwatching Ġit Ġfor Ġan Ġextra Ġfive Ġminutes Ġbefore Ġturning Ġthe Ġchannel , Ġbut Ġwhen ĠI Ġcaught Ġsite Ġof ĠGulf ax</td>
      <td>positive</td>
      <td>negative</td>
      <td>0.8572472333908081</td>
      <td>1.9466404914855957</td>
    </tr>
    <tr>
      <th>4</th>
      <td>&lt;s&gt; ĠThose Ġwho Ġhave Ġgiven Ġthis Ġproduction Ġsuch Ġa Ġlow Ġrating Ġprobably Ġhave Ġnever Ġseen Ġthe Ġcelebrated ĠGeorge ĠBal anch ine Ġproduction Ġlive Ġonstage , Ġor Ġare Ġletting Ġtheir Ġdisdain Ġfor Ġthe Ġstar Ġcasting Ġof ĠMac aul ay ĠCul kin Ġinfluence Ġtheir Ġjudgement . ĠThe ĠAtlanta ĠBal let Ġwas Ġfortunate Ġenough , Ġfrom Ġthe Ġ1960 's Ġto Ġthe Ġ1980 's , Ġto Ġbe Ġthe Ġfirst Ġballet Ġcompany Ġauthorized Ġto Ġstage Ġthis Ġproduction Ġother Ġthan Ġthe ĠNew ĠYork ĠCity ĠBal let , Ġand ĠI Ġhave Ġseen Ġit Ġlive Ġonstage Ġseveral Ġtimes . ĠI Ġcan Ġassure Ġreaders Ġthat Ġthe Ġfilm Ġis Ġa Ġquite Ġaccurate Ġrendering Ġof Ġthis Ġproduction , Ġand Ġthat Ġthe Ġuse Ġof Ġa Ġchild Ġwith Ġlimited Ġdancing Ġabilities Ġin Ġthe Ġtitle Ġrole Ġis Ġnot Ġa Ġcheap Ġstunt Ġdreamed Ġup Ġto Ġshowcase ĠCul kin ; Ġit Ġwas ĠBal anch ine 's Ġidea Ġto Ġuse Ġa Ġchild Ġin Ġthis Ġrole , Ġjust</td>
      <td>positive</td>
      <td>negative</td>
      <td>0.8415194749832153</td>
      <td>1.8421235084533691</td>
    </tr>
  </tbody>
</table>

