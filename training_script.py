from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor,WhisperTokenizer
from datasets import load_dataset
import torch
from jiwer import wer
from tqdm import tqdm
import pickle


STARTING_STEP = 0
MODEL = "openai/whisper-small.en"
RESET_METRICS = True
SAVE_TO = "models/"
TRAIN_DATASET = "train.100"
SAVE_EVERY = 3000
GRADIENT_ACCUMULATION_STEPS = 64
EVALUATION_SIZE = 150
LEARNING_RATE = 1e-4
EPOCHS = 1


model = WhisperForConditionalGeneration.from_pretrained(MODEL)
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small.en")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small.en")
dataset = load_dataset("librispeech_asr","clean")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = dataset.remove_columns(["file","speaker_id","id","chapter_id"])
train_dataset = dataset[TRAIN_DATASET].shuffle(seed=42).select(range(STARTING_STEP,len(dataset[TRAIN_DATASET])))
val_dataset = dataset['validation'].shuffle(seed=42)



model.get_encoder().requires_grad_(False)
model.get_decoder().requires_grad_(True)
model.proj_out.requires_grad_(True)




def preprocess(data):
    processed_data = {}
    processed_data['input_features'] = feature_extractor(data['audio']['array'], sampling_rate=16_000, return_tensors="pt")['input_features'].to(device)
    processed_data['decoder_input_ids'] = tokenizer('<|startoftranscript|>'+data['text'],return_tensors='pt').input_ids.to(device)
    processed_data['labels'] = tokenizer(data['text']+'<|endoftext|>',return_tensors='pt').input_ids.to(device)
    return processed_data


def evaluate(model,dataset):
    print("Evaluating...")
    with torch.no_grad():
        wer_scores = []
        losses = []
        for item in tqdm(dataset):
            
            input_features = feature_extractor(item['audio']['array'], sampling_rate=16_000, return_tensors="pt")['input_features'].to(device)
            generated_ids = model.generate(input_features, decoder_input_ids=tokenizer('<|startoftranscript|>',return_tensors='pt').input_ids.to(device), max_length=100)
            generated_test_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            wer_scores.append(wer(item['text'].lower(),generated_test_text))

            loss = model(**preprocess(item)).loss.item()
            losses.append(loss)

    return sum(losses)/len(losses),sum(wer_scores)/len(wer_scores)
        

if RESET_METRICS:
    with open(f'{SAVE_TO}losses.pkl','wb') as f:
        losses = []
        pickle.dump(losses,f)

    with open(f'{SAVE_TO}val_losses.pkl','wb') as  f:
        val_losses = []
        pickle.dump(val_losses,f)

    with open(f'{SAVE_TO}val_wers.pkl','wb') as f:
        val_wers = []
        pickle.dump(val_wers,f)


with open(f'{SAVE_TO}losses.pkl','rb') as f:
    losses = pickle.load(f)

with open(f'{SAVE_TO}val_losses.pkl','rb') as  f:
    val_losses = pickle.load(f)

with open(f'{SAVE_TO}val_wers.pkl','rb') as f:
    val_wers = pickle.load(f)

model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)


steps,gradient_steps,loss = 0,0,0
val_loss,val_wer = evaluate(model,val_dataset.shuffle().select(range(0,EVALUATION_SIZE)))
for Epoch in range(EPOCHS):
    print("Epoch: ",Epoch+1)
    training_loop = tqdm(train_dataset)
    for item in training_loop:

        loss = model(**preprocess(item)).loss
        loss.backward()
        losses.append(loss.item())
        steps+=1


        if steps%GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            gradient_steps+=1
        

        if steps%SAVE_EVERY == 0:
            
            val_loss,val_wer = evaluate(model,val_dataset.shuffle().select(range(0,EVALUATION_SIZE)))
            val_losses.append(val_loss)
            val_wers.append(val_wer)
            
            model.save_pretrained(f'{SAVE_TO}whisper-{steps}')

            with open(f'{SAVE_TO}losses.pkl','wb') as f:
                pickle.dump(losses,f)

            with open(f'{SAVE_TO}val_losses.pkl','wb') as f:
                pickle.dump(val_losses,f)

            with open(f'{SAVE_TO}val_wers.pkl','wb') as f:
                pickle.dump(val_wers,f)

        training_loop.set_description(f"Steps: {steps}   Gradient Updates: {gradient_steps}   Loss: {loss.item():.4f}   Val Loss: {val_loss:.4f}   Val WER: {val_wer:.4f}")

            


