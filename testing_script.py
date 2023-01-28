from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor,WhisperTokenizer
from datasets import load_dataset
import torch
from tqdm import tqdm
from jiwer import wer

SIZE = -1

models_to_test = [
    "whisper-fine-tuned",
    "openai/whisper-small.en",
]

process_output = [
    True,
    True
]


test_dataset = load_dataset("librispeech_asr","clean",split="test")

test_dataset = test_dataset.remove_columns(["file",'speaker_id','id','chapter_id'])


feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small.en")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small.en")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decoder_input_ids = tokenizer("<|startoftranscript|>", return_tensors="pt").input_ids.to(device)


test_dataset = test_dataset.shuffle(seed=42).select(range(0,SIZE if SIZE != -1 else len(test_dataset)))


with torch.no_grad():
    for m, model_to_test in enumerate(models_to_test):
        print(f"Testing model: {model_to_test}")
        model = WhisperForConditionalGeneration.from_pretrained(model_to_test)
        model.to(device)
        model.eval()
        wer_scores = []
        dataset_loop = tqdm(range(len(test_dataset)))
        for i in dataset_loop:

            t = test_dataset[i]
            input_features = feature_extractor(t['audio']['array'], sampling_rate=16_000, return_tensors="pt")['input_features'].to(device)
            generated_ids = model.generate(input_features, decoder_input_ids=decoder_input_ids, max_length=100)
            generated_test_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            if process_output[m]:
                generated_test_text = generated_test_text.replace('.','').replace(',','').replace('?','').replace('!','').lower()

            wer_scores.append(wer(t['text'].lower(),generated_test_text))
            dataset_loop.set_description(f"WER: {wer_scores[-1]}")



        

        print(f"WER score for {model_to_test}: {sum(wer_scores)/len(wer_scores)}")





