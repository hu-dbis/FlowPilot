import torch
from transformers import T5ForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig, T5Config, AutoModel
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from sys import platform
import torch.distributed as dist
import warnings
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`.*")
from transformers import logging
logging.set_verbosity_error()

class T5P:
    def __init__(self):

        model_name = 'Salesforce/codet5p-16b'
        # model_name = '../../outputs/fine_tuned_codet5p_ud'
        if platform == "darwin":
            torch.set_num_threads(12)
            self.device = torch.device("mps")
        if platform == "linux" or platform == "linux2":
            # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # dist.init_process_group("nccl")
            # rank = dist.get_rank()
            # torch.cuda.set_device(rank)
            # self.device = torch.device(f"cuda:{rank}")
            self.device = torch.device("cuda")

        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                                                config=self.config,
                                                           torch_dtype=torch.float16,
                                                           low_cpu_mem_usage=True,
                                                           trust_remote_code=True).to(self.device)

        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs!")
        #     # self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[rank])
        #     self.model = torch.nn.DataParallel(self.model)



    def recommend(self, code):
        encoding = self.tokenizer(code, return_tensors="pt").to(self.device)
        encoding['decoder_input_ids'] = encoding['input_ids'].clone()
        outputs = self.model.generate(**encoding, max_length=len(code)+200)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def chat(self, code, length):
        encoding = self.tokenizer(code, return_tensors="pt").to(self.device)
        encoding['decoder_input_ids'] = encoding['input_ids'].clone()
        outputs = self.model.generate(**encoding, max_length=length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def finetune(self, data, model_suffix):

        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(data)

        # Tokenize
        def preprocess_function(examples):
            inputs = examples["input"]
            targets = examples["output"]
            model_inputs = self.tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
            labels = self.tokenizer(targets, max_length=512, truncation=True, padding="max_length")
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_dataset = dataset.map(preprocess_function, batched=True)

        # Split dataset
        train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split["train"]
        val_dataset = train_test_split["test"]

        # Data collator for padding
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=data_collator)
        val_dataloader = DataLoader(val_dataset, batch_size=8, collate_fn=data_collator)


        training_args = Seq2SeqTrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            save_total_limit=2,
            predict_with_generate=True,
            fp16=False,
            no_cuda=True,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        trainer.train()

        # Save the model
        self.model.save_pretrained(f"../../outputs/fine_tuned_codet5p_{model_suffix}")
        self.tokenizer.save_pretrained(f"../../outputs/fine_tuned_codet5p_{model_suffix}")