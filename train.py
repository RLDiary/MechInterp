import datasets
from gpt2 import GPT2, ModelConfig, GenerationConfig, TransformerSampler
import wandb
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence

from dataclasses import dataclass
from transformers import GPT2Tokenizer
from tqdm import tqdm

@dataclass
class TrainingConfig:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_ctx = 1024
    batch_size = 6
    epochs = 1
    lr: float = 1e-3
    weight_decay: float = 1e-2
    wandb_project: str | None = "training_gpt2"
    wandb_name: str | None = None
    pad_token_id: int = 0

class DynamicPaddingCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        
        input_ids = [torch.tensor(sample['input_ids']) for sample in batch]
        attention_mask = [torch.tensor(sample['attention_mask']) for sample in batch]
        
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id, padding_side='left')
        
        attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0.0, padding_side='left')
        
        return {
            'input_ids': input_ids_padded,
            'attention_mask': attention_mask_padded
        }

def apply_chat_template(sample, tokenizer):
        text = (
            tokenizer.eos_token +
            "User: " + sample["prompt"] + tokenizer.eos_token + '\n\n'
            "Assistant: " + sample["text"] + tokenizer.eos_token
        )
        return text

def load_format_story_dataset(tokenizer):
    story_ds = datasets.load_dataset("/home/ubuntu/MechInter/GPT-2/datasets/children-stories", split="train")
    
    def prepare_story_dataset(ds, tokenizer):

        def format_and_tokenize(sample):
            text = apply_chat_template(sample, tokenizer)
            tokens = tokenizer(text, truncation=True, padding=False)
            return {"input_ids": tokens['input_ids'], "attention_mask": tokens['attention_mask']}
        
        ds = ds.map(
            format_and_tokenize, 
            num_proc=16,
            remove_columns=ds.column_names,
            desc="Formatting and tokenizing",
            cache_file_name="/home/ubuntu/MechInter/GPT-2/datasets/children-stories/cache.arrow",
            load_from_cache_file=True,
            writer_batch_size=50000
        )

    return prepare_story_dataset(story_ds, tokenizer)

def get_sample_prompts(tokenizer):
    file_path = '/home/ubuntu/MechInter/GPT-2/datasets/children-stories/Children-Stories-9-Final.json'
    sample_prompts = datasets.load_dataset("json", data_files=file_path, split="train")
    sample_prompts = [apply_chat_template(sample_prompts[i], tokenizer) for i in range(2)]
    return sample_prompts

def get_model_and_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    training_config = TrainingConfig()
    training_config.pad_token_id = tokenizer(tokenizer.pad_token)['input_ids'][0]

    model_cfg = ModelConfig()
    model_cfg.vocab_size = tokenizer.vocab_size

    model = GPT2(model_cfg).to(training_config.device)
    return model, tokenizer, model_cfg, training_config

class Trainer():
    def __init__(self,
    model_cfg: ModelConfig,
    training_config: TrainingConfig,
    model: GPT2,
    tokenizer: GPT2Tokenizer,
    use_wandb: bool = True,
    sample_prompts: list[str] = None):

        self.training_config = training_config
        self.model_cfg = model_cfg
        self.gen_cfg = GenerationConfig()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=training_config.lr, weight_decay=training_config.weight_decay)
        self.data_collator = DynamicPaddingCollator(training_config.pad_token_id)
        self.sampler = TransformerSampler(self.model_cfg, self.gen_cfg, model = self.model, tokenizer = self.tokenizer)
        self.sample_prompts = sample_prompts
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(
                project="gpt2-training",
                name="tuning-training-code",
                config={
                    "learning_rate": training_config.lr,
                    "weight_decay": training_config.weight_decay,
                    "epochs": training_config.epochs,
                    "batch_size": training_config.batch_size,
                }
            )
            wandb.watch(self.model, log="all", log_freq=100)
        self.current_step = 0
        os.makedirs("GPT-2/Checkpoints", exist_ok=True)

    def step(self, batch: dict):
        input_ids = batch['input_ids'].to(self.training_config.device)
        attention_mask = batch['attention_mask'].to(self.training_config.device)
        logits = self.model.forward(input_ids, attention_mask)
        loss = self.compute_loss(logits, input_ids, attention_mask)
        return loss
    
    def sample_completions(self, prompts, max_new_tokens: int = 100):
        prompts = [p[:100] for p in prompts]
        tokens = self.tokenizer(prompts, return_tensors='pt', truncation = True, padding = True, padding_side = 'left')
        for _ in range(max_new_tokens):
            outputs = self.sampler.forward(tokens)
            prompts = [prompt + output for prompt, output in zip(prompts, outputs)]
        
        if self.use_wandb:
            samples_table = wandb.Table(columns=["step", "sample_id", "completion"])
            for i, completion in enumerate(prompts):
                samples_table.add_data(self.current_step, i, completion)
            wandb.log({"sample_completions": samples_table}, step=self.current_step)
        else:
            for prompt in prompts:
                print(prompt)
                print('****************')
    
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
        # if self.use_wandb:
        #     artifact = wandb.Artifact('model', type='model')
        #     artifact.add_file(path)
        #     wandb.log_artifact(artifact)


    def train(self, train_dataset: datasets.Dataset, val_dataset: datasets.Dataset):
        train_dataloader = DataLoader(train_dataset, batch_size=self.training_config.batch_size, shuffle=True, collate_fn=self.data_collator, num_workers=16, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.training_config.batch_size, shuffle=True, collate_fn=self.data_collator, num_workers=16, pin_memory=True)
        
        total_steps = len(train_dataloader) * self.training_config.epochs
        progress_bar = tqdm(total=total_steps, desc='Training')

        # Model weights save intervals
        checkpoint_intervals = [int(total_steps * p) for p in [0.2, 0.4, 0.6, 0.8, 1.0]]
        next_checkpoint_idx = 0
        
        for epoch in range(self.training_config.epochs):
            total_loss = 0

            for idx, batch in enumerate(train_dataloader):
                
                loss = self.step(batch)
                total_loss += loss.item()
                loss.backward()

                if self.use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/epoch": epoch,
                        "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                    }, step=self.current_step)
                
                self.current_step += 1

                if next_checkpoint_idx < len(checkpoint_intervals) and self.current_step >= checkpoint_intervals[next_checkpoint_idx]:
                    progress_pct = int((next_checkpoint_idx + 1) * 20)
                    checkpoint_path = f"GPT-2/Checkpoints/model_checkpoint_{progress_pct}pct_step_{self.current_step}.pt"
                    self.save_model(checkpoint_path)
                    next_checkpoint_idx += 1

                progress_bar.update(1)
                progress_bar.set_postfix({'epoch': epoch,'train_loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss/(idx+1):.4f}'})
                
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if idx % 100 == 0:
                    val_loss = self.evaluate(val_dataloader)
                    progress_bar.set_postfix({'epoch': epoch, 'train_loss': f'{loss.item():.4f}', 'val_loss': f'{val_loss:.4f}'})
                    self.sample_completions(prompts = self.sample_prompts)
                    
                    if self.use_wandb:
                        wandb.log({"val/loss": val_loss,}, step=self.current_step)
                    

            
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
        
        progress_bar.close()
        if self.use_wandb:
            wandb.finish()
        
        print("Training completed!")
    
    def compute_loss(self, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        gathered_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        
        masked_log_probs = gathered_log_probs * shift_mask
        loss = -masked_log_probs.sum() / shift_mask.sum()
        
        return loss
    
    def evaluate(self, val_dataloader: DataLoader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                total_loss += self.step(batch).item()
        avg_loss = total_loss / len(val_dataloader)
        print(f"Validation Loss: {avg_loss:.4f}")
        self.model.train()
        return avg_loss
    
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)



def main():
    model, tokenizer, model_cfg, training_config = get_model_and_tokenizer()
    story_ds = load_format_story_dataset(tokenizer)
    sample_prompts = get_sample_prompts(tokenizer)
    trainer = Trainer(model_cfg, training_config, model, tokenizer, sample_prompts)
    val_len = 200
    train_len = len(story_ds) - val_len
    train_ds, val_ds = random_split(story_ds, [train_len, val_len])
    trainer.train(train_ds, val_ds)

if __name__ == '__main__':
    main()