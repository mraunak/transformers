"""
Demo of implementation of a new method for fine-tuning transformer models that we call as 'IGF'
on WikiText data set and compared the results with the standard fine-tuning method

steps followed in the demo

1) Generate a objective dataset of pairs (X, IG(X)). IG(X)--Informativeness of context 'X'
Our IG (information gain) model is learning predict the ‘informativeness’ of a particular context
Informativeness is the change in perplexity between the model’s accuracy on an objective set
before and after seeing that context

2) A secondary learner is created to infer a function approximation for IG model using the dataset created in (1).
3) The learner created in (2) is used to inform the fine-tuning process.
`
We then generate a plot comparing the performance of IGF compared to standard fine-tuning without any context filtering

"""

# Prerequisite libraries:

import joblib
import numpy as np
import torch
from transformers import GPT2LMHeadModel
from typing import Optional
from igf.igf import *
import argparse



#Collecting *n* pairs for training the secondary learner
def generate_n_pairs(context_len = 32, n_pairs = 10, number = 100, min_len = 1026, trim = True,
                    data_file = 'data/tokenized_stories_train_wikitext103.jbl',
                    igf_data_file = 'igf_context_pairs.jbl'
            ):
    # generates same data everytime
    set_seed(3)
    # generate train_data and objective_set
    train_data, objective_set = generate_datasets(context_len, data_file, number, min_len, trim)
    # keeps model same across runs
    set_seed(4)
    # model, lm_optimizer, lm_scheduler = recopy_gpt2(model, device, n_pairs) # store original model weights
    # can we train on GPU?
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load pretrained model
    model = load_gpt2('gpt2').to(device)
    print('computing perplexity on objective set')
    orig_perp = compute_perplexity(model, objective_set, context_len).item()
    print('perplexity on objective set:', orig_perp)

    # collect igf pairs and save to file demo.jbl
    collect_objective_set(model, orig_perp, context_len, train_data, objective_set, n_pairs, device, igf_data_file)

    # clean up, delete model and data we don't need anymore
    del model, train_data, objective_set
    torch.cuda.empty_cache()


# Train the secondary learner
def training_secondary_learner(max_epochs=15, batch_size=128, eval_freq=100,
                               igf_model_path='igf_model.pt', igf_data_file='data/IGF_values.jbl'):
    set_seed(42)
    # Load model
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Initialize secondary learner to use embedding weights of model
    secondary_learner = SecondaryLearner(model)

    # Train secondary learner
    secondary_learner = train_secondary_learner(secondary_learner, train_data,
                                                max_epochs=15, batch_size=128, eval_freq=100,
                                                igf_model_path=igf_model_path)

    del model, train_data
    torch.cuda.empty_cache()

    return secondary_learner


def finetune(model, train_dataset, test_dataset, context_len=32,
             max_steps=1000, batch_size=16, threshold=1.0, recopy_model=recopy_gpt2,
             secondary_learner=secondary_learner, eval_interval=10,
             finetuned_model_name='gpt2_finetuned.pt'):
    # finetune with IGF if secondary_learner is not None, else standard finetuning
    # Initialize the model

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler)

    num_train_epochs = max_steps // (len(train_dataset)) + 1
    global_step = 0
    context = torch.zeros((1, context_len), dtype=torch.long, device=device)
    model, lm_optimizer, lm_scheduler = recopy_model(model, device, max_steps)

    model.train()
    if secondary_learner is not None:
        secondary_learner.to(device)
        secondary_learner.eval()
    contexts = []
    examples = 0

    observed_qs = []
    test_perps = []
    repeats = []
    val_perps = []

    # Compute the performance of the transformer model at the beginning
    real_perp = compute_perplexity(model, test_dataset, context_len)
    test_perps.append(real_perp)
    print('Test perplexity, step', global_step, ':', real_perp)
    for epoch in range(int(num_train_epochs)):
        for step, example in enumerate(train_dataloader):
            torch.cuda.empty_cache()
            start = random.randint(0, example.size(2) - context_len - 1)
            context[0, :] = example[0, 0, start:start + context_len]
            lm_optimizer.zero_grad()
            outputs = model(context, labels=context)
            do_backprop = True

            if secondary_learner is not None:
                predicted_Q = secondary_learner.forward(
                    torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0))[0].item()
                observed_qs.append(float(predicted_Q))

                # Here we implement the simple non-constant threshold for the predicted IG(X) value
                # We will decay the selectivity of our secondary learner filter from
                # 1 standard deviation above average to 1 below average after 10 batches.

                if global_step == 10:
                    threshold = -1
                if predicted_Q < threshold:
                    do_backprop = False

            # If we passed the filter, add the context to the batch!
            if do_backprop:
                contexts.append(np.array(context.cpu()))
                lm_loss = outputs[0]
                lm_loss.backward()
                examples += 1

            del outputs

            # Once the batch is filled with enough contexts, backprop on the batch.
            if examples == batch_size:
                torch.cuda.empty_cache()
                examples = 0
                # Do LM backprop
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                lm_optimizer.step()
                lm_scheduler.step()  # Update learning rate schedule
                global_step += 1
                # Compute the performance of the transformer model at this batch
                if global_step % eval_interval == 0:
                    real_perp = compute_perplexity(model, test_dataset, context_len)
                    test_perps.append(real_perp)

                    print('Test perplexity, step', global_step, ':', real_perp)
            # Break out of the loop after 60 batches
            if max_steps > 0 and global_step > 60:
                break
        if max_steps > 0 and global_step > 60:
            break

    # save finetuned transformer model
    torch.save(model.state_dict(), finetuned_model_name)
    torch.cuda.empty_cache()
    # Do some cleaning up so we can reinitialize for the next run of this function
    del lm_optimizer
    del lm_scheduler
    return model, observed_qs, test_perps, val_perps, contexts, repeats


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain data files for WikiText.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Where do you want to store the pre-trained models downloaded from huggingface.co",
    )

    parser.add_argument(
        "--n_pairs",
        default=10,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--context_len",
        default=32,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_steps", default=1000, type=int,
        help="To calculate training epochs")

    parser.add_argument(
        "--runs",
        default=1,
        type=int,
        help=".",
    )

    parser.add_argument(
        "--eval_interval",
        default=10,
        type=int,
        help="decay the selectivity of our secondary learner filter from"
             "1 standard deviation above average to 1 below average after 10 batches")

    parser.add_argument(
        "--number_in_test",
        default=100,
        type=int,
        help="The number of examples used as test_data")

    parser.add_argument(
        "--min_len",
        default=1026,
        type=int,
        help="The minimum length of the article to be used as objective set")

    parser.add_argument(
        "--trim",
        default=True,
        type=bool,
        help="The minimum length of the article to be used as objective set")


    # Load model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    set_seed(seed)
    train_data, test_data = generate_datasets(context_len=32, data_file='data/tokenized_stories_train_wikitext103.jbl',
                                              number_in_test=100, min_len=1026, trim=True)

    generate_n_pairs(context_len=32, n_pairs=10, number=100, min_len=1026, trim=True,
                     data_file='data/tokenized_stories_train_wikitext103.jbl',
                     igf_data_file='igf_context_pairs.jbl'
                     )

    secondary_learner = training_secondary_learner(max_epochs=15, batch_size=128, eval_freq=100,
                                                   igf_model_path='igf_model.pt', igf_data_file='data/IGF_values.jbl')

    finetune(model, train_data, test_data, context_len=32,
             max_steps=1000, batch_size=16, threshold=1.0, recopy_model=recopy_gpt2,
             secondary_learner=secondary_learner, eval_interval=10,
             finetuned_model_name='gpt2_finetuned.pt')


if __name__ == '__main__':
    main()

