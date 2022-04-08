import copy
import glob
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
from transformers import (AdamW, GPT2LMHeadModel)
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler)
from tqdm import tqdm
from scipy.stats import ttest_ind
import logging


logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_perplexity(model:'Pre-trained GPT2 model', test_data, context_len):
    """Computes perplexity of the transformer model on data in test_data"""

    model.eval()
    device = next(model.parameters()).device
    eval_batch_size = 1
    context = torch.zeros((eval_batch_size, context_len), dtype=torch.long, device=device)
    eval_dataloader = DataLoader(test_data, shuffle=False, batch_size=eval_batch_size)
    eval_loss = torch.zeros(1, device=device)
    nb_eval_examples = 0
    for batch in eval_dataloader:
        batch.to(device)
        # pad
        context.zero_()
        for i in range(eval_batch_size):
            context[i, :] = batch[i]
        outputs = model(context, labels = context)
        eval_loss += outputs[0].sum().item()
        nb_eval_examples += batch.size(0)
    eval_loss = eval_loss / nb_eval_examples
    perplexity = torch.exp(eval_loss)
    # perplexity = torch.exp(torch.tensor(eval_loss, device=device))
    model.train()
    return perplexity


def load_gpt2(model_name='gpt2'):
    # load original gpt2 and save off for quicker loading
    model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True)
    torch.save(model.state_dict(), model_name + 'local.pt')
    return model


def recopy_gpt2(orig_model, device, max_steps):
    # Reset the model to the original pretrained GPT-2 weights after each iteration
    model = copy.deepcopy(orig_model)
    model.to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    lm_optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
    lm_scheduler = get_linear_schedule_with_warmup(lm_optimizer, 0, max_steps)
    torch.cuda.empty_cache()
    return model, lm_optimizer, lm_scheduler


def intermittent_save(contexts, real_perps, past_perps, filename):
    # save the perplexity differences to filename
    avg = np.array(real_perps).mean()
    std = np.array(real_perps).std()
    perp_diff = (real_perps - avg)/std
    data_final = list(zip(contexts, perp_diff, past_perps))
    joblib.dump(data_final, filename)


def collect_objective_set(model, orig_perp, context_len, train_data, objective_set, max_steps,
                          device, filename='dev.jbl', recopy_model=recopy_gpt2):
    """ Collect individual IGF values from pre-trained transformer model
    max_steps samples of training data to train secondary model"""

    # initialize variables to record relevant information
    contexts = []
    real_perps = []
    past_perps = []

    # Initialize the transformer model
    orig_model = copy.deepcopy(model)
    orig_model.to(device='cpu')
    torch.cuda.empty_cache()

    # Compute perplexity of initial transformer model for comparison
    model.train()
    model, lm_optimizer, lm_scheduler = recopy_model(orig_model, device, max_steps)

    for step in tqdm(range(max_steps)):
        context = torch.zeros((1, context_len), dtype=torch.long, device=device)
        story = random.choice(train_data)
        start = random.randint(0, len(story[0]) - context_len - 1)
        context[0, :] = story[0][start:start + context_len]
        lm_optimizer.zero_grad()
        outputs = model(context, labels=context)
        lm_loss = outputs[0]
        past_perp = compute_perplexity(model, context, context_len)
        model.train()
        lm_loss.backward()
        # Do LM backprop
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        lm_optimizer.step()
        lm_scheduler.step()  # Update learning rate schedule

        # Compute perplexity after backpropogating on the selected context
        real_perp = compute_perplexity(model, objective_set, context_len)

        # Periodically save the stored (X, IG(X)) pairs
        if step % 1000 == 0 and step > 1:
            intermittent_save(contexts, real_perps, past_perps, filename)

        # Reset the pretrained model to the original pretrained GPT-2 weights after each iteration
        model, lm_optimizer, lm_scheduler = recopy_model(orig_model, device, max_steps)

        past_perps.append(past_perp.item())
        real_perps.append(orig_perp - real_perp.item())
        contexts.append(np.array(context.cpu()))

    intermittent_save(contexts, real_perps, past_perps, filename)


def generate_datasets(context_len,
                      file='data/tokenized_stories_train_wikitext103.jbl', number=100,
                      min_len=1026, trim=True):

    # Generate objective set and training set
    # Designate the first number (100) articles that are long enough to be used
    # as our objective set, rest (that are long enough) are training data for
    # secondary learner

    data = joblib.load(file)
    print('data loaded')
    objective_set = []
    if trim:
        for i, example in enumerate(data):
            if len(example[0]) > min_len:
                start = random.randint(0, len(example[0]) - context_len - 1)
                objective_set.append(example[0, start:start + context_len])
            if len(objective_set) >= number:
                break
        train_data = [data[j] for j in range(i + 1, len(data)) if len(data[j][0]) > min_len]
    else:
        objective_set = data[0:number]
        train_data = data[number:]

    joblib.dump(objective_set, "objective_set.jbl")
    print('objective set saved')
    return train_data, objective_set


def train_secondary_learner(secondary_learner, train_dataset, max_epochs,
                            batch_size, eval_freq=50,
                            igf_model_path='secondary_learner.pt'):
    """ Train the secondary learner (igf_model) """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # We will use the first 512 pairs from our dataset as a test set for
    # our secondary learner and the rest to train
    test_dataset = train_dataset[:512]
    train_dataset = train_dataset[512:]
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    # secondary learner model set up
    loss = nn.MSELoss()
    test_loss = nn.MSELoss(reduction='sum')
    secondary_learner.to(device)
    q_optimizer = torch.optim.Adam(secondary_learner.parameters(), lr=0.00001)
    secondary_learner.train()

    # TODO in original code this is written as number of actual batches seen
    # not number of items seen but other places it is number of items instead.
    # improve consistency! changed this to epochs for clarity
    best_test_loss = float('inf')
    # Iterate through batches until we've used max_steps batches
    for epoch in range(int(max_epochs)):
        tr_q_loss = 0.0
        secondary_learner.train()
        for step, batch in enumerate(train_dataloader):
            context = batch[0].to(device)
            real_Q = batch[1].to(device)
            predicted_Q = secondary_learner(context)
            q_optimizer.zero_grad()
            q_loss = loss(predicted_Q, real_Q.float())
            q_loss.backward()
            q_optimizer.step()
            tr_q_loss += q_loss.item()

            # model trains fairly quickly so we won't wait for a full epoch
            # eval is triggered at eval_freq and end of epochs
            if (step % eval_freq == 0 and step > 0) or ((step + 1) == len(train_dataloader)):
                tr_loss = tr_q_loss / (step + 1)

                secondary_learner.eval()
                q_loss2 = 0.0
                sum_q2 = 0.0
                predicted = []
                actual = []
                # Compute performance of the secondary learner after this batch
                for step2, batch2 in enumerate(test_dataloader):
                    features2 = batch2[0].to(device)
                    real_Q2 = batch2[1].to(device)
                    predicted_Q2 = secondary_learner(features2)
                    q_loss2 += test_loss(predicted_Q2, real_Q2).item()
                    sum_q2 += torch.sum(predicted_Q2).item()
                    for ei, i in enumerate(predicted_Q2.cpu().detach().numpy()):
                        predicted.append(i.item())
                    for ei, i in enumerate(real_Q2.cpu().detach().numpy()):
                        actual.append(i.item())

                q_loss2 /= len(test_dataset)
                print('Epoch: ', epoch, 'step: ', step, 'Avg. q:', sum_q2 / len(test_dataset),
                      "Train Loss: ", tr_loss, "Test Loss: ", q_loss2)
                if q_loss2 < best_test_loss:
                    joblib.dump((predicted, actual), "pred_vs_actual.jbl")
                    torch.save(secondary_learner.state_dict(), igf_model_path)
                    best_test_loss = q_loss2

            secondary_learner.train()
    return secondary_learner


class SecondaryLearner(nn.Module):
    '''Our secondary learner'''
    def __init__(self, model):
        '''We use a simple convulotional network as our secondary learner'''
        # embeddings are from the pretrained model
        super(SecondaryLearner, self).__init__()
        self.embeddings = model.transformer.wte
        self.embeddings.weight = copy.deepcopy(model.transformer.wte.weight)
        self.conv = nn.Conv1d(self.embeddings.weight.size(1), 256, 3, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(256, 32),
            nn.Dropout(p=0.1),
            nn.Linear(32, 32),
            nn.Linear(32, 1)
        )

    def forward(self, context):
        '''Forward pass through the secondary learner'''
        pooled = torch.max(self.conv(self.embeddings(context).squeeze(1).transpose(1,2)), 2)[0]
        Qs = self.fc(pooled)
        return Qs.squeeze(1)

    @classmethod
    def from_pretrained(cls, state_path, model):
        '''Load the secondary learner'''

        secondary_learner = cls(model) # this calls __init__
        state_dict = torch.load(state_path)
        secondary_learner.load_state_dict(state_dict)
        secondary_learner.embeddings = model.transformer.wte
        secondary_learner.embeddings.weight = copy.deepcopy(model.transformer.wte.weight)
        return secondary_learner


def finetune(model, train_dataset, test_dataset, context_len,
             max_steps, batch_size, threshold, recopy_model=recopy_gpt2,
             secondary_learner=None, eval_interval=10,
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
            # print(example)
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


def finetune_eval(model, context_len, runs, max_steps,
                  recopy_model=recopy_gpt2,
                  path_to_secondary_learner='data/secondary_learner.pt',
                  eval_interval=10,
                  data_file='data/tokenized_stories_train_wikitext103.jbl',
                  number_in_test=100, min_len=1026, trim=True,
                  finetuned_model_string='gpt2_finetuned', seed=42):
    # runs multiple fine-tuning cases, first runs IGF version runs times then
    # runs standard finetuning runs times. saves evaluation perplexity for each
    # of the runs
    set_seed(seed)
    train_data, test_data = generate_datasets(context_len, data_file,
                                              number_in_test, min_len, trim)
    threshold = 1.0
    for replica in range(0, runs*2):
        set_seed(replica+1001)
        # First runs will be IGF and second will be Standard Finetuning
        use_secondary_learner = replica < runs
        # Load the secondary learner
        if use_secondary_learner:
            # loads secondary learner from path_to_secondary_learner
            secondary_learner = SecondaryLearner.from_pretrained(path_to_secondary_learner, model)
            finetuned_model_string = finetuned_model_string + '_igf_' + str(replica)
        else:
            secondary_learner = None
            finetuned_model_string = finetuned_model_string + '_' + str(replica)
        # Run for this iteration on SF/IGF
        _, qs, eval_perps, val_perps, contexts, repeats = finetune(model,
                                                                   train_data,
                                                                   test_data,
                                                                   context_len,
                                                                   max_steps, 16,
                                                                   threshold,
                                                                   recopy_model=recopy_model,
                                                                   secondary_learner=secondary_learner,
                                                                   eval_interval=eval_interval,
                                                                   finetuned_model_name=finetuned_model_string + '.pt')

        # Prefixes of saved performance metrics
        # Save run statistics
        joblib.dump(np.array(eval_perps), f'eval_perps_{finetuned_model_string}.jbl')
        print('model', finetuned_model_string, 'done with finetuning and saved')
        del secondary_learner
        torch.cuda.empty_cache()


def plot_eval_perf(xmin, xmax, path='', val=10,
                   igf_string= 'eval_perps_gpt2_finetuned_igf_*.jbl',
                   sf_string= 'eval_perps_gpt2_finetuned_*.jbl'):
    # plots eval performance comparing igf and standard finetuning (sf)
    igf_files = glob.glob(path + igf_string)
    igf_runs = []
    for f in igf_files:
        igf_runs.append(joblib.load(f))
    igf_runs = np.array(igf_runs)

    sf_files = glob.glob(path + sf_string)
    sf_runs = []
    for f in sf_files:
        sf_runs.append(joblib.load(f))
    sf_runs = np.array(sf_runs)

    print(ttest_ind(sf_runs[:, -1], igf_runs[:, -1], equal_var=False))

    xmin = max(xmin, 0)
    xmax = min(len(sf_runs[0]), xmax)
    y = np.array(list((range(xmin, xmax))))
    fig = plt.figure(figsize=(10, 10))
    plt.errorbar(val*y, np.mean( sf_runs[:,xmin:xmax], axis=0), yerr=np.std( sf_runs[:,xmin:xmax],axis=0) / np.sqrt(len( sf_runs)), label='SF WikiText')
    plt.errorbar(val*y, np.mean(igf_runs[:,xmin:xmax], axis=0), yerr=np.std(igf_runs[:,xmin:xmax],axis=0) / np.sqrt(len(igf_runs)), label='IGF WikiText')
    plt.xlabel('Batch', fontsize=24)
    plt.ylabel("Perplexity", fontsize=24)
    plt.title("Comparison of Methods:\n"\
              "Standard Finetuning vs. IGF", fontsize=30)
    plt.tick_params(axis="x", labelsize=14)
    plt.tick_params(axis="y", labelsize=14)
    plt.legend(fontsize=12)
    plt.show()

    return sf_runs, igf_runs
