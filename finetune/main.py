import logging
import os
import sys
from logging.config import fileConfig

import fire
import torch
import torch.distributed
from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup, GPT2Tokenizer

# from evaluate import evaluate
import evaluate
from approaches.helpers import set_seed
from approaches.loader import load_dataloader_and_optimizers
from approaches.run_approaches import load_config, load_models_and_tokenizers
from finetune.cdg_dataloader import CDGDataLoader
from finetune.train import train

SAVED_CONFIG_FILENAME = 'training_config.ini'
SAVED_CMD_FILENAME = 'training_cmd.txt'


def main(do_train=False, local_rank=None, config_file=None, do_eval=False, **kwargs):
    logger = logging.getLogger("application.Main")
    config = load_config(config_file, kwargs=kwargs)
    config['DEFAULT']['local_rank'] = str(local_rank)
    output_dir = config["TRAINING"]["output_dir"]

    seed = int(config["DEFAULT"]["seed"])
    set_seed(seed)  # Added here for reproducibility (even between python 2 and 3)

    # Setup CUDA, GPU & distributed training
    if local_rank == -1 or config.getboolean("DEFAULT", "no_cuda", fallback=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend="nccl")
        n_gpu = 1

    if do_train:
        logger.info("Starting Training")
        if local_rank in [-1, 0]:
            # Create output directory if needed
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            with open(os.path.join(output_dir, SAVED_CONFIG_FILENAME), 'w') as config_f:
                config.write(config_f)
            with open(os.path.join(output_dir, SAVED_CMD_FILENAME), 'w') as cmd_f:
                if local_rank != -1:
                    distributed_cmd = ['-m', 'torch.distributed.launch', '--nproc_per_node',
                                       str(torch.distributed.get_world_size())]
                    argv = [arg for arg in sys.argv if 'local_rank' not in arg.replace('-', '_')]
                else:
                    distributed_cmd = []
                    argv = sys.argv
                cmd_f.write(' '.join(['python'] + distributed_cmd + argv))

        train_config = config['TRAINING']
        fp16 = train_config.getboolean("fp16", fallback=False)
        fp16_opt_level = train_config.get("fp16_opt_level", fallback="O2")

        # Load pretrained model and tokenizer
        logger.info("Loading models and preprocess_data")
        if local_rank not in [-1, 0]:
            # Barrier to make sure only the first process in distributed training download model & vocab
            torch.distributed.barrier()

        model, tokenizer = load_models_and_tokenizers(config, task="train")

        train_dataset = CDGDataLoader(config, model, tokenizer)

        train_dl, train_sampler, optimizer = load_dataloader_and_optimizers(model, train_dataset, train_config,
                                                                            local_rank, n_gpu)

        max_steps = int(train_config["max_steps"])
        warmup_steps = int(train_config["warmup_steps"])
        num_train_epochs = int(train_config["num_train_epochs"])
        gradient_accumulation_steps = float(train_config["gradient_accumulation_steps"])
        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = max_steps // (len(train_dl) // gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dl) // gradient_accumulation_steps * num_train_epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        model.to(device)

        if local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss, loss_list = train(
            config,
            train_dl,
            train_sampler,
            model,
            tokenizer,
            optimizer,
            scheduler,
            num_train_epochs,
            max_steps,
            t_total,
            gradient_accumulation_steps,
            local_rank,
            n_gpu,
            fp16=fp16,
            fp16_opt_level=fp16_opt_level,
            device=device
        )

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        if local_rank in [-1, 0]:
            logger.info("Saving model checkpoint to %s", output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(config, os.path.join(output_dir, "training_args.bin"))
    elif do_eval:
        logger.info("Starting Evaluation")
        # model, tokenizer = load_models_and_tokenizers(config, task="train")
        # evaluate(config, model, tokenizer, config['TRAINING']['output_dir'])
        evaluate.main(config_obj=config)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    fileConfig("../logger_config.conf")
    fire.Fire(main)
