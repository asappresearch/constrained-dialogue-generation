import logging
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from finetune.evaluate import evaluate
import math

IGNORE_INDEX = -100


def train(config, train_dl, train_sampler, model, tokenizer, optimizer, scheduler, num_train_epochs, max_steps,
          total_opt_steps, gradient_accumulation_steps, local_rank, n_gpu, fp16=False, fp16_opt_level="O2",
          device="cpu"):
    logger = logging.getLogger("application.Training")

    train_config = config['TRAINING']
    train_task = config["DEFAULT"]["train_task"]

    """ Train the model """
    if local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=os.path.join(train_config["output_dir"], "tb_logs"))

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", int(train_config["per_gpu_train_batch_size"]))
        logger.info("  Total optimization steps = %d", total_opt_steps)

    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
        )

    block_method = None
    if config.get("TRAINING", "block_method", fallback=None) is not None:
        block_method = train_config["block_method"]

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    loss_list = []
    model.zero_grad()
    train_iterator = trange(int(num_train_epochs), desc="Epoch", disable=local_rank not in [-1, 0])
    # import pdb; pdb.set_trace()
    for epoch in train_iterator:
        if local_rank != -1:
            train_sampler.set_epoch(epoch)
        epoch_iterator = tqdm(train_dl, desc="Iteration", disable=local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            decoder_input_ids = None
            if block_method == "block-onehot":
                tag_vecs = batch[3].to(device)
            if train_config['training_type'] == "regular":
                if len(batch) == 3:
                    inputs, mask_customer, mask_agent = batch[:3]
                elif len(batch) == 4:
                    issueids, inputs, mask_customer, mask_agent = batch[:4]
                elif len(batch) == 5:
                    issueids, inputs, mask_customer, mask_agent, _ = batch[:5]
                else:
                    raise ValueError(
                        f"batch with only 4 items is handled. the current batch has {len(batch)} items")
            elif train_config['training_type'] == "fop":
                issueids, inputs, mask_customer, mask_agent, future_contexts = batch[:5]

                future_contexts = future_contexts.to(device)

            inputs = inputs.to(device)
            if decoder_input_ids is None:
                with torch.no_grad():
                    label_mask = mask_customer.new_zeros(mask_customer.size())
                    if "customer" in train_task:
                        label_mask += mask_customer
                    if "agent" in train_task:
                        label_mask += mask_agent
                    if train_task == "all":
                        label_mask = inputs != tokenizer.pad_token_id

                    label_mask = label_mask.to(device)
                    labels = inputs.clone().detach()
                    labels[label_mask == 0] = IGNORE_INDEX
            else:
                decoder_input_ids = decoder_input_ids.to(device)
                labels = None

            model.train()
            if block_method == "block-onehot":
                outputs = model(inputs, labels=labels, tag_vecs=tag_vecs)
            else:
                if train_config['training_type'] == "regular":
                    if decoder_input_ids is None:
                        outputs = model(inputs, labels=labels)
                    else:
                        outputs = model(input_ids=inputs, labels=decoder_input_ids)
                elif train_config['training_type'] == "fop":
                    if decoder_input_ids is None:
                        outputs = model(inputs, labels=labels, future_contexts=future_contexts,
                                        combine_type=train_config['embedding_combine_type'])
                    else:
                        outputs = model(input_ids=inputs, labels=decoder_input_ids, future_contexts=future_contexts,
                                        combine_type=train_config['embedding_combine_type'])

            if isinstance(outputs, list) or isinstance(outputs, tuple):
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            elif isinstance(outputs, dict):
                loss = outputs['loss']
            else:
                logger.error(f"Model output of type {type(outputs)} is not handled")
                assert False

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            max_grad_norm = float(train_config["max_grad_norm"])
            logging_steps = int(train_config["logging_steps"])

            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                if fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                loss_list.append(tr_loss / global_step)

                if local_rank in [-1, 0] and logging_steps > 0 and global_step % logging_steps == 0:
                    # Log metrics
                    log_loss = (tr_loss - logging_loss) / logging_steps
                    lr = scheduler.get_lr()[0]
                    tb_writer.add_scalar("lr", lr, global_step)
                    tb_writer.add_scalar("loss", log_loss, global_step)
                    tb_writer.add_scalar("ppl", math.exp(log_loss), global_step)
                    logger.info(f"train step: {global_step} loss: {log_loss} ppl: {math.exp(log_loss)} lr: {lr}")
                    logging_loss = tr_loss

                save_steps = int(train_config["save_steps"])

                if local_rank in [-1, 0] and save_steps > 0 and global_step % save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(train_config["output_dir"], "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(config, os.path.join(output_dir, "training_config.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            # cleanup after each epoch to clear memory
            del inputs
            if decoder_input_ids is not None:
                del decoder_input_ids
                del label_mask
                del labels
            del mask_customer
            del mask_agent
            del outputs
            torch.cuda.empty_cache()

            if 0 < max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < max_steps < global_step:
            train_iterator.close()
            break

    if local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, loss_list
