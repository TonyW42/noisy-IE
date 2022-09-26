from data import character_level_wnut, tokenize_for_char_manual, tokenize_and_align_labels, tokenize_for_char
from utils.compute import compute_metrics

def hugging_face_model(args):
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=14)
    wnut = load_dataset("wnut_17")
    wnut_character_level = character_level_wnut(wnut)

    use_old_tok = ["xlm-roberta-base", "xlm-roberta-large"]
    use_new_tok = ["google/canine-s"]
    if args.model_name in use_old_tok:
        tokenized_wnut = wnut_character_level.map(tokenize_and_align_labels, batched=True)
    if args.model_name in use_new_tok:
        try:
            tokenized_wnut = wnut_character_level.map(tokenize_for_char, batched=True)
        except:
            tokenized_wnut = tokenize_for_char_manual(wnut_character_level)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                            add_prefix_space=args.prefix_space) ## changed here
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
                                        

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        num_train_epochs=args.n_epochs,
        # weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_wnut["train"],
        eval_dataset=tokenized_wnut["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        # optimizers = torch.optim.Adam(model.parameters()),
        compute_metrics = compute_metrics, 
    )

    trainer.train()