BERT 冻结策略
def set_tuning(model, mode):
    if mode == "freeze":
        for param in model.bert.parameters():
            param.requires_grad = False

    elif mode == "partial":
        # 只训练最后2层
        for name, param in model.bert.named_parameters():
            if "encoder.layer.10" in name or "encoder.layer.11" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    elif mode == "full":
        for param in model.bert.parameters():
            param.requires_grad = True

学习率调度策略
parser.add_argument("--scheduler", default="linear", choices=["linear", "cosine"])
if args.scheduler == "linear":
    scheduler = get_linear_schedule_with_warmup(...)
else:
    scheduler = get_cosine_schedule_with_warmup(...)


optimizer 对比
parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adam"])

