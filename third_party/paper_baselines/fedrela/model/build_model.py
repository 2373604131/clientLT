from model.model_res import ResNet18, ResNet34


def build_model(args):
    if args.model == "resnet18":
        netglob = ResNet18(args.num_classes).to(args.device)
    elif args.model == "resnet34":
        netglob = ResNet34(args.num_classes).to(args.device)
    else:
        exit(f"Error: unsupported model '{args.model}' (use resnet18 or resnet34)")
    return netglob
