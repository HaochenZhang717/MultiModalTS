
def debug():

    configs = {
        "name": "ai_readi",
        "data_path": "/Users/zhc/Documents/AI-READI",
        "window_size": 24,
        "retinal_resize": (256, 256),
        "include_target": True
    }

    print("\n====== TEST GenerationDataset ======\n")

    dataset = GenerationDataset(configs)

    loader = dataset.get_loader(
        split="train",
        text_type=None,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    batch = next(iter(loader))

    print("Batch keys:")
    print(batch.keys())

    print("\nShapes:")

    print("glucose_window:", batch["glucose_window"].shape)
    print("target:", batch["target"].shape)
    print("retinal_images:", batch["retinal_images"].shape)

    print("\nOther fields:")

    print("age:", batch["age"].shape)
    print("patient_id:", batch["patient_id"][:3])
    print("study_group:", batch["study_group"][:3])

    assert batch["retinal_images"].shape[1] == 4
    assert batch["retinal_images"].shape[2] == 3

    print("\nTEST PASSED\n")


if __name__ == "__main__":
    main()