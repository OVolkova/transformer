from data.code_to_text import CodeToText

if __name__ == "__main__":
    train = CodeToText("train", "../dataset/code_to_text", "../dataset/bpe_encoder")
    # test = CodeToText("test", "../dataset/code_to_text", "../dataset/bpe_encoder")
    # validation = CodeToText("validation", "../dataset/code_to_text", "../dataset/bpe_encoder")
    for k in range(10):
        print(train[k])
    # for k in range(10):
    #     print(test[k])
    # for k in range(10):
    #     print(validation[k])

    from torch.utils.data import DataLoader

    loader = DataLoader(train, batch_size=4, shuffle=True)
    for batch in loader:
        print(batch)
        break
