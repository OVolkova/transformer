import unittest

from model.bpe import Encoder


# TODO: cover with tests properly
class TestBPE(unittest.TestCase):
    encoder = Encoder()
    data = [["hello", "world"], ["hello", "planet"], ["welcome"], ["well", "well", "well"]]
    encoder.fit(data)

    def test_encoder_init(self):
        encoder = Encoder()
        self.assertDictEqual(
            encoder.encoder, {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
        )

    def test_bpe__repeating_in_encoding(self):
        encoder = Encoder()
        s = ["helllllo", "wowowowowo", 'lamlamlamlam', 'clapclapclap']
        encoder.fit(self.data + [s])
        encoded = encoder.encode(s)
        decoded = encoder.decode(encoded)
        self.assertListEqual(decoded, s)
        self.assertEqual(len(encoded), 14)

    def test_bpe__all_tokens_present(self):
        encoded = self.encoder.encode(["hello", "world", "he"])
        decoded = self.encoder.decode(encoded)
        self.assertListEqual(decoded, ["hello", "world", "he"])
        self.assertEqual(len(encoded), 10)

    def test_bpe__repeating(self):
        encoded = self.encoder.encode(["helllllo", "wowowowowo"])
        decoded = self.encoder.decode(encoded)
        self.assertListEqual(decoded, ["helllllo", "wowowowowo"])
        self.assertEqual(len(encoded), 15)

    def test_bpe__unk_tokens(self):
        encoded = self.encoder.encode([".", "world", "he", "well", "wel", "wely"])
        decoded = self.encoder.decode(encoded)
        self.assertListEqual(decoded, ["<unk>", "world", "he", "well", "wel", "wel<unk>"])

    def test_bpe__bos_eos_pad_tokens(self):
        encoded = self.encoder.encode(
            [
                "<s>",
                "world",
                "he",
                "well",
                "wel",
                "wely",
                "</s>",
                "<pad>",
                "<pad>",
                "<pad>",
            ]
        )
        decoded = self.encoder.decode(encoded)
        self.assertListEqual(
            decoded,
            [
                "<s>",
                "world",
                "he",
                "well",
                "wel",
                "wel<unk>",
                "</s>",
                "<pad>",
                "<pad>",
                "<pad>",
            ],
        )
