import unittest

import torch

from spai.models import utils


class TestUtils(unittest.TestCase):

    def test_exportable_std(self) -> None:
        sizes: list[tuple[int, ...]] = [
            (5, 32, 1024),
            (1, 32, 1024),
            (5, 1, 1024),
        ]

        for s in sizes:
            x: torch.Tensor = torch.randn(s)
            pytorch_std: torch.Tensor = x.std(dim=-1)
            exportable_std: torch.Tensor = utils.exportable_std(x, dim=-1)
            self.assertTrue(torch.allclose(pytorch_std, exportable_std))
