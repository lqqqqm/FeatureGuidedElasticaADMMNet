"""Numerical regression for SSIM local-moment cancellation on GPU."""
import unittest
from unittest import mock
import torch
from fg_elastica_inpaint.utils import metrics

class SSIMPrecisionTests(unittest.TestCase):
    def inputs(self):
        # Nearly flat patches expose variance subtraction precision errors.
        torch.manual_seed(291)
        gt=torch.rand(2,3,64,64)*.015+.60
        pred=gt.clone();pred[...,12:51,8:55]+=.035
        return pred,gt

    def test_cpu_autocast_preserves_float32_scoring(self):
        x,y=self.inputs()
        ref=metrics.ssim_map(x.double(),y.double())
        with torch.autocast('cpu',dtype=torch.bfloat16):got=metrics.ssim_map(x,y)
        self.assertEqual(got.dtype,torch.float32)
        self.assertLess(float((got.double().mean()-ref.mean()).abs()),1e-4)

    @unittest.skipUnless(torch.cuda.is_available(),'GPU required')
    def test_gpu_matches_double_reference_and_restores_flag(self):
        x,y=self.inputs();ref=metrics.ssim_map(x.double(),y.double())
        initial=torch.backends.cudnn.allow_tf32
        try:
            for flag in [True,False]:
                torch.backends.cudnn.allow_tf32=flag
                with torch.autocast('cuda',dtype=torch.float16):got=metrics.ssim_map(x.cuda(),y.cuda())
                self.assertEqual(torch.backends.cudnn.allow_tf32,flag)
                self.assertEqual(got.dtype,torch.float32)
                self.assertLess(float((got.cpu().double().mean()-ref.mean()).abs()),1e-4)
            torch.backends.cudnn.allow_tf32=True
            with mock.patch.object(metrics,'_ssim_map_moments',side_effect=RuntimeError('test failure')):
                with self.assertRaisesRegex(RuntimeError,'test failure'):metrics.ssim_map(x.cuda(),y.cuda())
            self.assertTrue(torch.backends.cudnn.allow_tf32)
        finally:torch.backends.cudnn.allow_tf32=initial

if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
