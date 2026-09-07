import unittest

import torch

from fg_elastica_inpaint.models.operators import div, grad
from fg_elastica_inpaint.models.unfolding import solve_p_shrink


class StructureCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_prior_creates_signed_rgb_gradients_only_in_holes(self):
        u = torch.zeros(1, 3, 3, 4, dtype=torch.float64)
        z = torch.zeros(1, 6, 3, 4, dtype=torch.float64)
        g = z.clone()
        g[:, 0], g[:, 1], g[:, 2] = .8, -.4, .2
        g.requires_grad_()
        mask = torch.zeros(1, 1, 3, 4, dtype=torch.float64)
        mask[..., 0] = 1
        result = solve_p_shrink(u, z, z, u, z, .1, 0., 1., 2.,
                                structure_gradient=g, M=mask, rho_s=4.)
        expected = z.clone()
        expected[:, 0, :, 1:] = .35
        expected[:, 1, :, 1:] = -1 / 12
        torch.testing.assert_close(result, expected)
        result.sum().backward()
        self.assertTrue(torch.isfinite(g.grad).all())
        self.assertGreater(g.grad[:, 0, :, 1:].abs().sum().item(), 0)
        self.assertEqual(g.grad[..., 0].abs().sum().item(), 0)

    def test_zero_coupling_is_exact_baseline(self):
        torch.manual_seed(2)
        u = torch.randn(1, 3, 4, 5)
        fields = [torch.randn(1, 6, 4, 5) for _ in range(3)]
        args = (u, fields[0], fields[1], torch.rand_like(u), fields[2], .1, .2, 1., 2.)
        baseline = solve_p_shrink(*args)
        disabled = solve_p_shrink(*args, structure_gradient=fields[0], M=torch.zeros(1, 1, 4, 5), rho_s=0.)
        self.assertTrue(torch.equal(baseline, disabled))

    def test_pcg_matches_dense_system_and_backpropagates(self):
        from fg_elastica_inpaint.models.solvers import solve_u_pcg
        torch.manual_seed(3)
        h, w = 3, 4
        mask = torch.zeros(1, 1, h, w, dtype=torch.float64)
        mask[..., 0] = 1
        u = torch.zeros(1, 3, h, w, dtype=torch.float64)
        p = torch.randn(1, 6, h, w, dtype=torch.float64, requires_grad=True)
        image = torch.randn_like(u)
        basis = torch.eye(h*w, dtype=u.dtype).reshape(h*w, 1, h, w)
        matrix = (-2*div(grad(basis))+10*mask*basis).reshape(h*w, h*w).T
        rhs = (10*mask*image-div(2*p)).reshape(3, h*w).T
        exact = torch.linalg.solve(matrix, rhs).T.reshape_as(u)
        actual, info = solve_u_pcg(u, p, torch.zeros_like(p), image, mask, 2., 10.,
                                  iterations=40, tolerance=1e-10)
        torch.testing.assert_close(actual, exact, atol=1e-8, rtol=1e-8)
        self.assertLess(info['relative_residual'].max().item(), 1e-8)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        actual_grad = p.grad.clone()
        p.grad = None
        exact.square().mean().backward()
        torch.testing.assert_close(actual_grad, p.grad, atol=1e-7, rtol=1e-6)

    def test_pcg_zero_rhs_has_finite_backward(self):
        from fg_elastica_inpaint.models.solvers import solve_u_pcg
        u = torch.zeros(1, 3, 4, 5, requires_grad=True)
        p = torch.zeros(1, 6, 4, 5, requires_grad=True)
        result, info = solve_u_pcg(u, p, torch.zeros_like(p), u, torch.ones(1, 1, 4, 5), 2., 10.)
        result.sum().backward()
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.isfinite(u.grad).all())
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertEqual(info['relative_residual'].max().item(), 0)

    def test_implicit_gradient_is_invariant_to_loss_scaling(self):
        from fg_elastica_inpaint.models.solvers import solve_u_pcg
        torch.manual_seed(20)
        u = torch.zeros(1, 3, 8, 8, dtype=torch.float64)
        p = torch.randn(1, 6, 8, 8, dtype=torch.float64, requires_grad=True)
        mask = torch.ones_like(u[:, :1]); mask[..., 1:7, 1:7] = 0
        output, info = solve_u_pcg(u, p, p*0, u, mask, 2., 10.,
                                   iterations=100, backward_iterations=100, tolerance=1e-8)
        upstream = torch.randn_like(u)*1e-3
        first, = torch.autograd.grad(output, p, upstream, retain_graph=True)
        scaled, = torch.autograd.grad(output, p, upstream*1e-7)
        torch.testing.assert_close(first, scaled/1e-7, atol=1e-10, rtol=1e-6)
        self.assertTrue(bool(info["backward_solved"]))
        self.assertLess(float(info["backward_relative_residual"].max()), 1e-7)


if __name__ == '__main__':
    unittest.main()
